"""Unit-ball squashed Gaussian policy distribution for the Pure PPO baseline.

The physical action contract is the open three-dimensional unit ball.  This
module keeps the Gaussian latent variable explicit and applies the radial map

    a = u * tanh(||u||) / ||u||

including its change-of-variables correction in ``log_prob``.  The class is a
subclass of SB3's diagonal Gaussian distribution so it can be used by the
unchanged PPO algorithm and ActorCriticPolicy rollout/evaluation paths.
"""
from __future__ import annotations

import math

import torch as th
from torch.distributions import Normal

from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy


class UnitBallSquashedGaussianDistribution(DiagGaussianDistribution):
    """A diagonal Gaussian transformed onto the open 3-D unit ball."""

    EPS = 1.0e-6
    SERIES_EPS = 1.0e-4
    RHO_MAX = 1.0 - 1.0e-6

    def __init__(self, action_dim: int):
        if int(action_dim) != 3:
            raise ValueError(f"UnitBallSquashedGaussianDistribution requires action_dim=3, got {action_dim}")
        super().__init__(int(action_dim))
        self.last_gaussian_latent: th.Tensor | None = None
        self.last_policy_action: th.Tensor | None = None

    @classmethod
    def squash(cls, latent: th.Tensor) -> th.Tensor:
        """Map Gaussian latent vectors to the open unit ball."""
        radius = th.linalg.vector_norm(latent, dim=-1, keepdim=True)
        radius_safe = radius.clamp_min(cls.EPS)
        # Keep the floating-point representation strictly inside the support
        # even when tanh(r) rounds to one for very large latent radii.
        rho = th.tanh(radius_safe).clamp(max=cls.RHO_MAX)
        scale = rho / radius_safe
        action = latent * scale
        return action

    @classmethod
    def unsquash(cls, action: th.Tensor) -> th.Tensor:
        """Invert ``squash`` for actions in the open unit ball."""
        rho = th.linalg.vector_norm(action, dim=-1, keepdim=True)
        rho_bounded = rho.clamp(max=1.0 - cls.EPS)
        radius = th.atanh(rho_bounded)
        rho_safe = rho.clamp_min(cls.EPS)
        return action * (radius / rho_safe)

    @classmethod
    def log_abs_det_jacobian(cls, latent: th.Tensor) -> th.Tensor:
        """Return log|det(da/du)| for the radial transform."""
        radius = th.linalg.vector_norm(latent, dim=-1)
        radius_safe = radius.clamp_min(cls.EPS)

        # log(sech^2(r)) is evaluated through log(cosh) to remain finite for
        # large latent radii.  The ratio term uses a Taylor series at zero;
        # this also avoids 0 * log(0) autograd paths.
        log_sech2 = -2.0 * (th.nn.functional.softplus(2.0 * radius) - radius - math.log(2.0))
        ratio = th.log(th.tanh(radius_safe)) - th.log(radius_safe)
        radius_sq = radius * radius
        ratio_series = -radius_sq / 3.0 + 7.0 * radius_sq * radius_sq / 90.0
        ratio = th.where(radius < cls.SERIES_EPS, ratio_series, ratio)
        return log_sech2 + 2.0 * ratio

    def proba_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> "UnitBallSquashedGaussianDistribution":
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def sample(self) -> th.Tensor:
        latent = self.distribution.rsample()
        action = self.squash(latent)
        self.last_gaussian_latent = latent
        self.last_policy_action = action
        return action

    def mode(self) -> th.Tensor:
        latent = self.distribution.mean
        action = self.squash(latent)
        self.last_gaussian_latent = latent
        self.last_policy_action = action
        return action

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        latent = self.unsquash(actions)
        log_det = self.log_abs_det_jacobian(latent)
        return sum_independent_dims(self.distribution.log_prob(latent)) - log_det

    def entropy(self) -> th.Tensor | None:
        # SB3's PPO has a supported log-probability fallback.  ent_coef is
        # frozen at zero for S4-R1, so no approximate entropy term is used.
        return None


class UnitBallActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy using the unit-ball transformed distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ActorCriticPolicy already built the same mean/log-std parameter pair
        # for a diagonal Gaussian.  Replacing only the distribution preserves
        # the frozen network and optimizer contract while routing forward(),
        # predict(), and evaluate_actions() through the radial transform.
        self.action_dist = UnitBallSquashedGaussianDistribution(self.action_space.shape[0])


def policy_action_support_violation(actions: th.Tensor, tolerance: float = 1.0e-6) -> th.Tensor:
    """Boolean mask for actions outside the physical unit-ball support."""
    return th.linalg.vector_norm(actions, dim=-1) > (1.0 + tolerance)
