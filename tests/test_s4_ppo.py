from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from quadrotor_diffusion_ppo.envs.action import map_normalized_velocity
from quadrotor_diffusion_ppo.ppo.contract import (
    ACTION_DIM, OBSERVATION_DIM, PPO_CONFIG, PPO_CONFIG_HASH, REWARD_CONTRACT_HASH,
    compute_reward,
)
from quadrotor_diffusion_ppo.ppo.unit_ball import (
    UnitBallActorCriticPolicy,
    UnitBallSquashedGaussianDistribution,
)


def test_reward_contract():
    assert compute_reward(2.0, 1.0)[0] == 1.998
    assert compute_reward(1.0, 2.0)[0] == -2.002
    assert compute_reward(1.0, 1.0)[0] == -0.002
    assert compute_reward(1.0, 1.0, success=True)[0] == 19.998
    assert compute_reward(1.0, 1.0, collision=True)[0] == -20.002
    assert compute_reward(1.0, 1.0, ground=True)[0] == -20.002


def test_ppo_observation_is_34d():
    assert OBSERVATION_DIM == 34
    assert ACTION_DIM == 3


def test_ppo_action_uses_shared_mapping():
    raw = np.asarray([1.0, 1.0, 0.0])
    mapped = map_normalized_velocity(raw, 0.801)
    assert mapped.clipped
    assert np.linalg.norm(mapped.post_clipping) <= 1.0 + 1e-12


def test_frozen_config_hash_is_stable():
    payload = json.dumps(PPO_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert PPO_CONFIG_HASH == hashlib.sha256(payload).hexdigest()
    assert len(REWARD_CONTRACT_HASH) == 64


def test_s4_runtime_has_no_privileged_dependency_imports():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_s4_ppo.py"
    text = script.read_text(encoding="utf-8").lower()
    assert "stable_baselines3" in text
    assert "diffusion.model" not in text
    assert "s2_dataset" not in text


def test_unit_ball_policy_support():
    latent = torch.cat((torch.zeros(1, 3), torch.randn(1024, 3) * 8.0), dim=0)
    action = UnitBallSquashedGaussianDistribution.squash(latent)
    assert torch.isfinite(action).all()
    assert torch.linalg.vector_norm(action, dim=-1).max().item() < 1.0


def test_unit_ball_forward_inverse():
    latent = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.4, 0.7], [2.0, -1.0, 0.5]])
    action = UnitBallSquashedGaussianDistribution.squash(latent)
    recovered = UnitBallSquashedGaussianDistribution.unsquash(action)
    assert torch.allclose(recovered, latent, atol=2.0e-5, rtol=2.0e-5)


def test_unit_ball_jacobian_matches_autograd():
    for values in ([0.15, -0.20, 0.35], [0.8, 0.4, -0.6], [1.4, -0.5, 0.2]):
        latent = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(UnitBallSquashedGaussianDistribution.squash, latent)
        numerical = torch.log(torch.abs(torch.linalg.det(jacobian)))
        analytic = UnitBallSquashedGaussianDistribution.log_abs_det_jacobian(latent)
        assert torch.allclose(numerical, analytic, atol=2.0e-8, rtol=2.0e-8)


def test_unit_ball_log_prob_finite_and_repeatable():
    distribution = UnitBallSquashedGaussianDistribution(3)
    mean = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    log_std = torch.tensor([-0.4, -0.2, 0.1])
    distribution.proba_distribution(mean, log_std)
    actions = distribution.sample().detach()
    first = distribution.log_prob(actions)
    second = distribution.log_prob(actions)
    assert first.shape == (2,)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_policy_action_equals_executed_action_without_projection():
    latent = torch.randn(256, 3)
    actions = UnitBallSquashedGaussianDistribution.squash(latent).detach().numpy()
    for action in actions:
        mapped = map_normalized_velocity(action, 0.801)
        assert not mapped.clipped
        assert np.allclose(action, mapped.post_clipping, atol=1.0e-6)


def test_sb3_rollout_uses_transformed_action():
    policy = UnitBallActorCriticPolicy(
        spaces.Box(-np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)),
        spaces.Box(-np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        lambda _: 3.0e-4,
        net_arch={"pi": [8, 8], "vf": [8, 8]},
    )
    actions, values, log_prob = policy(torch.zeros((16, 4)), deterministic=False)
    assert actions.shape == (16, 3)
    assert values.shape == (16, 1)
    assert log_prob.shape == (16,)
    assert torch.isfinite(log_prob).all()
    assert torch.linalg.vector_norm(actions, dim=-1).max().item() < 1.0


def test_sb3_evaluate_actions_logprob_consistency():
    policy = UnitBallActorCriticPolicy(
        spaces.Box(-np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)),
        spaces.Box(-np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        lambda _: 3.0e-4,
        net_arch={"pi": [8, 8], "vf": [8, 8]},
    )
    observations = torch.randn((10, 4))
    actions, _, forward_log_prob = policy(observations, deterministic=False)
    _, evaluated_log_prob, entropy = policy.evaluate_actions(observations, actions)
    assert entropy is None
    assert torch.allclose(forward_log_prob, evaluated_log_prob, atol=2.0e-5, rtol=2.0e-5)


def test_reward_hash_unchanged_from_r0():
    assert REWARD_CONTRACT_HASH == "92afefbf338d0ccbbcc42c12a572327413ca66f46bf1f061f637e68d125ae3ec"


def test_ppo_hyperparameters_unchanged():
    assert PPO_CONFIG["learning_rate"] == 3.0e-4
    assert PPO_CONFIG["n_steps"] == 1024
    assert PPO_CONFIG["batch_size"] == 512
    assert PPO_CONFIG["n_epochs"] == 10
    assert PPO_CONFIG["gamma"] == 0.99
    assert PPO_CONFIG["gae_lambda"] == 0.95
    assert PPO_CONFIG["clip_range"] == 0.20
    assert PPO_CONFIG["ent_coef"] == 0.0
    assert PPO_CONFIG["vf_coef"] == 0.5
    assert PPO_CONFIG["max_grad_norm"] == 0.5
    assert PPO_CONFIG["normalize_advantage"] is True
    assert PPO_CONFIG["n_envs"] == 8
    assert PPO_CONFIG["total_env_steps"] == 500_000


def test_train_val_test_contract_unchanged():
    manifest = Path(__file__).resolve().parents[1] / "artifacts" / "s2" / "task_manifest.csv"
    rows = manifest.read_text(encoding="utf-8-sig").splitlines()[1:]
    counts = {split: sum(1 for row in rows if f",{split}," in row) for split in ("train", "val", "test")}
    assert counts == {"train": 252, "val": 54, "test": 54}


def test_diffusion_and_teacher_dependency_absent():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_s4_ppo.py").read_text(encoding="utf-8").lower()
    assert "from quadrotor_diffusion_ppo.diffusion" not in source
    assert "import quadrotor_diffusion_ppo.diffusion" not in source
    assert "from quadrotor_diffusion_ppo.expert" not in source
    assert "import quadrotor_diffusion_ppo.expert" not in source


def test_test_after_val_gate_only():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_s4_ppo.py").read_text(encoding="utf-8")
    assert "if val_gate:" in source
    assert source.index("if val_gate:") < source.index("evaluate(best_model, tasks[\"test\"]")
