"""Deterministic H=16 BC baseline matched to the diffusion denoiser scale."""
from __future__ import annotations

import torch
from torch import nn

from quadrotor_diffusion_ppo.diffusion.model import ResidualMLPBlock, unit_ball_squash


class MatchedSequenceBC(nn.Module):
    """Map one 34-D observation to a bounded 16x3 action sequence."""

    def __init__(self, observation_dim: int = 34, horizon: int = 16,
                 action_dim: int = 3, observation_width: int = 128,
                 width: int = 256, residual_blocks: int = 4):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.observation_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, observation_width), nn.SiLU(),
            nn.Linear(observation_width, observation_width), nn.SiLU(),
        )
        self.input_projection = nn.Sequential(nn.Linear(observation_width, width), nn.SiLU())
        self.residual_core = nn.Sequential(*(ResidualMLPBlock(width) for _ in range(residual_blocks)))
        self.output_projection = nn.Linear(width, self.horizon * self.action_dim)

    def predict_raw(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"observation must have shape (B,{self.observation_dim})")
        hidden = self.input_projection(self.observation_encoder(observation))
        hidden = self.residual_core(hidden)
        return self.output_projection(hidden).reshape(-1, self.horizon, self.action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return unit_ball_squash(self.predict_raw(observation))
