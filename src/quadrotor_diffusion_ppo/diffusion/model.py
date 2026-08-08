"""Small conditional diffusion model used by the S3 sanity experiment."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 64):
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        if self.dimension % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class ConditionalDiffusionMLP(nn.Module):
    """34D observation conditioned epsilon predictor for 16x3 actions."""

    def __init__(self, observation_dim: int = 34, horizon: int = 16, action_dim: int = 3,
                 observation_width: int = 128, time_dim: int = 64, width: int = 256,
                 residual_blocks: int = 4):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.time_dim = int(time_dim)
        self.observation_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, observation_width), nn.SiLU(),
            nn.Linear(observation_width, observation_width), nn.SiLU(),
        )
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(observation_width + time_dim + horizon * action_dim, width), nn.SiLU()
        )
        self.residual_core = nn.Sequential(*(ResidualMLPBlock(width) for _ in range(residual_blocks)))
        self.output_projection = nn.Linear(width, horizon * action_dim)

    def forward(self, noisy_actions: torch.Tensor, observation: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"noisy_actions must have shape (B,{self.horizon},{self.action_dim})")
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"observation must have shape (B,{self.observation_dim})")
        condition = self.observation_encoder(observation)
        time = self.time_embedding(timesteps)
        noisy = noisy_actions.reshape(noisy_actions.shape[0], -1)
        hidden = self.input_projection(torch.cat((condition, time, noisy), dim=1))
        hidden = self.residual_core(hidden)
        return self.output_projection(hidden).reshape(-1, self.horizon, self.action_dim)
