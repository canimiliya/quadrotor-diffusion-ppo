"""Conditional temporal 1-D U-Net denoiser for the frozen S8-R4 contract.

The network deliberately keeps the observation/action/timestep interfaces of
``ConditionalDiffusionMLP`` while making the temporal axis explicit.  It
predicts bounded clean actions (x0), not epsilon or v.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .model import SinusoidalTimeEmbedding, unit_ball_squash


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1d layers with FiLM modulation from a global condition."""

    def __init__(self, in_channels: int, out_channels: int, global_dim: int = 512,
                 groups: int = 8):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("channels must be divisible by GroupNorm groups")
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.condition = nn.Linear(global_dim, 2 * out_channels)
        self.activation = nn.SiLU()
        self.residual = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, value: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual(value)
        hidden = self.activation(self.norm1(self.conv1(value)))
        scale, bias = self.condition(global_cond).unsqueeze(-1).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale) + bias
        hidden = self.activation(self.norm2(self.conv2(hidden)))
        return hidden + residual


class ConditionalUnet1D(nn.Module):
    """S8-R4 conditional U-Net with channels 256 -> 512 -> 1024."""

    def __init__(self, observation_dim: int = 34, horizon: int = 16,
                 action_dim: int = 3, condition_dim: int = 256,
                 bounded_output: bool = True):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)
        self.bounded_output = bool(bounded_output)
        self.observation_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self.time_embedding = SinusoidalTimeEmbedding(256)
        self.time_mlp = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.input_projection = nn.Conv1d(self.action_dim, 256, kernel_size=1)

        self.down0 = nn.ModuleList([
            ConditionalResidualBlock1D(256, 256), ConditionalResidualBlock1D(256, 256)
        ])
        self.downsample0 = nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1)
        self.down1 = nn.ModuleList([
            ConditionalResidualBlock1D(512, 512), ConditionalResidualBlock1D(512, 512)
        ])
        self.downsample1 = nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.bottleneck = nn.ModuleList([
            ConditionalResidualBlock1D(1024, 1024), ConditionalResidualBlock1D(1024, 1024)
        ])

        self.upsample1 = nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ConditionalResidualBlock1D(1024, 512), ConditionalResidualBlock1D(512, 512)
        ])
        self.upsample0 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up0 = nn.ModuleList([
            ConditionalResidualBlock1D(512, 256), ConditionalResidualBlock1D(256, 256)
        ])
        self.output_norm = nn.GroupNorm(8, 256)
        self.output_activation = nn.SiLU()
        self.output_projection = nn.Conv1d(256, self.action_dim, kernel_size=1)

    @staticmethod
    def _run_blocks(value: torch.Tensor, blocks: nn.ModuleList,
                    global_cond: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            value = block(value, global_cond)
        return value

    def predict_raw(self, noisy_actions: torch.Tensor, observation: torch.Tensor,
                    timesteps: torch.Tensor) -> torch.Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"noisy_actions must have shape (B,{self.horizon},{self.action_dim})")
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"observation must have shape (B,{self.observation_dim})")
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_actions.shape[0]:
            raise ValueError("timesteps must have shape (B,)")
        global_cond = torch.cat((self.observation_encoder(observation),
                                 self.time_mlp(self.time_embedding(timesteps))), dim=1)
        value = self.input_projection(noisy_actions.transpose(1, 2))
        skip16 = self._run_blocks(value, self.down0, global_cond)
        value = self.downsample0(skip16)
        skip8 = self._run_blocks(value, self.down1, global_cond)
        value = self.downsample1(skip8)
        value = self._run_blocks(value, self.bottleneck, global_cond)
        value = self.upsample1(value)
        value = self._run_blocks(torch.cat((value, skip8), dim=1), self.up1, global_cond)
        value = self.upsample0(value)
        value = self._run_blocks(torch.cat((value, skip16), dim=1), self.up0, global_cond)
        value = self.output_projection(self.output_activation(self.output_norm(value)))
        return value.transpose(1, 2)

    def forward(self, noisy_actions: torch.Tensor, observation: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        raw = self.predict_raw(noisy_actions, observation, timesteps)
        return unit_ball_squash(raw) if self.bounded_output else raw

