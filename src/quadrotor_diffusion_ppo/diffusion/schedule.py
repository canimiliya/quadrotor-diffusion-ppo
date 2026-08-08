"""Fixed cosine DDPM schedule and deterministic DDIM sampler."""
from __future__ import annotations

import math

import torch

from .model import unit_ball_squash


def cosine_betas(steps: int = 100, s: float = 0.008) -> torch.Tensor:
    positions = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    alpha_bar = torch.cos(((positions + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class DiffusionSchedule:
    def __init__(self, steps: int = 100):
        self.steps = int(steps)
        self.betas = cosine_betas(self.steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].reshape(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def predict_x0(self, noisy: torch.Tensor, predicted_noise: torch.Tensor,
                   timestep: int) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timestep]
        return (noisy - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()

    @torch.no_grad()
    def ddim_sample(self, model, observation: torch.Tensor, *, steps: int = 10,
                    generator: torch.Generator | None = None) -> torch.Tensor:
        if steps != 10:
            raise ValueError("S3 freezes deterministic DDIM inference to 10 steps")
        batch = observation.shape[0]
        sample = torch.randn((batch, model.horizon, model.action_dim), device=observation.device,
                             generator=generator)
        schedule = torch.linspace(self.steps - 1, 0, steps, device=observation.device).round().long()
        for index, timestep in enumerate(schedule):
            t = int(timestep.item())
            t_batch = torch.full((batch,), t, device=observation.device, dtype=torch.long)
            predicted_noise = model(sample, observation, t_batch)
            x0 = self.predict_x0(sample, predicted_noise, t)
            if index == len(schedule) - 1:
                sample = x0
            else:
                next_t = int(schedule[index + 1].item())
                alpha_next = self.alpha_bars[next_t]
                sample = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * predicted_noise
        return sample

    @torch.no_grad()
    def ddim_sample_x0(self, model, observation: torch.Tensor, *, steps: int = 10,
                       generator: torch.Generator | None = None,
                       return_diagnostics: bool = False):
        """Deterministic DDIM for a model whose target is bounded clean action x0."""
        if steps != 10:
            raise ValueError("S3 freezes deterministic DDIM inference to 10 steps")
        batch = observation.shape[0]
        sample = torch.randn((batch, model.horizon, model.action_dim), device=observation.device,
                             generator=generator)
        schedule = torch.linspace(self.steps - 1, 0, steps, device=observation.device).round().long()
        latent_abs = 0.0
        latent_norm = 0.0
        support_violations = 0
        for index, timestep in enumerate(schedule):
            t = int(timestep.item())
            t_batch = torch.full((batch,), t, device=observation.device, dtype=torch.long)
            raw = model.predict_raw(sample, observation, t_batch)
            latent_abs = max(latent_abs, float(torch.max(torch.abs(raw)).item()))
            latent_norm = max(latent_norm, float(torch.max(torch.linalg.vector_norm(raw, dim=-1)).item()))
            x0 = unit_ball_squash(raw)
            norms = torch.linalg.vector_norm(x0, dim=-1)
            support_violations += int((norms > 1.0 + 1e-6).sum().item())
            epsilon = (sample - self.alpha_bars[t].sqrt() * x0) / (1.0 - self.alpha_bars[t]).sqrt()
            if index == len(schedule) - 1:
                sample = x0
            else:
                next_t = int(schedule[index + 1].item())
                alpha_next = self.alpha_bars[next_t]
                sample = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * epsilon
        if return_diagnostics:
            return sample, {"latent_max_abs": latent_abs, "latent_max_norm": latent_norm,
                            "support_violation_count": support_violations,
                            "support_violation_fraction": support_violations / max(1, batch * model.horizon * steps)}
        return sample
