"""Frozen matched-BC prior and residual composition adapter."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from quadrotor_diffusion_ppo.bc.model import MatchedSequenceBC


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenBCPrior:
    """Execute the immutable deterministic BC policy using receding horizon."""

    def __init__(self, checkpoint: Path, device: torch.device | str = "cuda"):
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.device = torch.device(device)
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        required = {"model_frozen": True, "horizon": 16, "action_dim": 3,
                    "action_support": "per_step_unit_ball_squash",
                    "deployment": "receding_horizon_first_action"}
        if any(payload.get(key) != value for key, value in required.items()):
            raise RuntimeError("BLOCKED_S7_BC_CHECKPOINT_METADATA")
        self.model = MatchedSequenceBC(**payload["model_config"]).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval().requires_grad_(False)
        self.mean = torch.as_tensor(payload["observation_mean"], dtype=torch.float32, device=self.device)
        self.scale = torch.clamp(torch.as_tensor(payload["observation_std"], dtype=torch.float32,
                                                 device=self.device), min=1e-6)
        self.payload = payload

    @torch.inference_mode()
    def predict_sequences(self, observations: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        normalized = (values - self.mean) / self.scale
        result = self.model(normalized)
        if not torch.isfinite(result).all():
            raise FloatingPointError("nonfinite BC output")
        return result.cpu().numpy().astype(np.float32)

    def predict(self, observations: np.ndarray, seeds: Sequence[int] | None = None) -> np.ndarray:
        del seeds
        return self.predict_sequences(observations)[:, 0]

    @property
    def frozen(self) -> bool:
        return not self.model.training and all(not p.requires_grad for p in self.model.parameters())
