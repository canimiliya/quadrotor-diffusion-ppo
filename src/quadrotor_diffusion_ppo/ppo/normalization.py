"""Frozen TRAIN-only observation standardization for the S4-R2 PPO policy."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


STD_FLOOR = 1.0e-6
CONSTANT_FEATURE_SCALE = 1.0


@dataclass(frozen=True)
class FixedObservationStatistics:
    mean: np.ndarray
    scale: np.ndarray
    raw_std: np.ndarray
    constant_mask: np.ndarray
    source_sha256: str
    sample_count: int

    def contract(self) -> dict:
        return {
            "method": "(observation - train_mean) / train_scale",
            "source": "artifacts/s2/dataset/train.npz::observations",
            "source_sha256": self.source_sha256,
            "sample_count": int(self.sample_count),
            "dimension": int(self.mean.size),
            "statistics_dtype": "float64",
            "runtime_dtype": "float32",
            "std_floor": STD_FLOOR,
            "constant_feature_scale": CONSTANT_FEATURE_SCALE,
            "constant_feature_indices": np.flatnonzero(self.constant_mask).astype(int).tolist(),
            "mean": self.mean.tolist(),
            "raw_std": self.raw_std.tolist(),
            "scale": self.scale.tolist(),
        }

    @property
    def contract_sha256(self) -> str:
        payload = json.dumps(self.contract(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_train_statistics(train_npz: Path) -> FixedObservationStatistics:
    """Derive immutable statistics from S2 TRAIN observations only."""
    with np.load(train_npz, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] != 34:
        raise ValueError(f"expected TRAIN observations with shape (N, 34), got {observations.shape}")
    if not np.isfinite(observations).all():
        raise FloatingPointError("S2 TRAIN observations contain nonfinite values")
    mean = observations.mean(axis=0, dtype=np.float64)
    raw_std = observations.std(axis=0, dtype=np.float64)
    constant_mask = raw_std < STD_FLOOR
    scale = raw_std.copy()
    scale[constant_mask] = CONSTANT_FEATURE_SCALE
    return FixedObservationStatistics(
        mean=mean,
        scale=scale,
        raw_std=raw_std,
        constant_mask=constant_mask,
        source_sha256=sha256_file(train_npz),
        sample_count=observations.shape[0],
    )


class FixedObservationStandardizer(BaseFeaturesExtractor):
    """34D-to-34D, checkpoint-persistent, non-updating policy preprocessing."""

    def __init__(self, observation_space: spaces.Box, mean: list[float], scale: list[float]):
        if observation_space.shape != (34,):
            raise ValueError(f"expected 34D observation space, got {observation_space.shape}")
        super().__init__(observation_space, features_dim=34)
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        if mean_tensor.shape != (34,) or scale_tensor.shape != (34,):
            raise ValueError("normalization mean and scale must both be 34D")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(scale_tensor).all():
            raise ValueError("normalization statistics must be finite")
        if not torch.all(scale_tensor > 0):
            raise ValueError("normalization scale must be strictly positive")
        self.register_buffer("fixed_mean", mean_tensor)
        self.register_buffer("fixed_scale", scale_tensor)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        standardized = (observations - self.fixed_mean) / self.fixed_scale
        if not torch.isfinite(standardized).all():
            raise FloatingPointError("nonfinite fixed-standardized observation")
        return standardized
