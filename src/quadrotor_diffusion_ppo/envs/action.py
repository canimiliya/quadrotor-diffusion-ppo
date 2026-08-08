"""The single shared M0 normalized 3-D velocity action mapping."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VelocityAction:
    raw_normalized: np.ndarray
    post_clipping: np.ndarray
    requested_direction: np.ndarray
    speed_ratio: float
    velocity_aviary_command: np.ndarray
    clipped: bool
    speed_limit_excess: float


def map_normalized_velocity(action: np.ndarray, speed_limit_m_per_s: float) -> VelocityAction:
    """Map a world-frame normalized 3-D velocity to VelocityAviary's 4-D path.

    The input is always the shared PPO/Diffusion/GCOPTER contract. Uniform
    unit-ball clipping preserves direction and makes the fourth legacy
    VelocityAviary field a speed ratio, never an expert motor label.
    """
    raw = np.asarray(action, dtype=float)
    if raw.shape != (3,):
        raise ValueError(f"normalized velocity action must have shape (3,), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("normalized velocity action must be finite")
    limit = float(speed_limit_m_per_s)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("speed limit must be finite and positive")
    norm = float(np.linalg.norm(raw))
    clipped = norm > 1.0 + 1.0e-6
    post = raw / norm if clipped and norm > 0.0 else raw.copy()
    post_norm = float(np.linalg.norm(post))
    direction = np.zeros(3, dtype=float) if post_norm == 0.0 else post / post_norm
    command = np.concatenate((direction, np.asarray([post_norm], dtype=float)))
    return VelocityAction(raw_normalized=raw.copy(), post_clipping=post,
                          requested_direction=direction, speed_ratio=post_norm,
                          velocity_aviary_command=command, clipped=clipped,
                          speed_limit_excess=max(0.0, (norm - 1.0) * limit))


def velocity_reference_to_action(v_ref: np.ndarray, speed_limit_m_per_s: float) -> VelocityAction:
    """Convert analytic GCOPTER world velocity directly to the shared action."""
    velocity = np.asarray(v_ref, dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("GCOPTER reference velocity must be finite with shape (3,)")
    return map_normalized_velocity(velocity / float(speed_limit_m_per_s), speed_limit_m_per_s)

