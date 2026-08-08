"""Frozen S4 Pure PPO observation, action, reward, and optimizer contract."""
from __future__ import annotations

import hashlib
import json
from typing import Any

OBSERVATION_DIM = 34
ACTION_DIM = 3
CONTROL_HZ = 48
MAX_EPISODE_STEPS = 960
GOAL_TOLERANCE_M = 0.30
N_ENVS = 8
SEED = 20260812
TOTAL_ENV_STEPS = 500_000

REWARD_CONTRACT: dict[str, Any] = {
    "progress_scale": 2.0,
    "time_penalty": -0.002,
    "success_bonus": 20.0,
    "collision_penalty": -20.0,
    "ground_penalty": -20.0,
    "goal_tolerance_m": GOAL_TOLERANCE_M,
    "formula": "2.0*(d_prev-d_now)-0.002 plus exactly one terminal event penalty/bonus",
}
PPO_CONFIG: dict[str, Any] = {
    "seed": SEED,
    "total_env_steps": TOTAL_ENV_STEPS,
    "n_envs": N_ENVS,
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.20,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "policy": "MlpPolicy",
    "policy_net_arch": {"pi": [256, 256], "vf": [256, 256]},
    "observation_normalization": "none",
    "device": "cuda",
}


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


REWARD_CONTRACT_HASH = _stable_hash(REWARD_CONTRACT)
PPO_CONFIG_HASH = _stable_hash(PPO_CONFIG)


def compute_reward(d_prev: float, d_now: float, *, success: bool = False,
                   collision: bool = False, ground: bool = False) -> tuple[float, dict[str, float]]:
    """Compute the exact S4 reward and expose each auditable component."""
    progress = REWARD_CONTRACT["progress_scale"] * (float(d_prev) - float(d_now))
    time_penalty = REWARD_CONTRACT["time_penalty"]
    terminal = (REWARD_CONTRACT["success_bonus"] if success else 0.0)
    terminal += REWARD_CONTRACT["collision_penalty"] if collision else 0.0
    terminal += REWARD_CONTRACT["ground_penalty"] if ground else 0.0
    components = {"progress": float(progress), "time": float(time_penalty), "terminal": float(terminal)}
    return float(progress + time_penalty + terminal), components
