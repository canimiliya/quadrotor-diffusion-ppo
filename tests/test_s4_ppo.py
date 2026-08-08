from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from quadrotor_diffusion_ppo.envs.action import map_normalized_velocity
from quadrotor_diffusion_ppo.ppo.contract import (
    ACTION_DIM, OBSERVATION_DIM, PPO_CONFIG, PPO_CONFIG_HASH, REWARD_CONTRACT_HASH,
    compute_reward,
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
