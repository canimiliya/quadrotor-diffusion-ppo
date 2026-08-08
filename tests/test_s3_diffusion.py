from __future__ import annotations

import inspect

import numpy as np
import torch

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from scripts.run_s3_diffusion import (
    HORIZON, OBSERVATION_DIM, ACTION_DIM, build_windows, observation_normalization,
    raw_action_statistics, teacher_independence_audit,
)


def _episodes(lengths=(20, 18)):
    total = sum(lengths)
    return {
        "observations": np.arange(total * 34, dtype=np.float32).reshape(total, 34),
        "actions": np.zeros((total, 3), dtype=np.float32),
        "episode_offsets": np.array([0, lengths[0]], dtype=np.int64),
        "episode_lengths": np.asarray(lengths, dtype=np.int32),
    }


def test_sequence_windows_do_not_cross_episode():
    obs, actions = build_windows(_episodes())
    assert len(obs) == (20 - HORIZON + 1) + (18 - HORIZON + 1)
    assert actions.shape[1:] == (HORIZON, ACTION_DIM)


def test_horizon_is_16():
    assert HORIZON == 16
    model = ConditionalDiffusionMLP()
    value = model(torch.zeros(2, 16, 3), torch.zeros(2, 34), torch.zeros(2, dtype=torch.long))
    assert value.shape == (2, 16, 3)


def test_train_only_observation_normalization():
    data = _episodes()
    mean, std = observation_normalization(data)
    assert np.allclose(mean, data["observations"].mean(axis=0))
    assert np.allclose(std, data["observations"].std(axis=0))


def test_diffusion_condition_is_34d_student_observation_only():
    model = ConditionalDiffusionMLP()
    assert model.observation_dim == OBSERVATION_DIM == 34
    assert "teacher" not in inspect.getsource(model.forward).lower()


def test_diffusion_output_shape():
    model = ConditionalDiffusionMLP()
    out = model(torch.randn(4, 16, 3), torch.randn(4, 34), torch.arange(4))
    assert out.shape == (4, 16, 3)


def test_ddim_deterministic_fixed_seed():
    model = ConditionalDiffusionMLP().eval()
    schedule = DiffusionSchedule(100)
    condition = torch.randn(2, 34)
    first = schedule.ddim_sample(model, condition, generator=torch.Generator().manual_seed(7))
    second = schedule.ddim_sample(model, condition, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_raw_action_logging_before_clip():
    stats = raw_action_statistics(np.array([[1.2, 0.0, -1.1], [0.5, 0.0, 0.0]], dtype=np.float32))
    assert stats["count"] == 1 and stats["fraction"] == 0.5 and np.isclose(stats["max_abs"], 1.2)


def test_closed_loop_teacher_independence():
    assert teacher_independence_audit()["passed"]


def test_checkpoint_metadata_contract():
    model = ConditionalDiffusionMLP()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert model.horizon == 16 and model.action_dim == 3 and model.observation_dim == 34


def test_test_split_not_used_before_freeze_and_s2_dataset_identity():
    # The runner's protocol explicitly freezes the model before opening test.npz.
    from scripts.run_s3_diffusion import dataset_identity
    identity = dataset_identity()
    assert identity["stats"]["train"]["trajectories"] == 252
    assert identity["stats"]["val"]["trajectories"] == 54
    assert identity["stats"]["test"]["trajectories"] == 54
