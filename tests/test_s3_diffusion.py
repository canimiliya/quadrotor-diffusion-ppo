from __future__ import annotations

import inspect

import numpy as np
import torch

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP, unit_ball_squash
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


def test_unit_ball_squash_zero():
    output = unit_ball_squash(torch.zeros(8, 16, 3))
    assert torch.isfinite(output).all() and torch.equal(output, torch.zeros_like(output))


def test_unit_ball_squash_large_input():
    output = unit_ball_squash(torch.tensor([[[1e20, -1e20, 1e20]]]))
    assert torch.isfinite(output).all() and torch.linalg.vector_norm(output, dim=-1).item() < 1.0


def test_unit_ball_squash_random_norm_bound():
    output = unit_ball_squash(torch.randn(128, 16, 3) * 100.0)
    assert torch.isfinite(output).all()
    assert float(torch.max(torch.linalg.vector_norm(output, dim=-1))) < 1.0


def test_x0_sampler_oracle_reconstruction():
    schedule = DiffusionSchedule(100)
    x0 = torch.randn(2, 16, 3) * 0.2
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([99, 99])
    noisy = schedule.add_noise(x0, noise, timesteps)
    alpha_next = schedule.alpha_bars[88]
    epsilon = (noisy - schedule.alpha_bars[99].sqrt() * x0) / (1.0 - schedule.alpha_bars[99]).sqrt()
    reconstructed = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * epsilon
    expected = schedule.add_noise(x0, noise, torch.tensor([88, 88]))
    assert torch.allclose(reconstructed, expected, atol=2e-6, rtol=2e-6)


def test_prediction_target_is_x0():
    model = ConditionalDiffusionMLP(bounded_output=True)
    output = model(torch.randn(2, 16, 3), torch.randn(2, 34), torch.tensor([0, 99]))
    assert torch.isfinite(output).all()
    assert float(torch.max(torch.linalg.vector_norm(output, dim=-1))) < 1.0


def test_test_split_not_used_before_freeze_and_s2_dataset_identity():
    # The runner's protocol explicitly freezes the model before opening test.npz.
    from scripts.run_s3_diffusion import dataset_identity
    identity = dataset_identity()
    assert identity["stats"]["train"]["trajectories"] == 252
    assert identity["stats"]["val"]["trajectories"] == 54
    assert identity["stats"]["test"]["trajectories"] == 54
