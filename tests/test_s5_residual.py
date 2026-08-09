"""S5 frozen-prior residual-action unit tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quadrotor_diffusion_ppo.ppo.residual import (
    EXPECTED_S3R2_SHA256,
    RESIDUAL_SCALE,
    FrozenDiffusionPrior,
    compose_residual_action,
    sha256_file,
    stable_prior_seed,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "s3r2" / "best.pt"


def test_s3r2_checkpoint_identity_and_seed_contract():
    assert sha256_file(CHECKPOINT) == EXPECTED_S3R2_SHA256
    assert stable_prior_seed("BLOCK_00_TASK_003") == stable_prior_seed("BLOCK_00_TASK_003")
    assert stable_prior_seed("BLOCK_00_TASK_003") != stable_prior_seed("SBEND_00_TASK_016")


def test_zero_residual_is_exact_identity_and_composition_is_supported():
    rng = np.random.default_rng(20260812)
    prior = rng.normal(size=(128, 3)).astype(np.float32)
    prior /= np.maximum(1.0, np.linalg.norm(prior, axis=1, keepdims=True))
    same, projected = compose_residual_action(prior, np.zeros_like(prior))
    assert np.array_equal(same, prior)
    assert not projected.any()
    residual = rng.normal(size=(128, 3)).astype(np.float32)
    residual /= np.maximum(1.0, np.linalg.norm(residual, axis=1, keepdims=True))
    combined, _ = compose_residual_action(prior, residual, RESIDUAL_SCALE)
    assert np.isfinite(combined).all()
    assert np.max(np.linalg.norm(combined, axis=1)) <= 1.0 + 1.0e-6


def test_invalid_residual_scale_is_rejected():
    values = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        compose_residual_action(values, values, 0.0)
    with pytest.raises(ValueError):
        compose_residual_action(values, values, 0.51)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA preflight")
def test_frozen_prior_batched_path_matches_s3_reference():
    prior = FrozenDiffusionPrior(CHECKPOINT, "cuda")
    with np.load(ROOT / "artifacts" / "s2" / "dataset" / "train.npz", allow_pickle=False) as data:
        observations = data["observations"][[0, 100, 1000, 10000, 50000, 100000, 120000, 129000]]
    seeds = [stable_prior_seed(f"SYNTHETIC_{index}") + index for index in range(len(observations))]
    reference = prior.predict_reference(observations, seeds)
    optimized = prior.predict_batched(observations, seeds)
    assert prior.frozen
    assert not prior.gradient_present
    assert np.max(np.abs(reference - optimized)) <= 2.0e-6
    assert np.max(np.linalg.norm(optimized, axis=1)) <= 1.0 + 1.0e-6
