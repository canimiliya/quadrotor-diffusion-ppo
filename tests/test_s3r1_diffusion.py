from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
from scripts.run_s3_diffusion import PROJECT_ROOT


SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "s3r1" / "summary.json"


def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_prediction_target_is_x0_and_output_is_bounded():
    value = ConditionalDiffusionMLP(bounded_output=True)(torch.randn(2, 16, 3), torch.randn(2, 34), torch.tensor([0, 99]))
    assert torch.isfinite(value).all()
    assert torch.all(torch.linalg.vector_norm(value, dim=-1) <= 1.0 + 1e-6)


def test_r0_dataset_identity_unchanged():
    data = summary()["dataset_identity"]
    assert data["sha256"] == {
        "train": "701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21",
        "val": "c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc",
        "test": "88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66",
    }


def test_train_only_normalization_unchanged():
    r0 = json.loads((PROJECT_ROOT / "artifacts" / "s3" / "summary.json").read_text(encoding="utf-8"))
    r1 = summary()
    assert r1["observation_normalization"] == "train_only"
    assert np.allclose(r1["observation_mean"], r0["observation_mean"])
    assert np.allclose(r1["observation_std"], r0["observation_std"])


def test_val_gate_precedes_test():
    result = summary()
    assert result["val_gate"] == "FAIL"
    assert result["final_label"] == "BLOCKED_S3R1_BOUNDED_X0_VAL_WEAK"
    assert result["r1_test_executed"] is False
    assert not (PROJECT_ROOT / "artifacts" / "s3r1" / "test_rollout.csv").exists()


def test_teacher_independence():
    assert summary()["privileged_information_leakage"]["passed"] is True


def test_r1_test_single_run_contract():
    result = summary()
    assert result["r1_test_run_count"] == 0
    assert result["r1_test_used_for_selection"] is False
