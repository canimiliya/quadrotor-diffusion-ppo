from __future__ import annotations

import numpy as np

from scripts.run_s3r2 import HORIZON, POSITION_DIVERGENCE_THRESHOLD, select_first_anchor


def test_first_divergence_anchor_is_deterministic_and_single():
    rollout = {"trace": [
        {"step": 0, "time": 1 / 48, "position": np.array([0.0, 0.0, 0.0])},
        {"step": 1, "time": 2 / 48, "position": np.array([0.1, 0.0, 0.0])},
        {"step": 2, "time": 3 / 48, "position": np.array([0.31, 0.0, 0.0])},
        {"step": 3, "time": 4 / 48, "position": np.array([0.8, 0.0, 0.0])},
    ]}
    expert = {"times": np.array([1 / 48, 4 / 48]), "positions": np.zeros((2, 3))}
    first = select_first_anchor(rollout, expert)
    second = select_first_anchor(rollout, expert)
    assert first is not None
    assert first["step"] == 2
    assert first["position_error_to_expert"] > POSITION_DIVERGENCE_THRESHOLD
    assert first["step"] == second["step"]


def test_recovery_contract_constants_are_frozen():
    assert HORIZON == 16
    assert POSITION_DIVERGENCE_THRESHOLD == 0.30


def test_recovery_source_split_is_not_val_or_test():
    manifest = {"source_split": "TRAIN", "val_recovery_samples": 0, "test_recovery_samples": 0}
    assert manifest["source_split"] == "TRAIN"
    assert manifest["val_recovery_samples"] == 0
    assert manifest["test_recovery_samples"] == 0
