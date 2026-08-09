"""Frozen evidence checks for the completed S4-R2 scientific result."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "s4r2"


def summary() -> dict:
    return json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))


def test_s4r2_completed_exact_budget_but_failed_obstacle_gate():
    data = summary()
    assert data["final_label"] == "BLOCKED_S4R2_PPO_NO_OBSTACLE_LEARNING"
    assert data["total_env_steps"] == 500_000
    assert data["best_val"]["success"] == 14
    assert {family: data["val_by_family"][family]["success"] for family in ("OPEN", "BLOCK", "SBEND")} == {
        "OPEN": 14, "BLOCK": 0, "SBEND": 0,
    }
    assert data["val_gate"] == "FAIL"
    assert data["formal_progress"] == "45%"


def test_s4r2_test_and_main_boundaries_held():
    data = summary()
    assert data["ppo_test_run_count"] == 0
    assert data["test"] is None
    assert data["test_accessed"] is False
    assert not (ARTIFACTS / "test_rollout.csv").exists()
    assert data["main_fast_forward"] == "NOT_AUTHORIZED_VAL_GATE_FAIL"


def test_s4r2_curve_and_normalization_effect_are_preserved():
    data = summary()
    with (ARTIFACTS / "learning_curve.csv").open(encoding="utf-8", newline="") as handle:
        curve = list(csv.DictReader(handle))
    assert [int(row["env_steps"]) for row in curve] == list(range(0, 500_001, 50_000))
    assert data["normalization"]["source"].endswith("train.npz::observations")
    assert data["normalization"]["constant_feature_indices"] == [20]
    assert data["ray_goal_sensitivity"]["r2"]["ray_to_goal_ratio"] > data["ray_goal_sensitivity"]["r1"]
    assert data["policy_action_support_violation_fraction"] == 0.0
    assert data["env_action_clip_fraction"] == 0.0
