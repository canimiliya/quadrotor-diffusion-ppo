"""Structural and evidence checks for the read-only S4-D0 audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts" / "s4d0" / "summary.json"


def load_summary() -> dict:
    assert SUMMARY.exists()
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_s4d0_exact_r1_reproduction_and_test_is_untouched():
    summary = load_summary()
    reproduction = summary["r1_val_reproduction"]
    assert summary["final_label"] == "PASS_S4D0_PPO_OBSTACLE_ROOT_CAUSE_AUDIT"
    assert reproduction["match"] is True
    assert (reproduction["success"], reproduction["collision"], reproduction["ground"], reproduction["timeout"]) == (18, 36, 0, 0)
    assert {family: reproduction["by_family"][family]["success"] for family in ("OPEN", "BLOCK", "SBEND")} == {
        "OPEN": 18, "BLOCK": 0, "SBEND": 0
    }
    assert summary["test_accessed"] is False
    assert summary["test_run_count"] == 0
    assert summary["protocol"]["no_test"] is True
    assert not (ROOT / "artifacts" / "s4d0" / "test_rollout.csv").exists()


def test_s4d0_diagnosis_contains_required_independent_evidence():
    summary = load_summary()
    assert summary["checkpoint_identity"] == "UnitBallSquashedGaussian=true"
    assert len(summary["checkpoint_sha256"]) == 64
    assert summary["ray_intervention"]["near_obstacle_1s_all_blocked"]["samples"] == 1764
    sensitivity = summary["normalized_policy_sensitivity"]
    assert sensitivity["groups"]["goal_delta"]["mean"] > sensitivity["groups"]["rays"]["mean"]
    assert sensitivity["goal_delta_to_rays_mean_ratio"] < 0.25
    assert summary["collision_direction"]["counts"] == {
        "toward_obstacle": 36, "away_from_obstacle": 0, "mixed": 0
    }
    assert summary["greedy_goal_val"]["by_family"]["BLOCK"]["collision"] == 18
    assert summary["greedy_goal_val"]["by_family"]["SBEND"]["collision"] == 18
    assert summary["teacher_independence"]["status"] == "PASS"
    assert summary["formal_progress"] == "45%"


def test_s4d0_protocol_and_artifact_contract():
    summary = load_summary()
    assert all(summary["protocol"].values())
    for relative in (
        "artifacts/s4d0/summary.json",
        "artifacts/s4d0/per_episode.csv",
        "artifacts/s4d0/reward_audit.csv",
        "docs/S4D0_PPO_FAILURE_AUDIT.md",
    ):
        assert (ROOT / relative).exists()
    source = (ROOT / "scripts" / "audit_s4d0.py").read_text(encoding="utf-8")
    assert "model.learn" not in source
    assert "test.npz" not in source
    assert "PPO.load" in source
    assert "UnitBallSquashedGaussianDistribution" in source
