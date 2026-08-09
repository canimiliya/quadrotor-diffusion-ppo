"""S5 generated-evidence and source-boundary checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "s5"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_s5_summary_and_gate_evidence():
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_label"] == "PASS_S5_DIFFUSION_PRIOR_RESIDUAL_PPO_SANITY"
    assert summary["total_env_steps"] == 500_000
    assert summary["best_val"]["success"] == 37
    assert {family: summary["val_by_family"][family]["success"] for family in ("OPEN", "BLOCK", "SBEND")} == {
        "OPEN": 17, "BLOCK": 9, "SBEND": 11,
    }
    assert summary["diffusion_frozen"] is True
    assert summary["diffusion_gradient_present"] is False
    assert summary["teacher_runtime_dependence"] is False
    assert summary["formal_progress"] == "70%"


def test_s5_curve_and_selection_are_complete():
    curve = read_csv("learning_curve.csv")
    assert [int(row["env_steps"]) for row in curve] == list(range(0, 500001, 50000))
    assert max(int(row["success"]) for row in curve) == 37
    assert max(int(row["success"]) for row in curve[1:]) == 35
    assert int(curve[-1]["success"]) == 17
    assert float(curve[-1]["projection_fraction"]) < 0.10


def test_s5_test_was_single_post_gate_and_not_selection_data():
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv("test_rollout.csv")
    assert summary["val_gate"] == "PASS"
    assert summary["test_executed"] is True
    assert summary["test_run_count"] == 1
    assert summary["test_used_for_selection"] is False
    assert len(rows) == 54
    assert sum(int(row["success"]) for row in rows) == 38


def test_independent_verifier_passes_and_preserves_limitation():
    evidence = json.loads((ARTIFACTS / "independent_verification.json").read_text(encoding="utf-8"))
    assert evidence["protocol_audit_pass"] is True
    assert evidence["checks"]["residual_did_not_improve_prior"] is True
    assert evidence["checks"]["pure_ppo_preserved_as_valid_weak_baseline"] is True


def test_s5_runtime_source_has_no_teacher_or_s6_path():
    source = (ROOT / "scripts" / "run_s5_residual_ppo.py").read_text(encoding="utf-8").lower()
    assert "quadrotor_diffusion_ppo.expert" not in source
    assert "gcoptertrajectory" not in source
    assert "run_s6" not in source
    assert "test_used_for_selection\": false" in source
