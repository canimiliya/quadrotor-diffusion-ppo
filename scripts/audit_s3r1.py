"""Independent read-only audit for the S3-R1 VAL-gated result."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from scripts.run_s3r1_bounded_x0 import EXPECTED_DATASET_SHA256, PROJECT_ROOT


def main() -> int:
    root = PROJECT_ROOT / "artifacts" / "s3r1"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    val_rows = list(csv.DictReader((root / "val_rollout.csv").open(encoding="utf-8", newline="")))
    val = summary["val_closed_loop"]["overall"]
    actions = summary["val_action_statistics"]
    checks = {
        "final_label": summary["final_label"] == "BLOCKED_S3R1_BOUNDED_X0_VAL_WEAK",
        "dataset_identity": summary["dataset_identity"]["sha256"] == EXPECTED_DATASET_SHA256,
        "dataset_counts": summary["dataset_identity"]["stats"] == {
            "train": {"trajectories": 252, "transitions": 129458},
            "val": {"trajectories": 54, "transitions": 27937},
            "test": {"trajectories": 54, "transitions": 27476},
        },
        "x0_parameterization": summary["prediction_target"] == "x0" and summary["action_support_parameterization"] == "unit_ball_squash",
        "checkpoint_frozen": summary["model_frozen"] and summary["training"]["all_finite"],
        "val_task_count": len(val_rows) == 54,
        "val_success_gate": val["success"] >= 27,
        "val_family_gate": all(summary["val_closed_loop"]["by_family"][family]["success"] > 0 for family in ("OPEN", "BLOCK", "SBEND")),
        "val_safety_gate": val["collision"] + val["ground_contact"] <= 13.5,
        "val_nonfinite_gate": val["nonfinite"] == 0,
        "x0_support_gate": actions["diffusion_x0"]["support_violation_fraction"] <= 0.01,
        "executed_clip_gate": actions["executed_action"]["clip_fraction"] <= 0.01,
        "teacher_independence": summary["privileged_information_leakage"]["passed"],
        "val_determinism": all(item["match"] for item in summary["val_determinism_smoke"].values()),
        "test_not_opened": summary["r1_test_executed"] is False and summary["r1_test_run_count"] == 0 and not (root / "test_rollout.csv").exists(),
        "pytest": False,
    }
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, text=True, capture_output=True)
    checks["pytest"] = completed.returncode == 0
    audit_pass = all(value for name, value in checks.items() if name != "val_safety_gate") and not checks["val_safety_gate"]
    result = {"final_label": summary["final_label"], "audit_pass": audit_pass, "checks": checks,
              "val_success": val["success"], "val_collision": val["collision"],
              "val_ground_contact": val["ground_contact"],
              "val_collision_ground_fraction": (val["collision"] + val["ground_contact"]) / len(val_rows),
              "pytest_returncode": completed.returncode}
    print(json.dumps(result, indent=2))
    return 0 if audit_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
