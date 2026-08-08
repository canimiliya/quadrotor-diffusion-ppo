"""Independent S4 artifact and protocol verifier."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "s4"
CHECKPOINTS = ROOT / "checkpoints" / "s4"


def load_json(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def main() -> None:
    summary = load_json("summary.json")
    curve = list(csv.DictReader((ARTIFACTS / "learning_curve.csv").open("r", encoding="utf-8", newline="")))
    val = list(csv.DictReader((ARTIFACTS / "val_best_rollout.csv").open("r", encoding="utf-8", newline="")))
    test = list(csv.DictReader((ARTIFACTS / "test_rollout.csv").open("r", encoding="utf-8", newline="")))
    checks = {
        "canonical_main": summary["canonical_main"] == "b10c8edf2fc834aac9136626a969c0c3108820ae",
        "main_fast_forward": summary["main_fast_forward"] == "PASS",
        "split_counts": (summary["train_tasks"], summary["val_tasks"], summary["test_tasks"]) == (252, 54, 54),
        "exact_training_budget": summary["total_env_steps"] == 500000,
        "evaluation_schedule": [int(row["env_steps"]) for row in curve] == [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000],
        "val_rows": len(val) == 54 and int(summary["best_val"]["success"]) == sum(int(row["success"]) for row in val),
        "test_rows": len(test) == 54 and summary["ppo_test_run_count"] == 1 and not summary["ppo_test_used_for_selection"],
        "model_frozen": summary["model_frozen"] is True,
        "finite": summary["numerical_finite"] is True and int(summary["test"]["nonfinite"]) == 0,
        "shape_contract": summary["observation_dim"] == 34 and summary["action_dim"] == 3,
        "no_vec_normalize": summary["ppo_config"]["observation_normalization"] == "none",
        "no_privileged_dependencies": not summary["pure_ppo_diffusion_dependence"] and not summary["pure_ppo_teacher_dependence"],
        "checkpoint_pair": (CHECKPOINTS / "best_model.zip").exists() and (CHECKPOINTS / "last_model.zip").exists(),
    }
    # This audit is intentionally narrow: it rejects only forbidden runtime imports/tokens
    # in the newly added Pure PPO code, not historical S3 source files.
    runtime_files = [ROOT / "scripts" / "run_s4_ppo.py", ROOT / "src" / "quadrotor_diffusion_ppo" / "ppo"]
    forbidden = re.compile(
        r"(?:from|import)\s+(?:quadrotor_diffusion_ppo\.)?(?:diffusion|expert|gcopter)\b|"
        r"(?:from|import)\s+quadrotor_diffusion_ppo\.(?:diffusion|expert)\b", re.I)
    violations = []
    for location in runtime_files:
        paths = [location] if location.is_file() else list(location.rglob("*.py"))
        for path in paths:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if forbidden.search(line):
                    violations.append(f"{path.relative_to(ROOT)}:{line_no}:{line.strip()}")
    checks["forbidden_runtime_tokens"] = not violations
    best = summary["best_val"]
    family_success = {family: int(values["success"]) for family, values in summary["val_by_family"].items()}
    usability_gate = bool(
        int(best["success"]) >= 9 and all(value >= 1 for value in family_success.values()) and
        float(best["mean_return"]) > float(summary["step0_val"]["mean_return"]) and
        int(summary["test"]["nonfinite"]) == 0
    )
    payload = {"audit_pass": all(checks.values()) and usability_gate,
               "protocol_audit_pass": all(checks.values()), "usability_gate_pass": usability_gate,
               "val_family_success": family_success, "checks": checks, "forbidden_runtime_tokens": violations,
               "artifact_hashes": {name: hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
                                    for name in ("summary.json", "learning_curve.csv", "val_best_rollout.csv", "test_rollout.csv")}}
    (ARTIFACTS / "independent_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
