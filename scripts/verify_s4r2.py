"""Independent protocol and frozen-evidence verifier for S4-R2."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "s4r2"
CHECKPOINTS = ROOT / "checkpoints" / "s4r2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    curve = list(csv.DictReader((ARTIFACTS / "learning_curve.csv").open(encoding="utf-8", newline="")))
    val = list(csv.DictReader((ARTIFACTS / "val_best_rollout.csv").open(encoding="utf-8", newline="")))
    family_success = {
        family: sum(int(row["success"]) for row in val if row["family"] == family)
        for family in ("OPEN", "BLOCK", "SBEND")
    }
    runtime_source = (ROOT / "scripts" / "run_s4r2_ppo.py").read_text(encoding="utf-8")
    forbidden = re.compile(r"(?:from|import)\s+quadrotor_diffusion_ppo\.(?:diffusion|expert)\b", re.I)
    violations = [line.strip() for line in runtime_source.splitlines() if forbidden.search(line)]
    test_path = ARTIFACTS / "test_rollout.csv"
    checks = {
        "exact_start_head": summary["start_head"] == "8f02f831821b84cb0cb53564892797e18edd4ba3",
        "split_counts": (summary["train_tasks"], summary["val_tasks"], summary["test_tasks"]) == (252, 54, 54),
        "train_only_normalization_source": summary["normalization"]["source"] == "artifacts/s2/dataset/train.npz::observations",
        "train_source_hash": summary["normalization"]["source_sha256"] == "701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21",
        "normalization_frozen_in_checkpoint": summary["normalization_checkpoint_persistence"] == "PASS",
        "raw_34d_unchanged": summary["observation_dim"] == 34 and summary["raw_environment_observation_unchanged"],
        "actor_critic_same_preprocessing": summary["actor_critic_identical_preprocessing"],
        "exact_training_budget": summary["total_env_steps"] == 500_000 and summary["n_envs"] == 8,
        "evaluation_schedule": [int(row["env_steps"]) for row in curve] == list(range(0, 500_001, 50_000)),
        "best_val_recomputed": len(val) == 54 and sum(int(row["success"]) for row in val) == summary["best_val"]["success"],
        "family_success_recomputed": family_success == {family: summary["val_by_family"][family]["success"] for family in family_success},
        "reward_hash_unchanged": summary["reward_contract_hash"] == "92afefbf338d0ccbbcc42c12a572327413ca66f46bf1f061f637e68d125ae3ec",
        "unit_ball_unchanged": summary["action_distribution"] == "UnitBallSquashedGaussian" and all(value == "PASS" for value in summary["unit_ball_oracle"].values()),
        "support_and_clipping": summary["policy_action_support_violation_fraction"] <= 0.01 and summary["env_action_clip_fraction"] <= 0.01,
        "no_privileged_runtime_dependency": not violations and not summary["pure_ppo_diffusion_dependence"] and not summary["pure_ppo_teacher_dependence"],
        "test_untouched_after_gate_fail": not test_path.exists() and summary["ppo_test_run_count"] == 0 and not summary["test_accessed"],
        "main_not_authorized": summary["main_fast_forward"] == "NOT_AUTHORIZED_VAL_GATE_FAIL",
        "checkpoint_pair": (CHECKPOINTS / "best_model.zip").exists() and (CHECKPOINTS / "last_model.zip").exists(),
    }
    best = summary["best_val"]
    val_gate = bool(
        best["success"] >= 9
        and all(value > 0 for value in family_success.values())
        and best["mean_return"] > summary["step0_val"]["mean_return"]
        and best["nonfinite"] == 0
        and summary["policy_action_support_violation_fraction"] <= 0.01
        and summary["env_action_clip_fraction"] <= 0.01
    )
    evidence_files = ["summary.json", "normalization.json", "learning_curve.csv", "val_best_rollout.csv", "engineering_failure_01.json"]
    payload = {
        "protocol_audit_pass": all(checks.values()),
        "val_gate_recomputed": "PASS" if val_gate else "FAIL",
        "summary_val_gate_match": summary["val_gate"] == ("PASS" if val_gate else "FAIL"),
        "family_success": family_success,
        "checks": checks,
        "forbidden_runtime_tokens": violations,
        "checkpoint_sha256": {
            "best_model.zip": sha256(CHECKPOINTS / "best_model.zip"),
            "last_model.zip": sha256(CHECKPOINTS / "last_model.zip"),
        },
        "artifact_sha256": {name: sha256(ARTIFACTS / name) for name in evidence_files},
    }
    (ARTIFACTS / "independent_verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["protocol_audit_pass"] or not payload["summary_val_gate_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
