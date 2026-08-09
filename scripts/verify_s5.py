"""Independent machine-readable verification for the frozen S5 evidence."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quadrotor_diffusion_ppo.ppo.contract import PPO_CONFIG, REWARD_CONTRACT_HASH
from quadrotor_diffusion_ppo.ppo.residual import EXPECTED_S3R2_SHA256, RESIDUAL_SCALE, sha256_file


ARTIFACTS = ROOT / "artifacts" / "s5"
CHECKPOINTS = ROOT / "checkpoints" / "s5"
SUMMARY = ARTIFACTS / "summary.json"
S3R2 = ROOT / "checkpoints" / "s3r2" / "best.pt"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> dict:
    count = len(rows)
    total_steps = sum(int(row["episode_steps"]) for row in rows)
    return {
        "tasks": count,
        "success": sum(int(row["success"]) for row in rows),
        "collision": sum(int(row["collision"]) for row in rows),
        "ground_contact": sum(int(row["ground_contact"]) for row in rows),
        "unsafe": sum(int(row["unsafe"]) for row in rows),
        "timeout": sum(int(row["timeout"]) for row in rows),
        "nonfinite": sum(int(row["nonfinite"]) for row in rows),
        "action_clip_count": sum(int(row["action_clip_count"]) for row in rows),
        "support_violation_count": sum(int(row["policy_action_support_violation_count"]) for row in rows),
        "projection_count": sum(int(row["projection_count"]) for row in rows),
        "projection_fraction": sum(int(row["projection_count"]) for row in rows) / max(1, total_steps),
        "by_family": {
            family: sum(int(row["success"]) for row in rows if row["family"] == family)
            for family in ("OPEN", "BLOCK", "SBEND")
        },
    }


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    curve = read_csv(ARTIFACTS / "learning_curve.csv")
    val_rows = read_csv(ARTIFACTS / "val_best_rollout.csv")
    test_rows = read_csv(ARTIFACTS / "test_rollout.csv")
    val = aggregate(val_rows)
    test = aggregate(test_rows)
    curve_key = lambda row: (
        int(row["success"]), -int(row["unsafe"]), float(row["mean_return"]), -int(row["env_steps"])
    )
    selected = max(curve, key=curve_key)
    source = (ROOT / "scripts" / "run_s5_residual_ppo.py").read_text(encoding="utf-8").lower()
    forbidden = [token for token in (
        "quadrotor_diffusion_ppo.expert", "gcoptertrajectory", "pmm_uav_planner",
        "recovery_dataset", "test_used_for_selection\": true",
    ) if token in source]
    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): stable_hash(path)
        for path in (
            CHECKPOINTS / "best_model.zip", CHECKPOINTS / "last_model.zip",
            ARTIFACTS / "summary.json", ARTIFACTS / "learning_curve.csv",
            ARTIFACTS / "val_best_rollout.csv", ARTIFACTS / "test_rollout.csv",
            ARTIFACTS / "prior_reference.json",
        )
    }
    checks = {
        "diffusion_checkpoint_identity": sha256_file(S3R2) == EXPECTED_S3R2_SHA256,
        "diffusion_frozen": summary["diffusion_frozen"] is True,
        "diffusion_gradient_absent": summary["diffusion_gradient_present"] is False,
        "normalization_identity": all(summary["observation_normalization_identity"][key] for key in (
            "normalization_mean_float32_identity", "normalization_std_float32_identity",
            "constant_feature_runtime_equivalence", "optimized_reference_pass",
        )),
        "zero_residual_identity": summary["observation_normalization_identity"]["zero_residual_exact_identity"] is True,
        "reward_identity": summary["reward_contract_hash"] == REWARD_CONTRACT_HASH,
        "ppo_contract_identity": all(summary["ppo_config"][key] == PPO_CONFIG[key] for key in (
            "seed", "total_env_steps", "n_envs", "learning_rate", "n_steps", "batch_size",
            "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
        )),
        "residual_scale_frozen": summary["residual_scale"] == RESIDUAL_SCALE == 0.25,
        "curve_schedule_exact": [int(row["env_steps"]) for row in curve] == list(range(0, 500001, 50000)),
        "total_steps_exact": summary["total_env_steps"] == 500000,
        "best_selection_recomputed": int(selected["env_steps"]) == 0 and int(selected["success"]) == 37,
        "val_recomputed": val["tasks"] == 54 and val["success"] == 37 and val["by_family"] == {"OPEN": 17, "BLOCK": 9, "SBEND": 11},
        "test_recomputed": test["tasks"] == 54 and test["success"] == 38 and test["by_family"] == {"OPEN": 18, "BLOCK": 8, "SBEND": 12},
        "support_and_clipping_zero": val["support_violation_count"] == test["support_violation_count"] == 0 and val["action_clip_count"] == test["action_clip_count"] == 0,
        "test_protocol": summary["test_executed"] is True and summary["test_run_count"] == 1 and summary["test_used_for_selection"] is False,
        "teacher_runtime_independent": summary["teacher_runtime_dependence"] is False and not forbidden,
        "gate_recomputed": val["success"] >= 27 and all(val["by_family"][family] > 0 for family in ("OPEN", "BLOCK", "SBEND")),
        "residual_did_not_improve_prior": max(int(row["success"]) for row in curve[1:]) < int(curve[0]["success"]),
        "pure_ppo_preserved_as_valid_weak_baseline": summary["pure_ppo_baseline"]["status"] == "VALID_WEAK_BASELINE" and summary["pure_ppo_baseline"]["success"] == 14,
    }
    result = {
        "task": summary["task"],
        "protocol_audit_pass": all(checks.values()),
        "final_label_recomputed": "PASS_S5_DIFFUSION_PRIOR_RESIDUAL_PPO_SANITY" if checks["gate_recomputed"] else "BLOCKED_S5_PRIOR_INTEGRATION_NO_BENEFIT",
        "checks": checks,
        "best_val_recomputed": val,
        "test_recomputed": test,
        "selected_curve_row": selected,
        "forbidden_runtime_tokens": forbidden,
        "sha256": hashes,
    }
    (ARTIFACTS / "independent_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not result["protocol_audit_pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"S5 independent verification failed: {failed}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
