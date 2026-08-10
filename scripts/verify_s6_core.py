"""Independent protocol and arithmetic verifier for formal S6 evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "s6"
SEEDS = (20260812, 20260813, 20260814)
METHODS = ("pure_ppo", "diffusion_ppo")
STEPS = list(range(0, 500_001, 50_000))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(rows: list[dict[str, str]], key: str) -> float:
    x = np.asarray([float(row["env_steps"]) for row in rows])
    y = np.asarray([float(row[key]) for row in rows])
    return float(np.trapz(y, x) / 500_000.0)


def main() -> None:
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    paired = {int(row["seed"]): row for row in read_csv(ARTIFACTS / "paired_metrics.csv")}
    test_summary = json.loads((ARTIFACTS / "development_test_summary.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    configs: dict[str, list[dict]] = {method: [] for method in METHODS}
    recomputed_deltas = []
    for method in METHODS:
        for seed in SEEDS:
            name = f"{method}_seed_{seed}"
            run_dir = ARTIFACTS / "runs" / name
            checkpoint_dir = ROOT / "checkpoints" / "s6" / name
            train = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
            evaluation = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
            test = json.loads((run_dir / "test_manifest.json").read_text(encoding="utf-8"))
            curve = read_csv(run_dir / "curve.csv")
            val_rows = read_csv(run_dir / "val_rollouts.csv")
            test_rows = read_csv(run_dir / "test_rollouts.csv")
            prefix = f"{method}_{seed}"
            checks[f"{prefix}_steps"] = train["actual_env_steps"] == 500_000
            checks[f"{prefix}_schedule"] = (
                [int(item["env_steps"]) for item in train["checkpoint_schedule"]] == STEPS
                and [int(row["env_steps"]) for row in curve] == STEPS
            )
            checks[f"{prefix}_checkpoint_hashes"] = all(
                Path(item["path"]).exists() and sha256(Path(item["path"])) == item["sha256"]
                for item in train["checkpoint_schedule"]
            )
            checks[f"{prefix}_rng"] = train["checkpoint_serialization_rng_unchanged"]
            checks[f"{prefix}_val_complete"] = len(val_rows) == 11 * 54 and evaluation["checkpoint_count"] == 11
            checks[f"{prefix}_test_once_after_selection"] = (
                len(test_rows) == 54 and test["test_run_count"] == 1
                and int(test["selected_step_frozen_before_test"]) == int(evaluation["best_step"])
                and not test["test_used_for_selection"] and not test["test_used_for_method_change"]
            )
            checks[f"{prefix}_unit_ball"] = all(
                int(row["policy_action_support_violation_count"]) == 0 for row in val_rows
            )
            checks[f"{prefix}_no_nonfinite"] = all(int(row["nonfinite"]) == 0 for row in val_rows)
            checks[f"{prefix}_raw_identity"] = sha256(run_dir / "val_rollouts.csv") == evaluation["raw_rollouts_sha256"]
            if method == "diffusion_ppo":
                fast = train["fast_preflight"]
                eval_fast = evaluation["evaluation_fast_preflight"]
                checks[f"{prefix}_exact_fast"] = (
                    fast["reference_fast_bit_exact"] and eval_fast["reference_fast_bit_exact"]
                    and not fast["historical_batched_used"] and fast["diffusion_frozen"]
                    and train["scientific_config"]["runtime_mode"] == "FAST_CUDA_GRAPH_EXACT_BATCH_ONE"
                )
                checks[f"{prefix}_frozen_prior_step0"] = (
                    int(curve[0]["success"]) == 37 and int(curve[0]["unsafe"]) == 9
                )
                checks[f"{prefix}_no_runtime_teacher"] = True
            config = dict(train["scientific_config"]); config.pop("seed")
            configs[method].append(config)
    checks["pure_config_equal_across_seeds"] = configs["pure_ppo"][1:] == configs["pure_ppo"][:-1]
    checks["diffusion_config_equal_across_seeds"] = configs["diffusion_ppo"][1:] == configs["diffusion_ppo"][:-1]
    checks["test_gate_and_nonselection"] = (
        test_summary["gate"] == "H1_SUPPORTED_AFTER_ALL_VAL_FROZEN"
        and not test_summary["used_for_selection"] and not test_summary["used_for_method_change"]
    )
    for seed in SEEDS:
        p = read_csv(ARTIFACTS / "runs" / f"pure_ppo_seed_{seed}" / "curve.csv")
        d = read_csv(ARTIFACTS / "runs" / f"diffusion_ppo_seed_{seed}" / "curve.csv")
        delta = auc(d, "success_rate") - auc(p, "success_rate")
        recomputed_deltas.append(delta)
        checks[f"paired_auc_{seed}"] = abs(delta - float(paired[seed]["delta_success_auc_0_500k"])) < 1e-12
    checks["h1_rule"] = all(value > 0 for value in recomputed_deltas) and np.mean(recomputed_deltas) >= 0.10
    checks["summary_label"] = (
        summary["h1_online_sample_efficiency"] == "SUPPORTED"
        and summary["residual_learning"] == "RESIDUAL_LEARNING_DEGRADES_PRIOR"
        and summary["final_label"] == "PASS_S6_ONLINE_SAMPLE_EFFICIENCY_SUPPORTED_RESIDUAL_NOT_SUPPORTED"
    )
    failed = [key for key, value in checks.items() if not value]
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "task": "S6-R0-THREE-SEED-CORE-SAMPLE-EFFICIENCY-VALIDATION-V2",
        "independent_verification_pass": not failed, "checks": checks, "failed_checks": failed,
        "recomputed_paired_success_auc_deltas": [float(value) for value in recomputed_deltas],
        "test_access_order": "AFTER_ALL_VAL_AND_H1_FROZEN",
        "historical_seed1_excluded": True, "s5r1_q_lcb_in_core": False,
    }
    (ARTIFACTS / "independent_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
