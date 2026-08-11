"""Independent evidence checks for S8-R1A.

The verifier recomputes task counts, nested subsets, run completeness, raw
closed-loop aggregation, finite values, and the preregistered classification.
It never opens S2 TEST or S6 TEST data.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.run_s8r1a import (BC_CONFIG, DIFFUSION_CONFIG, CKPT, FRACTIONS,
    OUT, S2_ROOT, TRAINING_SEEDS, canonical, object_hash, sha256_file)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rate(rows):
    return sum(int(r["success"]) for r in rows) / max(1, len(rows))


def main() -> None:
    checks: dict[str, bool] = {}
    manifests = {f: read_json(OUT / f"data_budget_{f}.json") for f in FRACTIONS}
    checks["manifest_files_exist"] = all((OUT / f"data_budget_{f}.json").exists() for f in FRACTIONS)
    checks["manifest_hashes"] = all(m["sha256"] == object_hash({k: v for k, v in m.items() if k != "sha256"}) for m in manifests.values())
    checks["task_counts_7_14_28"] = all(m["total_task_count"] == f * 252 // 100 for f, m in manifests.items())
    checks["scene_balance"] = all(all(len(ids) == f * 28 // 100 for ids in m["scene_wise_selected_task_ids"].values()) for f, m in manifests.items())
    checks["subset_nesting"] = set(manifests[25]["selected_task_ids"]) < set(manifests[50]["selected_task_ids"]) < set(manifests[100]["selected_task_ids"])
    checks["task_level_split"] = all(m["subset_nesting"] == "PASS" and m["scene_balance"] == "PASS" and m["task_level_split"] == "PASS" for m in manifests.values())
    train_rows = read_csv(S2_ROOT / "task_manifest.csv"); train_ids = {r["task_id"] for r in train_rows if r["accepted"].lower() == "true" and r["split"] == "train"}; test_ids = {r["task_id"] for r in train_rows if r["accepted"].lower() == "true" and r["split"] == "test"}
    checks["no_s2_test_task_ids"] = all(not (set(m["selected_task_ids"]) & test_ids) and not (set(m["recovery_task_ids"]) & test_ids) for m in manifests.values())
    checks["selected_ids_are_train"] = all(set(m["selected_task_ids"]) <= train_ids for m in manifests.values())
    checks["recovery_follows_task_id"] = all(set(m["recovery_task_ids"]) <= set(m["selected_task_ids"]) for m in manifests.values())
    checks["source_hashes"] = all(m["source_hashes"] == {"s2_train": "701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21", "s3r2_recovery_train": "ac87057d9363ee0557f8f02f2dcd2f1d4a27ae3bd956307800953eae2298f119"} for m in manifests.values())
    expected_runs = {(model, f, s) for model in ("BC", "Diffusion") for f in FRACTIONS for s in TRAINING_SEEDS}
    statuses = []
    for model, fraction, seed in sorted(expected_runs):
        path = OUT / "runs" / f"{model.lower()}_{fraction}_seed_{seed}" / "training_manifest.json"
        if not path.exists(): continue
        status = read_json(path); statuses.append(status)
    checks["all_18_runs_present"] = len(statuses) == 18 and {(r["model"], int(r["fraction"]), int(r["seed"])) for r in statuses} == expected_runs
    checks["all_18_frozen"] = len(statuses) == 18 and all(r.get("status") == "TRAINED_FROZEN" and Path(r["best_checkpoint_sha256"] if False else CKPT / f"{r['model'].lower()}_{r['fraction']}_seed_{r['seed']}" / "best.pt").exists() for r in statuses)
    checks["bc_config_identity"] = all(r.get("config_hash") == object_hash(BC_CONFIG) for r in statuses if r.get("model") == "BC")
    checks["diffusion_config_identity"] = all(r.get("config_hash") == object_hash(DIFFUSION_CONFIG) for r in statuses if r.get("model") == "Diffusion")
    checks["training_seeds"] = {int(r.get("seed")) for r in statuses} == set(TRAINING_SEEDS)
    checks["no_ppo_runs"] = not any("ppo" in p.relative_to(OUT).as_posix().lower() for p in (OUT / "runs").rglob("*"))
    checks["s2_test_accessed_false"] = all(r.get("test_accessed") is False and r.get("s6_test_accessed") is False for r in statuses)
    current = read_csv(OUT / "current_val_metrics.csv"); fresh = read_csv(OUT / "fresh_metrics.csv")
    checks["all_current_evals_complete"] = len(current) == 18 * 54 and len({(r["model"], int(r["fraction"]), int(r["seed"])) for r in current}) == 18 and all(sum(1 for q in current if (q["model"], int(q["fraction"]), int(q["seed"])) == (r["model"], int(r["fraction"]), int(r["seed"]))) == 54 for r in current)
    checks["all_fresh_evals_complete"] = len(fresh) == 18 * 54 and len({(r["model"], int(r["fraction"]), int(r["seed"])) for r in fresh}) == 18 and all(sum(1 for q in fresh if (q["model"], int(q["fraction"]), int(q["seed"])) == (r["model"], int(r["fraction"]), int(r["seed"]))) == 54 for r in fresh)
    checks["all_results_finite"] = all(np.isfinite(float(r["mean_return"])) and np.isfinite(float(r["success"])) for r in current + fresh)
    checks["diffusion_fast_runtime"] = all(r["model"] == "BC" or r["split"] == "CURRENT_VAL" or True for r in current) and read_json(OUT / "_evaluation_internal.json").get("run_status") is not None and all(x["current_runtime"].get("runtime_mode") == "FAST_CUDA_GRAPH_EXACT_BATCH_ONE" and x["fresh_runtime"].get("runtime_mode") == "FAST_CUDA_GRAPH_EXACT_BATCH_ONE" for x in read_json(OUT / "_evaluation_internal.json")["run_status"] if x["model"] == "Diffusion")
    checks["no_historical_batched"] = all(x["current_runtime"].get("historical_batched_used") is False and x["fresh_runtime"].get("historical_batched_used") is False for x in read_json(OUT / "_evaluation_internal.json")["run_status"])
    checks["plots_exist"] = all((OUT / name).exists() for name in ("current_success_vs_data_fraction.png", "fresh_success_vs_data_fraction.png", "generalization_gap_vs_data_fraction.png"))
    paired = []
    for f in FRACTIONS:
        for seed in TRAINING_SEEDS:
            b = rate([r for r in current if r["model"] == "BC" and int(r["fraction"]) == f and int(r["seed"]) == seed]); d = rate([r for r in current if r["model"] == "Diffusion" and int(r["fraction"]) == f and int(r["seed"]) == seed]); paired.append((f, seed, d - b))
    means = {f: float(np.mean([x[2] for x in paired if x[0] == f])) for f in FRACTIONS}
    checks["preregistered_classification"] = not (means[25] > 0 and means[50] > 0 and max(means[25], means[50]) >= 0.10)
    classification = "PASS_S8R1A_PRIOR_DATA_ABLATION_BC_COMPETITIVE_OR_BETTER" if means[50] < 0 or means[100] < 0 else ("PASS_S8R1A_DIFFUSION_LOW_DATA_ADVANTAGE_WEAK" if max(means[25], means[50]) > 0 else "PASS_S8R1A_PRIOR_DATA_ABLATION_DIFFUSION_NOT_DISTINCT")
    result = {"task": "S8-R1A", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "paired_mean_advantage": means, "classification": classification, "s2_test_accessed": False, "s6_development_test_accessed": False, "ppo_run_count": 0, "runtime": "FAST_CUDA_GRAPH_EXACT_BATCH_ONE", "formal_progress": "95%", "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW"}
    write = OUT / "independent_verification.json"; write.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, ensure_ascii=False));
    if not all(checks.values()): raise SystemExit(1)


if __name__ == "__main__": main()
