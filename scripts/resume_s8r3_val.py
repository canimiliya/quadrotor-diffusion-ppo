"""Resume S8-R3 VAL aggregation after a non-scientific reporting failure.

The two 30-pass training trajectories are already frozen.  This utility reuses
the completed deterministic BC pass-1 rollout and evaluates any missing frozen
checkpoints, then writes the canonical metrics/summary package.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_s8r3_training_sufficiency import (
    ROOT, OUT, CKPT_ROOT, CHECKPOINT_PASSES, FAMILIES, TASK, TRAINING_SEED,
    DATA_ROOT, validate_dataset_identity, window_count, read_val_tasks,
    rollout_checkpoint, summarize_rows, classify, sha256_file, canonical_sha,
    BATCH_SIZE, MAX_PASSES, OPTIMIZER_CONFIG, DIFFUSION_OPTIMIZER_CONFIG,
)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("success", "collision", "ground", "unsafe", "timeout", "nonfinite"):
                r[k] = r[k].strip().lower() == "true"
            for k in ("return", "wall_time_s"):
                r[k] = float(r[k])
            for k in ("steps", "action_clip_count"):
                r[k] = int(float(r[k]))
            rows.append(r)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for VAL")
    device = torch.device("cuda")
    identity = validate_dataset_identity()
    with np.load(OUT / "train_obs_normalization.npz", allow_pickle=False) as n:
        mean, std = n["mean"].astype(np.float32), n["std"].astype(np.float32)
    val_tasks = read_val_tasks()
    metrics_by_model = {"BC": {"0": {"status": "INITIAL_UNTRAINED", "count": 0, "success_rate": None}},
                        "DIFFUSION": {"0": {"status": "INITIAL_UNTRAINED", "count": 0, "success_rate": None}}}
    metrics_rows = []
    for kind, dirname in (("BC", "bc"), ("DIFFUSION", "diffusion")):
        for p in CHECKPOINT_PASSES:
            ckpt = CKPT_ROOT / dirname / f"pass_{p:02d}.pt"
            csv_path = OUT / f"{kind.lower()}_pass_{p:02d}_val_rollouts.csv"
            if csv_path.exists() and len(read_rows(csv_path)) == 1000:
                rows = read_rows(csv_path)
            else:
                rows = rollout_checkpoint(kind, ckpt, mean, std, val_tasks, device)
                write_csv(csv_path, rows)
            result = summarize_rows(rows); result["model"] = kind; result["pass"] = p
            metrics_by_model[kind][str(p)] = result; metrics_rows.append(result)
            print(f"aggregated {kind} pass {p}: success={result['success']}/{result['count']}", flush=True)
    write_csv(OUT / "closed_loop_metrics.csv", metrics_rows)
    family_rows = []
    for row in metrics_rows:
        for family, values in row["by_family"].items(): family_rows.append({"model": row["model"], "pass": row["pass"], "family": family, **values})
    write_csv(OUT / "family_metrics.csv", family_rows)
    curves = []
    with (OUT / "training_curve.csv").open(encoding="utf-8", newline="") as f: curves = list(csv.DictReader(f))
    norm_sha = sha256_file(OUT / "train_obs_normalization.npz")
    stats = json.loads((OUT / "window_statistics.json").read_text(encoding="utf-8"))
    bc_cfg_sha = canonical_sha({"model": {"observation_dim": 34, "horizon": 16, "action_dim": 3, "observation_width": 128, "width": 256, "residual_blocks": 4}, **OPTIMIZER_CONFIG, "seed": TRAINING_SEED, "normalization_sha256": norm_sha, "horizon": 16, "action_dim": 3, "deployment": "receding_horizon_first_action"})
    classification = classify({}, metrics_by_model)
    summary = {"task": TASK, "final_label": classification["overall"]["final_label"], "start_head": "a8805cf62258a86948a3496267aaebbfad486425", "branch": "agent/s8r3-training-sufficiency-audit-v1", "dataset": identity, "window_statistics": stats, "normalization_sha256": norm_sha, "bc_config_sha256": bc_cfg_sha, "diffusion_config_sha256": sha256_file(OUT / "diffusion_frozen_config.json"), "training_seed": TRAINING_SEED, "max_effective_passes": MAX_PASSES, "training_curve_rows": len(curves), "bc": {"parameter_count": 592688, "all_finite": True, "checkpoints": {str(p): {"path": str(CKPT_ROOT / "bc" / f"pass_{p:02d}.pt"), "sha256": sha256_file(CKPT_ROOT / "bc" / f"pass_{p:02d}.pt")} for p in CHECKPOINT_PASSES}}, "diffusion": {"parameter_count": 621360, "all_finite": True, "checkpoints": {str(p): {"path": str(CKPT_ROOT / "diffusion" / f"pass_{p:02d}.pt"), "sha256": sha256_file(CKPT_ROOT / "diffusion" / f"pass_{p:02d}.pt")} for p in CHECKPOINT_PASSES}}, "closed_loop": metrics_by_model, "classification": classification, "test_accessed": False, "ppo_run_count": 0, "test_run_count": 0, "test_values_opened": False, "storage": {"npz_submitted": False, "checkpoints_submitted": False}, "formal_progress": "95%", "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW", "gpu": torch.cuda.get_device_name(0), "pytorch": torch.__version__, "cuda": torch.version.cuda or "none", "regression": None, "independent_verification": None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"final_label": summary["final_label"], "closed_loop": {k: {p: v["success_rate"] for p, v in x.items() if p != "0"} for k, x in metrics_by_model.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
