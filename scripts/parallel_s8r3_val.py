"""Parallel task-sharded S8-R3 VAL evaluator.

Each worker owns independent PyBullet environments and a frozen copy of the
same policy.  Task IDs, reset seeds, checkpoint bytes, and rollout horizon are
unchanged; only independent VAL tasks are executed concurrently.
"""
from __future__ import annotations
import argparse
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_s8r3_training_sufficiency import OUT, CKPT_ROOT, CHECKPOINT_PASSES, read_val_tasks, rollout_checkpoint


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def worker(args: tuple[str, str, list[dict], np.ndarray, np.ndarray]) -> list[dict]:
    kind, checkpoint, tasks, mean, std = args
    import torch
    return rollout_checkpoint(kind, Path(checkpoint), mean, std, tasks, torch.device("cuda"))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); args = ap.parse_args()
    with np.load(OUT / "train_obs_normalization.npz", allow_pickle=False) as n:
        mean, std = n["mean"].astype(np.float32), n["std"].astype(np.float32)
    tasks = read_val_tasks(); ctx = mp.get_context("spawn")
    jobs = []
    for kind, dirname in (("BC", "bc"), ("DIFFUSION", "diffusion")):
        for p in CHECKPOINT_PASSES:
            target = OUT / f"{kind.lower()}_pass_{p:02d}_val_rollouts.csv"
            if target.exists():
                with target.open(encoding="utf-8", newline="") as f:
                    if sum(1 for _ in csv.DictReader(f)) == 1000: continue
            ckpt = CKPT_ROOT / dirname / f"pass_{p:02d}.pt"
            for worker_index in range(args.workers):
                shard = tasks[worker_index::args.workers]
                jobs.append((kind, str(ckpt), shard, mean, std, target, worker_index))
    if not jobs:
        print("all VAL CSVs already complete"); return
    for group_start in range(0, len(jobs), args.workers):
        group = jobs[group_start:group_start + args.workers]
        kind, _, _, _, _, target, _ = group[0]
        print(f"parallel VAL {kind} {target.stem} with {len(group)} workers", flush=True)
        rows: list[dict] = []
        with ProcessPoolExecutor(max_workers=len(group), mp_context=ctx) as pool:
            futures = [pool.submit(worker, item[:5]) for item in group]
            for future in as_completed(futures): rows.extend(future.result())
        rows.sort(key=lambda r: r["task_id"]); write_csv(target, rows)
        print(f"wrote {target.name}: {len(rows)} rows", flush=True)


if __name__ == "__main__": main()
