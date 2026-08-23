"""Task-sharded eager-exact U-Net VAL evaluator for S8-R4."""
from __future__ import annotations
import argparse, csv, multiprocessing as mp, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_s8r4_unet_audit import OUT, CKPT_ROOT, CHECKPOINT_PASSES, read_val_tasks, rollout_unet

def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def worker(args):
    checkpoint, tasks, mean, std = args
    import torch
    return rollout_unet(Path(checkpoint), mean, std, tasks, torch.device("cuda"))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); args = ap.parse_args()
    with np.load(OUT.parent / "s8r3" / "train_obs_normalization.npz", allow_pickle=False) as n:
        mean, std = n["mean"].astype(np.float32), n["std"].astype(np.float32)
    tasks = read_val_tasks(); ctx = mp.get_context("spawn")
    for p in CHECKPOINT_PASSES:
        target = OUT / f"unet_pass_{p:02d}_val_rollouts.csv"
        if target.exists():
            with target.open(encoding="utf-8", newline="") as f:
                if sum(1 for _ in csv.DictReader(f)) == 1000: print(f"skip complete {target.name}"); continue
        jobs = [(str(CKPT_ROOT / f"pass_{p:02d}.pt"), tasks[i::args.workers], mean, std) for i in range(args.workers)]
        rows = []
        print(f"parallel VAL UNET pass {p} with {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(futures): rows.extend(future.result())
        rows.sort(key=lambda r: r["task_id"]); write_csv(target, rows); print(f"wrote {target.name}: {len(rows)}", flush=True)

if __name__ == "__main__": main()
