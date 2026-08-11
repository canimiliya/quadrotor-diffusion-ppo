"""Regenerate the committed S8-R2 preview figures from the frozen cache."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts.generate_s8r2_10k_dataset import FAMILIES, generate_map, _plot_previews


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", type=Path, default=ROOT); args = ap.parse_args()
    root = args.project_root.resolve(); out = root / "artifacts" / "s8r2_10k"
    maps = [generate_map(fi, mi) for fi in range(len(FAMILIES)) for mi in range(100)]
    tasks = []
    import csv
    with (out / "task_manifest.csv").open(encoding="utf-8", newline="") as f:
        tasks = list(csv.DictReader(f))
    for row in tasks:
        row["cache_npz"] = str(out / "_cache" / row["map_id"] / f"task_{int(row['task_slot']):02d}.npz")
    _plot_previews(root, maps, tasks, out / "_cache")
    print(json.dumps({"preview_dir": str(out / "previews"), "figures": 33}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
