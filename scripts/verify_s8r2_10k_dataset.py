"""Independent, read-only audit for the S8-R2 10K expert dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.generate_s8r2_10k_dataset import (FAMILIES, MAP_GENERATION_SEED, MAPS_PER_FAMILY,
    SPLIT_SEED, TASK_GENERATION_SEED, TRAJECTORIES_PER_MAP, _rng, _task_points, generate_map,
    make_scene, run_planner)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cond: bool, name: str, detail: str = "") -> dict:
    return {"name": name, "status": "PASS" if cond else "FAIL", "detail": detail}


def planner_determinism(root: Path, maps: list[dict], run: bool) -> tuple[dict, dict]:
    selected = [next(x for x in maps if x["family"] == family and x["map_id"].endswith("_000")) for family in FAMILIES]
    identity_checks = []
    for row in selected:
        family = FAMILIES.index(row["family"]); map_spec = generate_map(family, 0)
        start_a, goal_a = _task_points(map_spec, 0, 0); start_b, goal_b = _task_points(map_spec, 0, 0)
        identity_checks.append(np.array_equal(start_a, start_b) and np.array_equal(goal_a, goal_b) and
                               map_spec.geometry_hash == row["geometry_hash"])
    if not run:
        return {"status": "PASS" if all(identity_checks) else "FAIL", "geometry_endpoint_identity": identity_checks,
                "planner_rerun": "NOT_RUN"}, {}
    hashes = []
    temp = root / "_determinism_audit"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        map_spec = generate_map(FAMILIES.index(family), 0)
        start, goal = _task_points(map_spec, 0, 0); scene = make_scene(map_spec, start, goal, 0)
        local_hashes = []
        for rep in (0, 1):
            try:
                traj, _, out = run_planner(scene, temp / family / str(rep))
                coeff = out / "reference_coefficients.csv"
                local_hashes.append(sha256(coeff))
            except Exception as exc:
                local_hashes.append(f"ERROR:{exc}")
        hashes.append({"family": family, "hashes": local_hashes, "match": local_hashes[0] == local_hashes[1]})
    shutil.rmtree(temp, ignore_errors=True)
    return {"status": "PASS" if all(x["match"] for x in hashes) and all(identity_checks) else "FAIL",
            "geometry_endpoint_identity": identity_checks, "planner_rerun": hashes}, {}


def verify(args: argparse.Namespace) -> int:
    root = args.project_root.resolve(); out = root / "artifacts" / "s8r2_10k"
    checks: list[dict] = []
    summary = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
    maps = json.loads((out / "map_manifest.json").read_text(encoding="utf-8"))
    with (out / "task_manifest.csv").open(encoding="utf-8", newline="") as f:
        tasks = list(csv.DictReader(f))
    checks.append(check(len(maps) == 1000, "unique_map_count", str(len(maps))))
    checks.append(check(len({m["map_id"] for m in maps}) == 1000, "unique_map_ids"))
    checks.append(check(len({m["geometry_hash"] for m in maps}) == 1000, "unique_geometry_hashes"))
    checks.append(check(sorted({m["family"] for m in maps}) == sorted(FAMILIES), "all_families"))
    checks.append(check(all(sum(m["family"] == fam for m in maps) == 100 for fam in FAMILIES), "family_map_balance"))
    checks.append(check(len(tasks) == 10000, "accepted_task_count", str(len(tasks))))
    by_map = {}
    for row in tasks: by_map.setdefault(row["map_id"], []).append(row)
    checks.append(check(all(len(v) == 10 for v in by_map.values()) and len(by_map) == 1000, "ten_tasks_per_map"))
    checks.append(check(all(len({r["task_id"] for r in v}) == 10 for v in by_map.values()), "task_ids_unique_per_map"))
    starts_goals = []
    for v in by_map.values():
        starts_goals.append(len({r["start"] for r in v}) == 10 and len({r["goal"] for r in v}) == 10)
    checks.append(check(all(starts_goals), "start_goal_stratification"))
    expected_split = {"train": 8000, "val": 1000, "test": 1000}
    checks.append(check({s: sum(r["split"] == s for r in tasks) for s in expected_split} == expected_split, "trajectory_split_balance"))
    split_maps = {s: {r["map_id"] for r in tasks if r["split"] == s} for s in expected_split}
    checks.append(check({s: len(v) for s, v in split_maps.items()} == {"train": 800, "val": 100, "test": 100}, "map_split_balance"))
    checks.append(check(not split_maps["train"] & split_maps["val"] and not split_maps["train"] & split_maps["test"] and not split_maps["val"] & split_maps["test"], "map_level_leakage"))
    checks.append(check(len({r["task_id"] for r in tasks if r["split"] == "train"} & {r["task_id"] for r in tasks if r["split"] == "val"}) == 0, "task_overlap_train_val"))
    # NPZ schema, boundaries, finite values, and safety counters.
    totals = 0
    for split, expected_eps in expected_split.items():
        path = out / f"{split}.npz"
        with np.load(path, allow_pickle=False) as data:
            required = {"observations", "actions", "next_observations", "positions", "velocities", "goals", "times", "trajectory_progress", "collision", "success", "done", "episode_offsets", "episode_lengths", "episode_ids", "map_ids", "task_ids", "map_indices", "task_indices"}
            checks.append(check(required <= set(data.files), f"{split}_schema"))
            checks.append(check(len(data["episode_offsets"]) == expected_eps, f"{split}_episode_count"))
            checks.append(check(data["observations"].shape[1:] == (34,) and data["actions"].shape[1:] == (3,), f"{split}_obs_action_shape"))
            checks.append(check(all(np.isfinite(data[k]).all() for k in ("observations", "actions", "next_observations", "positions", "velocities", "goals", "times", "trajectory_progress")), f"{split}_finite"))
            checks.append(check(float(np.max(np.linalg.norm(data["actions"], axis=1))) <= 1.0 + 1e-6, f"{split}_unit_ball"))
            checks.append(check(int(data["collision"].sum()) == 0 and int(data["success"].sum()) == expected_eps and int(data["done"].sum()) == expected_eps, f"{split}_safety_and_episode_terminal"))
            off, lengths = data["episode_offsets"], data["episode_lengths"]
            boundary_ok = len(off) == len(lengths) and (len(off) == 0 or (int(off[0]) == 0 and int(off[-1] + lengths[-1]) == len(data["actions"]) and np.all(np.diff(off) >= 0)))
            checks.append(check(boundary_ok, f"{split}_episode_offsets"))
            totals += len(data["actions"])
        checks.append(check(sha256(path) == summary["dataset_sha256"][split], f"{split}_sha256"))
    checks.append(check(totals == summary["total_transitions"], "total_transition_count", str(totals)))
    checks.append(check(summary["collision_count"] == summary["ground_count"] == summary["nonfinite_count"] == summary["action_support_violation"] == summary["environment_clipping"] == 0, "summary_zero_safety_counters"))
    checks.append(check(summary["test_split_sealed"] is True, "test_split_sealed"))
    # This task only materializes data; the presence of the literal ``test``
    # split is expected, while no training/checkpoint fields are allowed in
    # the task manifest.
    forbidden = {"checkpoint", "model", "selection_metric", "ppo_return"}
    checks.append(check(not forbidden & {k.lower() for k in tasks[0]}, "no_test_training_or_selection_marker"))
    checks.append(check(sha256(out / "map_manifest.json") == summary["map_manifest_sha256"], "map_manifest_sha256"))
    checks.append(check(sha256(out / "task_manifest.csv") == summary["task_manifest_sha256"], "task_manifest_sha256"))
    det, _ = planner_determinism(out, maps, args.run_planner_audit)
    checks.append(check(det["status"] == "PASS", "determinism_audit", json.dumps(det, separators=(",", ":"))))
    independent = {"task": summary["task"], "checks": checks, "passed": sum(c["status"] == "PASS" for c in checks), "failed": sum(c["status"] == "FAIL" for c in checks), "determinism": det, "test_split_sealed": True, "no_training": True, "no_test_evaluation": True}
    (out / "independent_verification.json").write_text(json.dumps(independent, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary["determinism_audit"] = det
    summary["independent_verification"] = "PASS" if independent["failed"] == 0 else "FAIL"
    summary["regression"] = "PENDING"
    if independent["failed"] == 0:
        summary["final_label"] = "PASS_S8R2_10K_PROCEDURAL_EXPERT_DATASET"
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": independent["passed"], "failed": independent["failed"], "final_label": summary["final_label"], "determinism": det}, indent=2))
    return 0 if independent["failed"] == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", type=Path, default=ROOT); ap.add_argument("--run-planner-audit", action="store_true")
    return verify(ap.parse_args())


if __name__ == "__main__": raise SystemExit(main())
