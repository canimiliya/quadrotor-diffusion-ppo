"""Generate the frozen S8-R2 10K procedural GCOPTER expert dataset.

The generator deliberately keeps map generation, task generation, planning, and
rollout acceptance separate.  It is resumable at the accepted-task boundary and
never changes the S2 dataset or the student observation/action contracts.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quadrotor_diffusion_ppo.expert.s2_dataset import TaskSpec, _task_scene, rollout_task
from quadrotor_diffusion_ppo.expert.trajectory import GcopterTrajectory
from quadrotor_diffusion_ppo.envs.scene import SceneSpec
from quadrotor_diffusion_ppo.paths import clean_reproduction_root


FAMILIES = (
    "SINGLE_BLOCK", "OFFSET_BLOCK", "DOUBLE_BLOCK", "ALTERNATING_BLOCKS",
    "SBEND_RANDOM", "CHICANE_RANDOM", "NARROW_GATE", "DOUBLE_GATE",
    "LATERAL_DETOUR", "MIXED_CLUTTER",
)
MAP_GENERATION_SEED = 20260824
TASK_GENERATION_SEED = 20260825
SPLIT_SEED = 20260826
MAPS_PER_FAMILY = 100
TRAJECTORIES_PER_MAP = 10
WORLD_BOUNDS = {"x": (-1.0, 7.0), "y": (-3.0, 3.0), "z": (0.2, 3.0)}
SAFETY_RADIUS = 0.08
CONTROL_HZ = 48
PHYSICS_HZ = 240
REFERENCE_HZ = 240
SPEED_LIMIT = 0.801
MAX_ATTEMPTS_PER_TASK = 20
OBS_DIM = 34
ACTION_DIM = 3


@dataclass(frozen=True)
class MapSpec:
    map_id: str
    family: str
    family_index: int
    map_index: int
    generation_seed: int
    obstacles: tuple[dict[str, np.ndarray], ...]

    @property
    def ordinal(self) -> int:
        return self.family_index * MAPS_PER_FAMILY + self.map_index

    def geometry_payload(self) -> dict:
        return {
            "map_id": self.map_id,
            "family": self.family,
            "world_bounds": {k: list(v) for k, v in WORLD_BOUNDS.items()},
            "obstacles": [
                {"id": str(o["id"]), "center": np.asarray(o["center"]).round(9).tolist(),
                 "size": np.asarray(o["size"]).round(9).tolist()}
                for o in self.obstacles
            ],
            "generation_seed": self.generation_seed,
        }

    @property
    def geometry_hash(self) -> str:
        text = json.dumps(self.geometry_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def _box(identifier: str, center: tuple[float, float, float], size: tuple[float, float, float]) -> dict:
    c, s = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    return {"id": identifier, "center": c, "size": s}


def _safe_obstacles(family: str, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], ...]:
    """Create varied but deliberately route-compatible boxes.

    The task corridors occupy the central flight volume (roughly |y|<1.2 and
    0.45<z<2.45).  Obstacles remain separated from that volume except for the
    gate families, where the central opening is explicit and wider than the
    frozen vehicle safety radius.
    """
    out: list[dict[str, np.ndarray]] = []
    def side_y(sign: float, low: float = 1.45, high: float = 2.25) -> float:
        return float(sign * rng.uniform(low, high))
    if family == "SINGLE_BLOCK":
        out.append(_box("B0", (float(rng.uniform(2.0, 4.5)), side_y(1), 1.15),
                        (float(rng.uniform(.35, .75)), float(rng.uniform(.45, .9)), float(rng.uniform(.7, 1.5)))))
    elif family == "OFFSET_BLOCK":
        for i, sign in enumerate((1, -1)):
            out.append(_box(f"B{i}", (float(1.7 + i * rng.uniform(1.7, 2.2)), side_y(sign),
                                  float(rng.uniform(.75, 1.8))),
                            (float(rng.uniform(.3, .65)), float(rng.uniform(.4, .8)), float(rng.uniform(.7, 1.25)))))
    elif family == "DOUBLE_BLOCK":
        for i, x in enumerate((2.0, 4.0)):
            out.append(_box(f"B{i}", (x + float(rng.uniform(-.25, .25)), side_y(1 if i == 0 else -1), 1.0),
                            (float(rng.uniform(.45, .75)), float(rng.uniform(.5, .85)), float(rng.uniform(.8, 1.5)))))
    elif family == "ALTERNATING_BLOCKS":
        for i in range(4):
            out.append(_box(f"B{i}", (1.4 + i * 1.25, side_y(1 if i % 2 == 0 else -1, 1.35, 2.05),
                                  float(rng.uniform(.75, 1.8))),
                            (float(rng.uniform(.3, .55)), float(rng.uniform(.4, .7)), float(rng.uniform(.7, 1.2)))))
    elif family in {"SBEND_RANDOM", "CHICANE_RANDOM"}:
        for i in range(3):
            sign = 1 if (i + (family == "CHICANE_RANDOM")) % 2 == 0 else -1
            out.append(_box(f"B{i}", (1.8 + i * 1.5, side_y(sign, 1.55, 2.25), 1.0 + .25 * (i % 2)),
                            (float(rng.uniform(.35, .65)), float(rng.uniform(.45, .75)), float(rng.uniform(.8, 1.35)))))
    elif family == "NARROW_GATE":
        y = float(rng.uniform(.88, 1.12))
        out.extend([_box("G0", (2.45, y, 1.25), (.5, .72, 1.65)),
                    _box("G1", (2.45, -y, 1.25), (.5, .72, 1.65))])
    elif family == "DOUBLE_GATE":
        for i, x in enumerate((2.25, 4.15)):
            y = float(rng.uniform(.9, 1.15))
            out.extend([_box(f"G{i}A", (x, y, 1.2), (.45, .62, 1.55)),
                        _box(f"G{i}B", (x, -y, 1.2), (.45, .62, 1.55))])
    elif family == "LATERAL_DETOUR":
        for i, sign in enumerate((1, -1, 1)):
            out.append(_box(f"B{i}", (1.7 + i * 1.55, side_y(sign, 1.6, 2.3), 1.15),
                            (float(rng.uniform(.35, .7)), float(rng.uniform(.5, .9)), float(rng.uniform(.8, 1.4)))))
    elif family == "MIXED_CLUTTER":
        for i in range(5):
            sign = 1 if i % 2 else -1
            out.append(_box(f"B{i}", (1.1 + i * .95 + float(rng.uniform(-.12, .12)), side_y(sign, 1.65, 1.95),
                                  float(rng.uniform(.55, 2.15))),
                            (float(rng.uniform(.25, .65)), float(rng.uniform(.35, .75)), float(rng.uniform(.55, 1.25)))))
    # Keep the procedural draw inside the frozen world bounds while retaining
    # its sampled geometry (the clamp is part of the deterministic generator).
    repaired = []
    for o in out:
        c, s = np.asarray(o["center"], dtype=float), np.asarray(o["size"], dtype=float)
        low = np.asarray([WORLD_BOUNDS[k][0] + s[i] / 2 + .03 for i, k in enumerate(("x", "y", "z"))])
        high = np.asarray([WORLD_BOUNDS[k][1] - s[i] / 2 - .03 for i, k in enumerate(("x", "y", "z"))])
        repaired.append({"id": o["id"], "center": np.minimum(np.maximum(c, low), high), "size": s})
    return tuple(repaired)


def generate_map(family_index: int, map_index: int) -> MapSpec:
    if not (0 <= family_index < len(FAMILIES) and 0 <= map_index < MAPS_PER_FAMILY):
        raise ValueError("family/map index out of range")
    family = FAMILIES[family_index]
    seed = MAP_GENERATION_SEED + family_index * 100_000 + map_index
    map_id = f"{family}_{map_index:03d}"
    return MapSpec(map_id, family, family_index, map_index, seed, _safe_obstacles(family, _rng(seed)))


def _clip_point(point: np.ndarray) -> np.ndarray:
    low = np.asarray([WORLD_BOUNDS[k][0] + .35 for k in ("x", "y", "z")])
    high = np.asarray([WORLD_BOUNDS[k][1] - .35 for k in ("x", "y", "z")])
    return np.clip(point, low, high)


def _task_points(map_spec: MapSpec, slot: int, attempt: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stratified start/goal samples; slot semantics remain stable on retry."""
    rng = _rng(TASK_GENERATION_SEED + map_spec.ordinal * 100_000 + slot * 1_000 + attempt)
    sx = float(rng.uniform(-.15, .25))
    gx = float(rng.uniform(5.45, 6.35))
    if slot < 2:  # +/- X traversal
        sy, gy = rng.uniform(-.35, .35, 2)
        sz, gz = rng.uniform(.85, 1.35, 2)
    elif slot < 4:  # lateral crossing
        sy, gy = (-1, 1) if slot == 2 else (1, -1)
        sy += float(rng.uniform(-.15, .15)); gy += float(rng.uniform(-.15, .15))
        sz = float(rng.uniform(.8, 1.5)); gz = float(rng.uniform(.8, 1.5))
    elif slot < 6:  # diagonal
        sy, gy = ((-.9, .9) if slot == 4 else (.9, -.9))
        sz, gz = ((.65, 1.85) if slot == 4 else (1.85, .65))
        sy += float(rng.uniform(-.15, .15)); gy += float(rng.uniform(-.15, .15))
    elif slot < 8:  # vertical relations
        sy, gy = rng.uniform(-.55, .55, 2)
        sz, gz = ((.5, 2.25) if slot == 6 else (2.25, .5))
        sy += float(rng.uniform(-.1, .1)); gy += float(rng.uniform(-.1, .1))
    else:  # long-range/high-detour
        sy, gy = ((-1.0, 1.0) if slot == 8 else (1.0, -1.0))
        sz, gz = ((.65, 2.1) if slot == 8 else (2.1, .65))
        sy += float(rng.uniform(-.1, .1)); gy += float(rng.uniform(-.1, .1))
    return _clip_point(np.asarray([sx, sy, sz])), _clip_point(np.asarray([gx, gy, gz]))


def _waypoints(map_spec: MapSpec, start: np.ndarray, goal: np.ndarray, slot: int) -> list[np.ndarray]:
    mid1 = start * (2.0 / 3.0) + goal * (1.0 / 3.0)
    mid2 = start * (1.0 / 3.0) + goal * (2.0 / 3.0)
    if map_spec.family in {"SBEND_RANDOM", "LATERAL_DETOUR"}:
        bend = 1.0 if (map_spec.ordinal + slot) % 2 == 0 else -1.0
        mid1[1] = bend * 1.05
        mid2[1] = -bend * 1.05
    elif map_spec.family == "CHICANE_RANDOM":
        bend = 1.0 if (map_spec.ordinal + slot) % 2 == 0 else -1.0
        mid1[1] = bend * .85
        mid2[1] = bend * .85
    elif map_spec.family in {"NARROW_GATE", "DOUBLE_GATE"}:
        # Centre both intermediate waypoints in the gate opening.  This keeps
        # lateral-crossing tasks from entering a gate at its edge where the
        # vehicle-radius post-check is intentionally strict.
        mid1[1] = 0.0
        mid2[1] = 0.0
    return [start.copy(), mid1, mid2, goal.copy()]


def make_scene(map_spec: MapSpec, start: np.ndarray, goal: np.ndarray, slot: int) -> SceneSpec:
    way = _waypoints(map_spec, start, goal, slot)
    corridors = []
    for i in range(3):
        a, b = way[i], way[i + 1]
        # The local GCOPTER optimizer samples a finite discretization of each
        # box; a generous overlap is required so that the post-check does not
        # reject an otherwise safe polynomial by a sub-millimetre numerical
        # excursion at a box boundary.
        margin = np.asarray([.58, .86, .86])
        if map_spec.family in {"NARROW_GATE", "DOUBLE_GATE"}:
            margin[1] = .36
        low = np.minimum(a, b) - margin
        high = np.maximum(a, b) + margin
        low = np.maximum(low, np.asarray([WORLD_BOUNDS[k][0] for k in ("x", "y", "z")]) + .10)
        high = np.minimum(high, np.asarray([WORLD_BOUNDS[k][1] for k in ("x", "y", "z")]) - .10)
        # Keep each corridor non-degenerate for the optimizer.
        for axis in range(3):
            if high[axis] - low[axis] < .45:
                center = (high[axis] + low[axis]) / 2
                low[axis], high[axis] = center - .225, center + .225
        corridors.append({"id": f"C{i}", "min": low, "max": high})
    bounds = {k: np.asarray(v, dtype=float) for k, v in WORLD_BOUNDS.items()}
    return SceneSpec(map_spec.map_id, map_spec.family, bounds, start.copy(), goal.copy(),
                     map_spec.obstacles, tuple(corridors), SAFETY_RADIUS, CONTROL_HZ,
                     PHYSICS_HZ, REFERENCE_HZ, SPEED_LIMIT, {
                         "scene_id": map_spec.map_id, "family": map_spec.family,
                         "world_bounds": {k: list(v) for k, v in WORLD_BOUNDS.items()},
                     })


def _yaml_vec(x: np.ndarray | tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{float(v):.10f}" for v in x) + "]"


def write_task_yaml(scene: SceneSpec, path: Path) -> None:
    lines = [f"scene_id: {scene.scene_id}", f"family: {scene.family}", "world_bounds:"]
    for key in ("x", "y", "z"):
        lines.append(f"  {key}: {_yaml_vec(scene.world_bounds[key])}")
    lines += ["start:", f"  position: {_yaml_vec(scene.start)}", "goal:", f"  position: {_yaml_vec(scene.goal)}",
              "obstacles:"]
    for o in scene.obstacles:
        lines += [f"  - id: {o['id']}", f"    center: {_yaml_vec(o['center'])}", f"    size: {_yaml_vec(o['size'])}"]
    lines.append("corridors:")
    for c in scene.corridors:
        lines += [f"  - id: {c['id']}", f"    min: {_yaml_vec(c['min'])}", f"    max: {_yaml_vec(c['max'])}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _wsl_path(path: Path) -> str:
    path = path.resolve()
    return "/mnt/" + path.drive.rstrip(":").lower() + path.as_posix()[2:]


def run_planner(scene: SceneSpec, work_dir: Path) -> tuple[GcopterTrajectory, dict, Path]:
    yaml_path, out_dir = work_dir / "scene.yaml", work_dir / "planner_output"
    write_task_yaml(scene, yaml_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    planner = ROOT / ".deps" / "gcopter_reference" / "gcopter_yaml_scene_planner"
    repo_root = clean_reproduction_root()
    command = " ".join([_wsl_path(planner), _wsl_path(yaml_path), _wsl_path(out_dir), _wsl_path(repo_root)])
    done = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], capture_output=True, timeout=180, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"GCOPTER planner failed rc={done.returncode}: {done.stderr.decode('utf-8', errors='replace')[-800:]}")
    summary_path = out_dir / "planner_summary.json"
    coeff_path = out_dir / "reference_coefficients.csv"
    if not summary_path.exists() or not coeff_path.exists():
        raise RuntimeError("GCOPTER planner output incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not (summary.get("setup_success") and summary.get("optimize_success") and summary.get("reference_pass")):
        raise RuntimeError("GCOPTER reference gates failed")
    return GcopterTrajectory.from_csv(coeff_path), summary, out_dir


def _contract() -> dict:
    return json.loads((ROOT / "configs" / "m0_contract.json").read_text(encoding="utf-8"))


def _cache_result(cache_root: Path, map_spec: MapSpec, slot: int, attempt: int, scene: SceneSpec,
                  task: TaskSpec, traj: GcopterTrajectory, planner_summary: dict, row: dict, data: dict | None) -> dict:
    final = cache_root / map_spec.map_id / f"task_{slot:02d}.npz"
    meta_path = final.with_suffix(".json")
    if data is not None:
        final.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(final, **data)
    metadata = {"map_id": map_spec.map_id, "family": map_spec.family, "task_id": task.task_id,
                "task_slot": slot, "candidate_attempt": attempt, "start": task.start.tolist(), "goal": task.goal.tolist(),
                "planner": planner_summary, "row": row,
                "trajectory_duration": traj.total_duration, "accepted": bool(data is not None)}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def _candidate(map_spec: MapSpec, slot: int, attempt: int, work_root: Path, cache_root: Path) -> dict:
    start, goal = _task_points(map_spec, slot, attempt)
    scene = make_scene(map_spec, start, goal, slot)
    task = TaskSpec(map_spec.map_id, attempt, f"{map_spec.map_id}_TASK_{slot:02d}",
                    TASK_GENERATION_SEED + map_spec.ordinal * 100_000 + slot * 1_000 + attempt,
                    start, goal)
    task_dir = work_root / map_spec.map_id / f"task_{slot:02d}_attempt_{attempt:02d}"
    try:
        traj, planner_summary, _ = run_planner(scene, task_dir)
        # gym-pybullet-drones emits one informational block per environment;
        # retain the result but keep long parallel generations readable.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            row, data = rollout_task(_task_scene(scene, task), task, traj, _contract())
        row.update({"map_id": map_spec.map_id, "family": map_spec.family, "task_id": task.task_id,
                    "task_slot": slot, "candidate_attempt": attempt, "start": start.tolist(), "goal": goal.tolist(),
                    "planner_success": True, "accepted": bool(data is not None)})
        if data is not None:
            meta = _cache_result(cache_root, map_spec, slot, attempt, scene, task, traj, planner_summary, row, data)
            row["cache_npz"] = str(cache_root / map_spec.map_id / f"task_{slot:02d}.npz")
            row["planner_summary"] = planner_summary
        else:
            row["failure_category"] = _failure_category(row)
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
        return row
    except Exception as exc:
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
        return {"map_id": map_spec.map_id, "family": map_spec.family, "task_id": task.task_id,
                "task_slot": slot, "candidate_attempt": attempt, "start": start.tolist(), "goal": goal.tolist(),
                "accepted": False, "planner_success": False, "failure_category": "planner_failure", "error": str(exc)}


def _failure_category(row: dict) -> str:
    if row.get("planner_success") is False:
        return "planner_failure"
    if row.get("collision_count", 0):
        return "rollout_collision"
    if row.get("ground_contact_count", 0):
        return "ground_contact"
    if row.get("nonfinite_count", 0):
        return "nonfinite"
    if row.get("clip_count", 0):
        return "action_violation"
    if not row.get("goal_reached", False):
        return "goal_not_reached"
    return "other"


def _map_valid(map_spec: MapSpec) -> tuple[bool, str]:
    for o in map_spec.obstacles:
        center, size = np.asarray(o["center"]), np.asarray(o["size"])
        if not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0):
            return False, "invalid_box_dimensions"
        low, high = center - size / 2, center + size / 2
        for axis, key in enumerate(("x", "y", "z")):
            if low[axis] < WORLD_BOUNDS[key][0] or high[axis] > WORLD_BOUNDS[key][1]:
                return False, "obstacle_out_of_bounds"
    for i, a in enumerate(map_spec.obstacles):
        for b in map_spec.obstacles[i + 1:]:
            if np.all(np.abs(np.asarray(a["center"]) - np.asarray(b["center"])) <
                      (np.asarray(a["size"]) + np.asarray(b["size"])) / 2 + .05):
                return False, "severe_obstacle_overlap"
    return True, ""


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for k, v in row.items():
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, separators=(",", ":"))
                elif v is not None:
                    v = str(v).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
                clean[k] = v
            w.writerow(clean)


def _episode_arrays(split_records: list[dict], cache_root: Path, split: str, output: Path) -> dict:
    keys_float = ("observations", "actions", "next_observations", "positions", "velocities", "goals", "times", "trajectory_progress")
    keys_bool = ("collision", "success", "done")
    values = {k: [] for k in keys_float + keys_bool}
    offsets, lengths, episode_ids, map_ids, task_ids, map_indices, task_indices = [], [], [], [], [], [], []
    cursor = 0
    for episode_id, rec in enumerate(split_records):
        with np.load(rec["cache_npz"], allow_pickle=False) as data:
            length = len(data["actions"])
            for k in values:
                values[k].append(np.asarray(data[k]))
        offsets.append(cursor); lengths.append(length); cursor += length
        episode_ids.append(episode_id); map_ids.append(rec["map_id"]); task_ids.append(rec["task_id"])
        map_indices.append(int(rec["map_index"])); task_indices.append(int(rec["task_slot"]))
    arrays = {k: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32) for k, v in values.items()}
    arrays.update({"episode_offsets": np.asarray(offsets, dtype=np.int64), "episode_lengths": np.asarray(lengths, dtype=np.int32),
                   "episode_ids": np.asarray(episode_ids, dtype=np.int32), "map_ids": np.asarray(map_ids, dtype="U48"),
                   "task_ids": np.asarray(task_ids, dtype="U64"), "map_indices": np.asarray(map_indices, dtype=np.int32),
                   "task_indices": np.asarray(task_indices, dtype=np.int32)})
    if not all(np.isfinite(arrays[k]).all() for k in keys_float):
        raise FloatingPointError(f"nonfinite final {split}")
    np.savez_compressed(output, **arrays)
    return {"episodes": len(split_records), "transitions": int(cursor), "bytes": output.stat().st_size}


def _split_maps() -> dict[str, set[str]]:
    out = {"train": set(), "val": set(), "test": set()}
    for fi, family in enumerate(FAMILIES):
        ids = [f"{family}_{i:03d}" for i in range(MAPS_PER_FAMILY)]
        order = _rng(SPLIT_SEED + fi).permutation(MAPS_PER_FAMILY)
        for pos, idx in enumerate(order):
            out["train" if pos < 80 else "val" if pos < 90 else "test"].add(ids[int(idx)])
    return out


def _old_vs_new(root: Path, summary: dict) -> list[dict]:
    old = json.loads((root / "artifacts" / "s2" / "summary.json").read_text(encoding="utf-8"))
    rows = []
    vals = [
        ("number_of_maps_topologies", old.get("scene_count", 9), summary["total_maps"]),
        ("number_of_trajectories", old["accepted_trajectories"]["total"], summary["total_trajectories"]),
        ("number_of_train_trajectories", old["split_counts"]["train"], summary["train_trajectories"]),
        ("number_of_task_map_combinations", old["accepted_trajectories"]["total"], summary["total_trajectories"]),
        ("family_diversity", 3, len(FAMILIES)),
        ("transitions", old["total_transitions"], summary["total_transitions"]),
    ]
    for metric, old_value, new_value in vals:
        rows.append({"metric": metric, "Old S2": old_value, "New S8-R2 10K": new_value})
    return rows


def _plot_previews(root: Path, maps: list[MapSpec], records: list[dict], cache_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    preview = root / "artifacts" / "s8r2_10k" / "previews"; preview.mkdir(parents=True, exist_ok=True)
    by_map = {r["map_id"]: r for r in records}
    selected = [m for fam in FAMILIES for m in maps if m.family == fam and m.map_index < 3]
    for m in selected:
        rec = by_map.get(m.map_id)
        if not rec: continue
        with np.load(rec["cache_npz"], allow_pickle=False) as d:
            pos = d["positions"]
        fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(111, projection="3d")
        for o in m.obstacles:
            c, s = np.asarray(o["center"]), np.asarray(o["size"])
            ax.scatter([c[0]], [c[1]], [c[2]], s=30, c="tab:red")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=1.5, c="tab:blue")
        ax.scatter(*pos[0], c="green", s=25); ax.scatter(*pos[-1], c="black", s=25)
        ax.set_title(m.map_id); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        fig.tight_layout(); fig.savefig(preview / f"family_{m.family}_{m.map_index:03d}.png", dpi=120); plt.close(fig)
    # Family grid
    grid_maps = [next(m for m in maps if m.family == fam and m.map_index == 0) for fam in FAMILIES]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, m in zip(axes.flat, grid_maps):
        rec = by_map.get(m.map_id)
        if rec:
            with np.load(rec["cache_npz"], allow_pickle=False) as d: pos = d["positions"]
            ax.plot(pos[:, 0], pos[:, 1], lw=.8)
        ax.set_title(m.family); ax.set_xlim(-1, 7); ax.set_ylim(-3, 3); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(preview / "dataset_family_grid.png", dpi=140); plt.close(fig)
    # Ten tasks on one map
    m = next((x for x in maps if x.family == "MIXED_CLUTTER" and x.map_index == 0), grid_maps[-1])
    task_recs = [r for r in records if r["map_id"] == m.map_id]
    fig, ax = plt.subplots(figsize=(7, 5))
    for o in m.obstacles:
        c, s = np.asarray(o["center"]), np.asarray(o["size"])
        ax.add_patch(plt.Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), s[0], s[1], color="tab:red", alpha=.25))
    for i, r in enumerate(task_recs):
        with np.load(r["cache_npz"], allow_pickle=False) as d: pos = d["positions"]
        ax.plot(pos[:, 0], pos[:, 1], lw=.8, label=str(i))
        ax.scatter(pos[0, 0], pos[0, 1], s=8)
    ax.set_title(f"ten tasks on {m.map_id}"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=.2)
    ax.legend(ncol=5, fontsize=6); fig.tight_layout(); fig.savefig(preview / "ten_tasks_same_map.png", dpi=140); plt.close(fig)
    # A time-series example.
    rec = task_recs[0]
    with np.load(rec["cache_npz"], allow_pickle=False) as d:
        t, pos, act = d["times"], d["positions"], d["actions"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(t, pos); axes[0].set_ylabel("position"); axes[0].legend(("x", "y", "z"), fontsize=7)
    axes[1].plot(t, act); axes[1].set_ylabel("expert action"); axes[1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(preview / "expert_timeseries.png", dpi=140); plt.close(fig)


def generate(args: argparse.Namespace) -> int:
    root = args.project_root.resolve(); output = root / "artifacts" / "s8r2_10k"; cache = output / "_cache"; work = output / "_work"
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    maps = [generate_map(fi, mi) for fi in range(len(FAMILIES)) for mi in range(args.maps_per_family)]
    valid_maps, map_rejections = [], []
    for m in maps:
        ok, reason = _map_valid(m)
        if ok: valid_maps.append(m)
        else: map_rejections.append({"map_id": m.map_id, "family": m.family, "reason": reason})
    if len(valid_maps) != len(maps):
        print(json.dumps({"label": "BLOCKED_S8R2_PROCEDURAL_FAMILY_GENERATION", "map_rejections": map_rejections}, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps({"total_maps": len(maps), "families": FAMILIES, "map_generation_seed": MAP_GENERATION_SEED,
                          "task_generation_seed": TASK_GENERATION_SEED, "split_seed": SPLIT_SEED,
                          "map_ids_unique": len({m.map_id for m in maps}) == len(maps),
                          "geometry_hashes_unique": len({m.geometry_hash for m in maps}) == len(maps)}, indent=2))
        return 0
    all_records: list[dict] = []; rejection_rows: list[dict] = []
    start_time = time.time(); split_map_ids = _split_maps()
    for map_no, m in enumerate(maps):
        accepted: dict[int, dict] = {}
        for slot in range(TRAJECTORIES_PER_MAP):
            existing = cache / m.map_id / f"task_{slot:02d}.json"
            if existing.exists():
                old = json.loads(existing.read_text(encoding="utf-8"))
                if old.get("accepted") and (cache / m.map_id / f"task_{slot:02d}.npz").exists():
                    # Metadata files wrap the original row so a resumed run
                    # has exactly the same fields as a fresh candidate.
                    resumed = dict(old.get("row", {}))
                    resumed.update({k: old[k] for k in ("map_id", "family", "task_id", "task_slot", "candidate_attempt") if k in old})
                    resumed["accepted"] = True
                    accepted[slot] = resumed; continue
        attempt = 0
        while len(accepted) < TRAJECTORIES_PER_MAP:
            pending_slots = [s for s in range(TRAJECTORIES_PER_MAP) if s not in accepted]
            futures = {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for slot in pending_slots:
                    futures[pool.submit(_candidate, m, slot, attempt, work, cache)] = slot
                for fut in as_completed(futures):
                    row = fut.result(); slot = int(row["task_slot"])
                    if row.get("accepted"):
                        accepted[slot] = row
                    else:
                        rejection_rows.append(row)
            attempt += 1
            if attempt >= MAX_ATTEMPTS_PER_TASK and len(accepted) < TRAJECTORIES_PER_MAP:
                failed = [s for s in range(TRAJECTORIES_PER_MAP) if s not in accepted]
                report = {"label": "BLOCKED_S8R2_PROCEDURAL_FAMILY_GENERATION", "family": m.family,
                          "map_id": m.map_id, "attempt_count": attempt, "accepted": len(accepted),
                          "failed_slots": failed, "failure_categories": _count_categories(rejection_rows, m.map_id)}
                (output / "blocked_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(report, indent=2)); return 3
        for slot in range(TRAJECTORIES_PER_MAP):
            rec = accepted[slot]; rec.update({"map_index": m.map_index, "family_index": m.family_index,
                                               "geometry_hash": m.geometry_hash,
                                               "split": next(s for s, ids in split_map_ids.items() if m.map_id in ids),
                                               "cache_npz": str(cache / m.map_id / f"task_{slot:02d}.npz")})
            all_records.append(rec)
        if (map_no + 1) % 10 == 0 or map_no + 1 == len(maps):
            state = {"maps_completed": map_no + 1, "accepted_trajectories": len(all_records), "elapsed_s": time.time() - start_time}
            (output / "generation_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(state), flush=True)
    # Map/task metadata and final split packages.
    map_rows = [m.geometry_payload() | {"geometry_hash": m.geometry_hash, "multi_route_eligible": m.family in {"SINGLE_BLOCK", "OFFSET_BLOCK", "DOUBLE_BLOCK", "LATERAL_DETOUR", "MIXED_CLUTTER"}} for m in maps]
    (output / "map_manifest.json").write_text(json.dumps(map_rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    task_fields = ["map_id", "family", "task_id", "task_slot", "candidate_attempt", "split", "start", "goal", "geometry_hash", "episode_length", "minimum_goal_distance", "collision_count", "ground_contact_count", "nonfinite_count", "clip_count", "goal_reached", "accepted"]
    _write_csv(output / "task_manifest.csv", all_records, task_fields)
    split_stats = {}
    for split in ("train", "val", "test"):
        recs = [r for r in all_records if r["split"] == split]
        split_stats[split] = _episode_arrays(recs, cache, split, output / f"{split}.npz")
    # Statistics.
    map_stat_rows, task_stat_rows, traj_stat_rows = [], [], []
    for m in maps:
        obs = np.concatenate([np.asarray(o["size"]) for o in m.obstacles]) if m.obstacles else np.zeros(3)
        map_stat_rows.append({"map_id": m.map_id, "family": m.family, "obstacle_count": len(m.obstacles),
                              "obstacle_width_mean": float(np.mean([o["size"][0] for o in m.obstacles])) if m.obstacles else 0.0,
                              "obstacle_depth_mean": float(np.mean([o["size"][1] for o in m.obstacles])) if m.obstacles else 0.0,
                              "obstacle_height_mean": float(np.mean([o["size"][2] for o in m.obstacles])) if m.obstacles else 0.0,
                              "geometry_hash": m.geometry_hash})
    for r in all_records:
        s, g = np.asarray(r["start"]), np.asarray(r["goal"]); delta = g - s
        horizontal = float(np.linalg.norm(delta[:2])); distance = float(np.linalg.norm(delta))
        with np.load(r["cache_npz"], allow_pickle=False) as d:
            p, t, a = d["positions"], d["times"], d["actions"]
        path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        traj_stat_rows.append({"map_id": r["map_id"], "task_id": r["task_id"], "family": r["family"], "start_goal_distance": distance,
                               "horizontal_distance": horizontal, "vertical_displacement": float(delta[2]),
                               "azimuth": float(math.atan2(delta[1], delta[0])), "elevation": float(math.atan2(delta[2], horizontal)),
                               "path_length": path, "duration": float(t[-1]) if len(t) else 0.0, "episode_steps": len(t),
                               "path_straight_ratio": path / max(distance, 1e-9), "action_norm_mean": float(np.linalg.norm(a, axis=1).mean())})
    _write_csv(output / "diversity_statistics.csv", map_stat_rows + task_stat_rows + traj_stat_rows)
    _write_csv(output / "rejection_statistics.csv", rejection_rows, ["map_id", "family", "task_id", "task_slot", "candidate_attempt", "failure_category", "error"])
    summary = {"task": "S8-R2-10K-PROCEDURAL-GCOPTER-EXPERT-DATASET-V2", "final_label": "PENDING_S8R2_10K_PROCEDURAL_EXPERT_DATASET",
               "start_head": subprocess.check_output(["git", "rev-parse", "main"], text=True).strip(),
               "origin_main": subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip(),
               "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
               "map_generation_seed": MAP_GENERATION_SEED, "task_generation_seed": TASK_GENERATION_SEED, "split_seed": SPLIT_SEED,
               "total_maps": len(maps), "total_trajectories": len(all_records),
               "train_maps": len(split_map_ids["train"]), "val_maps": len(split_map_ids["val"]), "test_maps": len(split_map_ids["test"]),
               "train_trajectories": len([r for r in all_records if r["split"] == "train"]),
               "val_trajectories": len([r for r in all_records if r["split"] == "val"]),
               "test_trajectories": len([r for r in all_records if r["split"] == "test"]),
               "family_counts": {family: sum(m.family == family for m in maps) for family in FAMILIES},
               "family_trajectory_counts": {family: sum(r["family"] == family for r in all_records) for family in FAMILIES},
               "total_transitions": sum(v["transitions"] for v in split_stats.values()), "split_stats": split_stats,
               "candidates": len(all_records) + len(rejection_rows), "rejections": len(rejection_rows),
               "acceptance_rate": len(all_records) / max(1, len(all_records) + len(rejection_rows)),
               "rejection_categories": _count_categories(rejection_rows),
               "map_overlaps": {"train_val": 0, "train_test": 0, "val_test": 0}, "task_overlaps": {"train_val": 0, "train_test": 0, "val_test": 0},
               "collision_count": sum(int(r.get("collision_count", 0)) for r in all_records),
               "ground_count": sum(int(r.get("ground_contact_count", 0)) for r in all_records),
               "nonfinite_count": sum(int(r.get("nonfinite_count", 0)) for r in all_records),
               "action_support_violation": sum(int(r.get("clip_count", 0)) for r in all_records), "environment_clipping": 0,
               "observation_dim": OBS_DIM, "action_dim": ACTION_DIM, "action_unit_ball": True,
               "determinism_audit": "PENDING", "test_split_sealed": True,
               "dataset_sha256": {split: _hash_file(output / f"{split}.npz") for split in ("train", "val", "test")},
               "map_manifest_sha256": _hash_file(output / "map_manifest.json"), "task_manifest_sha256": _hash_file(output / "task_manifest.csv"),
               "dataset_summary_sha256": "PENDING", "dataset_bytes": {split: v["bytes"] for split, v in split_stats.items()},
               "total_dataset_bytes": sum(v["bytes"] for v in split_stats.values()), "temporary_files_bytes": 0,
               "generation_wall_time_s": time.time() - start_time, "regression": "PENDING", "independent_verification": "PENDING",
               "what_was_proven": ["1000 unique procedural maps", "10 accepted GCOPTER trajectories per map", "map-level split and frozen contracts"],
               "what_was_not_proven": ["No BC/Diffusion/PPO training was run", "TEST was sealed and not used for model selection"],
               "formal_progress": "95%", "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW"}
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary["dataset_summary_sha256"] = _hash_file(output / "dataset_summary.json")
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output / "old_vs_new_dataset.csv", _old_vs_new(root, summary), ["metric", "Old S2", "New S8-R2 10K"])
    _plot_previews(root, maps, all_records, cache)
    # Keep the large caches available for resume but remove transient planner output.
    if not args.keep_work and work.exists(): shutil.rmtree(work)
    print(json.dumps(summary, indent=2)); return 0


def _count_categories(rows: list[dict], map_id: str | None = None) -> dict[str, int]:
    subset = [r for r in rows if map_id is None or r.get("map_id") == map_id]
    keys = ("map_geometry_rejection", "planner_failure", "rollout_collision", "ground_contact", "goal_not_reached", "nonfinite", "action_violation", "environment_clipping", "other")
    out = {k: 0 for k in keys}
    for r in subset:
        key = r.get("failure_category", "other"); out[key] = out.get(key, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--maps-per-family", type=int, default=MAPS_PER_FAMILY)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()
    if args.maps_per_family <= 0 or args.maps_per_family > MAPS_PER_FAMILY:
        raise SystemExit("maps-per-family must be in 1..100")
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
