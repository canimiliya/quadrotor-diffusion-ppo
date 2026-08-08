"""Deterministic S2 task sampling, GCOPTER planning, and expert dataset IO."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
from dataclasses import dataclass, replace

import numpy as np

from quadrotor_diffusion_ppo.envs.action import velocity_reference_to_action
from quadrotor_diffusion_ppo.envs.scene import SCENE_IDS, SceneSpec, clean_reproduction_root, load_all_scenes
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary
from quadrotor_diffusion_ppo.expert.trajectory import GcopterTrajectory


TASK_GENERATION_BASE_SEED = 20260808
SPLIT_SEED = 20260809
TARGET_ACCEPTED = 40
MAX_CANDIDATE_ATTEMPTS = 100
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TaskSpec:
    scene_id: str
    candidate_id: int
    task_id: str
    candidate_seed: int
    start: np.ndarray
    goal: np.ndarray


def _sample_from_corridor(corridor: dict[str, np.ndarray | str], rng: np.random.Generator,
                          margin: float = 0.25) -> np.ndarray:
    low = np.asarray(corridor["min"], dtype=float)
    high = np.asarray(corridor["max"], dtype=float)
    span = high - low
    return rng.uniform(low + margin * span, high - margin * span).astype(np.float64)


def _point_to_box_distance(point: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
    delta = np.maximum(np.abs(point - center) - size / 2.0, 0.0)
    return float(np.linalg.norm(delta))


def _endpoint_is_legal(scene: SceneSpec, point: np.ndarray) -> bool:
    for axis, name in enumerate(("x", "y", "z")):
        bounds = scene.world_bounds[name]
        if not (bounds[0] <= point[axis] <= bounds[1]):
            return False
    return all(_point_to_box_distance(point, np.asarray(o["center"]), np.asarray(o["size"]))
               >= scene.safety_radius for o in scene.obstacles)


def candidate_task(scene: SceneSpec, scene_index: int, candidate_id: int) -> TaskSpec:
    """Generate one deterministic endpoint task from first/final corridor interiors."""
    seed = TASK_GENERATION_BASE_SEED + scene_index * 100_000 + candidate_id
    rng = np.random.Generator(np.random.PCG64(seed))
    start = _sample_from_corridor(scene.corridors[0], rng)
    goal = _sample_from_corridor(scene.corridors[-1], rng)
    return TaskSpec(scene.scene_id, candidate_id, f"{scene.scene_id}_TASK_{candidate_id:03d}",
                    seed, start, goal)


def candidate_input_rows(scene: SceneSpec, scene_index: int, count: int = MAX_CANDIDATE_ATTEMPTS) -> list[dict]:
    rows = []
    for candidate_id in range(count):
        task = candidate_task(scene, scene_index, candidate_id)
        rows.append({"scene_id": task.scene_id, "candidate_id": task.candidate_id, "task_id": task.task_id,
                     "start_x": task.start[0], "start_y": task.start[1], "start_z": task.start[2],
                     "goal_x": task.goal[0], "goal_y": task.goal[1], "goal_z": task.goal[2],
                     "candidate_seed": task.candidate_seed})
    return rows


def candidate_input_hash(rows: list[dict]) -> str:
    fields = ("scene_id", "candidate_id", "task_id", "start_x", "start_y", "start_z",
              "goal_x", "goal_y", "goal_z", "candidate_seed")
    canonical = "\n".join("\t".join(str(row[field]) for field in fields) for row in rows) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _task_scene(scene: SceneSpec, task: TaskSpec) -> SceneSpec:
    return replace(scene, start=task.start.copy(), goal=task.goal.copy())


def _wsl_path(path: Path) -> str:
    path = path.resolve()
    drive = path.drive.rstrip(":").lower()
    return "/mnt/" + drive + path.as_posix()[2:]


def _replace_yaml_position(text: str, section: str, position: np.ndarray) -> str:
    pattern = rf"(?ms)(^\s*{re.escape(section)}:\s*\n\s*position:\s*)\[[^\]]+\]"
    replacement = rf"\1[{', '.join(f'{value:.10f}' for value in position)}]"
    result, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"could not replace {section}.position in frozen scene YAML")
    return result


def _write_task_yaml(scene: SceneSpec, task: TaskSpec, path: Path) -> None:
    source = clean_reproduction_root() / "scenes" / "pybullet" / f"{scene.scene_id}.yaml"
    text = source.read_text(encoding="utf-8")
    text = _replace_yaml_position(text, "start", task.start)
    text = _replace_yaml_position(text, "goal", task.goal)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run_gcopter_planner(scene: SceneSpec, task: TaskSpec, work_root: Path) -> tuple[Path, dict]:
    """Run the approved local planner and return its coefficient directory/summary."""
    task_root = work_root / scene.scene_id / task.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    yaml_path = task_root / "task.yaml"
    output_dir = task_root / "planner_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_task_yaml(scene, task, yaml_path)
    planner = Path(__file__).resolve().parents[3] / ".deps" / "gcopter_reference" / "gcopter_yaml_scene_planner"
    repo_root = clean_reproduction_root()
    if not planner.exists():
        raise FileNotFoundError(f"GCOPTER planner missing: {planner}")
    command = " ".join(shlex.quote(value) for value in (
        _wsl_path(planner), _wsl_path(yaml_path), _wsl_path(output_dir), _wsl_path(repo_root)))
    completed = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command],
                               capture_output=True, timeout=120, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"GCOPTER planner failed rc={completed.returncode}: {stderr}")
    summary_path = output_dir / "planner_summary.json"
    coefficients = output_dir / "reference_coefficients.csv"
    if not summary_path.exists() or not coefficients.exists():
        raise RuntimeError("GCOPTER planner did not produce reference coefficients and summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not (summary.get("setup_success") and summary.get("optimize_success") and summary.get("reference_pass")):
        raise RuntimeError("GCOPTER planner summary did not pass setup/optimize/reference gates")
    return output_dir, summary


def _failure_reason(row: dict) -> str:
    if not row["sampling_valid"]:
        return "sampling_invalid"
    if not row["planner_success"]:
        return "planner_failed"
    if not row["rollout_success"]:
        return "rollout_failed"
    if row["collision_count"]:
        return "collision"
    if row["ground_contact_count"]:
        return "ground_contact"
    if not row["goal_reached"]:
        return "goal_failed"
    if row["nonfinite_count"]:
        return "nonfinite"
    if row["clip_count"]:
        return "action_clip"
    return "other"


def rollout_task(scene: SceneSpec, task: TaskSpec, trajectory: GcopterTrajectory,
                 contract: dict) -> tuple[dict, dict | None]:
    speed_limit = float(contract["speed_limit_m_per_s"])
    ray_range = float(contract["ray_max_range_m"])
    goal_tolerance = float(contract["goal_tolerance_m"])
    settle_s = float(contract["post_reference_settle_s"])
    env = ObstacleVelocityAviary(scene, speed_limit, ray_range, gui=False)
    observations = []
    actions = []
    next_observations = []
    positions = []
    velocities = []
    goals = []
    times = []
    progress = []
    collisions = []
    ground_contacts = []
    start_time = 0.0
    clip_count = 0
    nonfinite_count = 0
    collision_total = 0
    ground_total = 0
    min_goal_distance = float("inf")
    max_raw_action_component = 0.0
    max_commanded_speed = 0.0
    max_speed_limit_excess = 0.0
    rollout_success = False
    error = ""
    try:
        observation, _ = env.reset(seed=task.candidate_seed)
        total_duration = trajectory.total_duration
        total_steps = int(np.ceil((total_duration + settle_s) * scene.control_hz))
        for step in range(total_steps):
            elapsed = step / scene.control_hz
            state = np.asarray(env._getDroneStateVector(0), dtype=float)
            _, v_ref, _, _, _ = trajectory.evaluate(min(elapsed, total_duration))
            mapped = velocity_reference_to_action(v_ref, speed_limit)
            clip_count += int(mapped.clipped)
            max_raw_action_component = max(max_raw_action_component, float(np.max(np.abs(mapped.raw_normalized))))
            max_commanded_speed = max(max_commanded_speed, mapped.speed_ratio * speed_limit)
            max_speed_limit_excess = max(max_speed_limit_excess, mapped.speed_limit_excess)
            if not np.isfinite(np.r_[state[:16], observation, v_ref, mapped.raw_normalized]).all():
                nonfinite_count += 1
                raise FloatingPointError("nonfinite state, observation, reference, or action")
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            actions.append(np.asarray(mapped.raw_normalized, dtype=np.float32).copy())
            positions.append(state[0:3].astype(np.float32))
            velocities.append(state[10:13].astype(np.float32))
            goals.append(scene.goal.astype(np.float32).copy())
            times.append(np.float32(elapsed))
            progress.append(np.float32(min(1.0, elapsed / total_duration)))
            observation, _, _, _, _ = env.step(mapped.raw_normalized.astype(np.float32))
            next_observations.append(np.asarray(observation, dtype=np.float32).copy())
            obstacle_contacts, ground_contacts_now = env.contact_counts()
            collision_total += obstacle_contacts
            ground_total += ground_contacts_now
            collisions.append(bool(obstacle_contacts > 0))
            ground_contacts.append(bool(ground_contacts_now > 0))
            if not np.isfinite(observation).all():
                nonfinite_count += 1
                raise FloatingPointError("nonfinite next observation")
            state_after = np.asarray(env._getDroneStateVector(0), dtype=float)
            if not np.isfinite(state_after[:16]).all():
                nonfinite_count += 1
                raise FloatingPointError("nonfinite next state")
            min_goal_distance = min(min_goal_distance, float(np.linalg.norm(state_after[0:3] - scene.goal)))
        rollout_success = True
    except Exception as exc:
        error = str(exc)
    finally:
        env.close()
    reference_completed = len(actions) == int(np.ceil((trajectory.total_duration + settle_s) * scene.control_hz))
    goal_reached = bool(min_goal_distance <= goal_tolerance)
    accepted = bool(rollout_success and reference_completed and goal_reached and collision_total == 0 and
                    ground_total == 0 and nonfinite_count == 0 and clip_count == 0 and
                    max_speed_limit_excess <= speed_limit * 1.0e-6)
    row = {
        "sampling_valid": _endpoint_is_legal(scene, task.start) and _endpoint_is_legal(scene, task.goal),
        "planner_success": True,
        "rollout_success": rollout_success,
        "reference_completed": reference_completed,
        "goal_reached": goal_reached,
        "collision_count": collision_total,
        "ground_contact_count": ground_total,
        "nonfinite_count": nonfinite_count,
        "clip_count": clip_count,
        "max_raw_action_component": max_raw_action_component,
        "max_commanded_speed": max_commanded_speed,
        "max_speed_limit_excess": max_speed_limit_excess,
        "minimum_goal_distance": min_goal_distance if np.isfinite(min_goal_distance) else None,
        "episode_length": len(actions),
        "accepted": accepted,
        "error": error,
    }
    if not accepted:
        return row, None
    data = {"observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "next_observations": np.asarray(next_observations, dtype=np.float32),
            "positions": np.asarray(positions, dtype=np.float32),
            "velocities": np.asarray(velocities, dtype=np.float32),
            "goals": np.asarray(goals, dtype=np.float32),
            "times": np.asarray(times, dtype=np.float32),
            "trajectory_progress": np.asarray(progress, dtype=np.float32),
            "collision": np.asarray(collisions, dtype=np.bool_),
            "success": np.zeros(len(actions), dtype=np.bool_),
            "done": np.zeros(len(actions), dtype=np.bool_)}
    data["success"][-1] = True
    data["done"][-1] = True
    return row, data


def _empty_array(name: str, length: int) -> np.ndarray:
    shapes = {"observations": (length, 34), "actions": (length, 3), "next_observations": (length, 34),
              "positions": (length, 3), "velocities": (length, 3), "goals": (length, 3),
              "times": (length,), "trajectory_progress": (length,), "collision": (length,),
              "success": (length,), "done": (length,), "episode_ids": (length,),
              "scene_indices": (length,), "task_indices": (length,)}
    dtype = np.float32 if name not in {"collision", "success", "done", "episode_ids", "scene_indices", "task_indices"} else (np.bool_ if name in {"collision", "success", "done"} else np.int32)
    return np.zeros(shapes[name], dtype=dtype)


def write_split_npz(episodes: list[tuple[dict, dict]], path: Path) -> dict:
    total = sum(len(data["actions"]) for _, data in episodes)
    arrays = {name: _empty_array(name, total) for name in (
        "observations", "actions", "next_observations", "positions", "velocities", "goals", "times",
        "trajectory_progress", "collision", "success", "done", "episode_ids", "scene_indices", "task_indices")}
    offsets, lengths = [], []
    cursor = 0
    for local_episode, (record, data) in enumerate(episodes):
        length = len(data["actions"])
        sl = slice(cursor, cursor + length)
        for name in ("observations", "actions", "next_observations", "positions", "velocities", "goals", "times",
                     "trajectory_progress", "collision", "success", "done"):
            arrays[name][sl] = data[name]
        arrays["episode_ids"][sl] = int(record["episode_id"])
        arrays["scene_indices"][sl] = int(record["scene_index"])
        arrays["task_indices"][sl] = int(record["candidate_id"])
        offsets.append(cursor)
        lengths.append(length)
        cursor += length
    arrays["episode_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays["episode_lengths"] = np.asarray(lengths, dtype=np.int32)
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise FloatingPointError(f"nonfinite value in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"transitions": total, "episodes": len(episodes), "bytes": path.stat().st_size}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_rows = []
    for row in rows:
        cleaned = {}
        for key, value in row.items():
            if isinstance(value, str):
                value = value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()
            cleaned[key] = value
        cleaned_rows.append(cleaned)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned_rows)


def split_accepted_records(records: list[dict], split_seed: int = SPLIT_SEED) -> dict[str, list[dict]]:
    by_split = {split: [] for split in SPLITS}
    for scene_index, scene_id in enumerate(SCENE_IDS):
        scene_records = [record for record in records if record["scene_id"] == scene_id and record["accepted"]]
        if len(scene_records) != TARGET_ACCEPTED:
            raise ValueError(f"{scene_id} has {len(scene_records)} accepted tasks, expected {TARGET_ACCEPTED}")
        rng = np.random.Generator(np.random.PCG64(split_seed + scene_index))
        order = rng.permutation(len(scene_records))
        for index, record_index in enumerate(order):
            record = scene_records[int(record_index)]
            split = "train" if index < 28 else "val" if index < 34 else "test"
            record["split"] = split
            by_split[split].append(record)
    return by_split


def validate_npz(path: Path) -> dict:
    required = {"observations": (34,), "actions": (3,), "next_observations": (34,), "positions": (3,),
                "velocities": (3,), "goals": (3,), "times": (), "trajectory_progress": (), "collision": (),
                "success": (), "done": (), "episode_ids": (), "scene_indices": (), "task_indices": (),
                "episode_offsets": (), "episode_lengths": ()}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(required) - set(data.files))
        if missing:
            raise AssertionError(f"{path} missing arrays: {missing}")
        transitions = len(data["actions"])
        if any(data[name].shape[1:] != shape for name, shape in required.items() if name in data and len(shape) > 0):
            raise AssertionError(f"{path} has an invalid array shape")
        if not all(np.isfinite(data[name]).all() for name in data.files if data[name].dtype.kind == "f"):
            raise AssertionError(f"{path} contains nonfinite floating point data")
        offsets = data["episode_offsets"]
        lengths = data["episode_lengths"]
        if len(offsets) != len(lengths) or (len(offsets) and offsets[-1] + lengths[-1] != transitions):
            raise AssertionError(f"{path} has invalid episode boundaries")
        if np.any(data["success"] & ~data["done"]):
            raise AssertionError(f"{path} has success without done")
        return {"transitions": transitions, "episodes": len(offsets), "bytes": path.stat().st_size}


def cleanup_work_root(work_root: Path) -> None:
    if work_root.exists():
        shutil.rmtree(work_root)
