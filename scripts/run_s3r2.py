"""S3-R2 TRAIN-only recovery replan augmentation and VAL-first evaluation.

The only algorithmic change in this milestone is the TRAIN distribution:
frozen S3-R1 student rollouts provide at most one first-divergence anchor per
TRAIN task, and the frozen local GCOPTER teacher replans from that actual
position.  VAL and TEST are never used for recovery generation or training.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time

# Required by deterministic CUDA GEMM on CUDA >= 10.2.  Set before importing
# torch so the fixed R2 seed is a real deterministic configuration.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# R1 was serialized in the existing NumPy-2 environment.  The CUDA runtime
# reuses NumPy 1.24 for compatibility with its frozen SciPy stack; these
# read-only aliases let torch deserialize the checkpoint without rewriting it
# or changing its audited SHA256.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from quadrotor_diffusion_ppo.envs.action import velocity_reference_to_action
from quadrotor_diffusion_ppo.envs.scene import SCENE_IDS, clean_reproduction_root, load_all_scenes
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary
from quadrotor_diffusion_ppo.expert.trajectory import GcopterTrajectory
from scripts.run_s3_diffusion import (
    ACTION_DIM, BATCH_SIZE, DDIM_STEPS, DIFFUSION_STEPS, EVAL_BASE_SEED,
    GOAL_TOLERANCE, HORIZON, MAX_EPISODE_STEPS, OBSERVATION_DIM, PROJECT_ROOT,
    S2_DATASET, S2_ROOT, build_windows, load_npz, observation_normalization,
    read_task_rows, row_task, sha256_file, stable_seed, summarize_rollouts,
)


BRANCH = "agent/s3r2-recovery-replan-v1"
START_HEAD = "c3b7c3376a3cf54e0047d7edd5fcb729c6f4050e"
R1_POLICY_HEAD = "e2f4e19f02e1b07de48617b607ceeb32340a5095"
R1_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "s3r1" / "best.pt"
EXPECTED_CHECKPOINT_SHA256 = "8afa677d7c8a34179c60f067209b6924170fc7854445048d7a7e9b55bbbc5fc8"
EXPECTED_DATASET_SHA256 = {
    "train": "701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21",
    "val": "c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc",
    "test": "88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66",
}
TRAIN_SEED = 20260810
SPEED_LIMIT = 0.801
RAY_RANGE = 3.0
POSITION_DIVERGENCE_THRESHOLD = 0.30
SETTLE_S = 2.0
RECOVERY_ROOT = PROJECT_ROOT / "artifacts" / "s3r2" / "recovery_dataset"
S3R2_ROOT = PROJECT_ROOT / "artifacts" / "s3r2"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "s3r2"
RECOVERY_DATASET = RECOVERY_ROOT / "recovery_train.npz"
TRAIN_ROLLOUT_CACHE = Path(tempfile.gettempdir()) / "s3r2-train-rollouts.pkl"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key, value in row.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = json.dumps(np.asarray(value).tolist(), separators=(",", ":"))
                if isinstance(value, str):
                    value = value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()
                clean[key] = value
            writer.writerow(clean)


def assert_identity() -> dict:
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip() != START_HEAD:
        raise RuntimeError("BLOCKED_S3R2_START_HEAD")
    # The launch-time start was independently verified clean before adding the
    # diagnostic implementation and contract files on this branch.
    if not R1_CHECKPOINT.exists() or sha256_file(R1_CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("BLOCKED_S3R2_CHECKPOINT_IDENTITY")
    hashes = {name: sha256_file(S2_DATASET / f"{name}.npz") for name in ("train", "val", "test")}
    if hashes != EXPECTED_DATASET_SHA256:
        raise RuntimeError("BLOCKED_S3R2_DATASET_IDENTITY")
    manifest = read_csv(S2_ROOT / "task_manifest.csv")
    accepted = [row for row in manifest if row["accepted"].lower() == "true"]
    counts = {split: sum(row["split"] == split for row in accepted) for split in ("train", "val", "test")}
    if counts != {"train": 252, "val": 54, "test": 54}:
        raise RuntimeError(f"BLOCKED_S3R2_TASK_SPLIT_COUNTS:{counts}")
    return {"sha256": hashes, "task_counts": counts, "observation_dim": 34, "action_dim": 3,
            "episodes": {"train": 252, "val": 54, "test": 54}, "test_values_opened": False}


def load_r1_model(device: torch.device):
    payload = torch.load(R1_CHECKPOINT, map_location=device, weights_only=False)
    required = {"model_frozen": True, "prediction_target": "x0", "action_support": "unit_ball_squash",
                "ddim_steps": 10, "ddim_eta": 0.0, "diffusion_steps": 100}
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("BLOCKED_S3R2_R1_CHECKPOINT_METADATA")
    model = ConditionalDiffusionMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    schedule = DiffusionSchedule(payload["diffusion_steps"]).to(device)
    return model, schedule, payload


def predict_action(model, schedule, observation: np.ndarray, mean: np.ndarray, std: np.ndarray,
                   seed: int, device: torch.device) -> tuple[np.ndarray, dict]:
    normalized = (np.asarray(observation, dtype=np.float32) - mean) / np.maximum(std, 1e-6)
    condition = torch.from_numpy(normalized).reshape(1, -1).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sampled, diagnostics = schedule.ddim_sample_x0(model, condition, steps=DDIM_STEPS,
                                                    generator=generator, return_diagnostics=True)
    return sampled[0, 0].cpu().numpy().astype(np.float32), diagnostics


def expert_episode_index(data: dict[str, np.ndarray]) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    result = {}
    for offset, length in zip(data["episode_offsets"], data["episode_lengths"]):
        offset, length = int(offset), int(length)
        scene_id = SCENE_IDS[int(data["scene_indices"][offset])]
        candidate_id = int(data["task_indices"][offset])
        result[(scene_id, candidate_id)] = {key: data[key][offset:offset + length].copy() for key in (
            "observations", "actions", "positions", "velocities", "goals", "times")}
    return result


def interpolate_expert_position(expert: dict[str, np.ndarray], time_s: float) -> np.ndarray:
    return np.asarray([np.interp(time_s, expert["times"].astype(float), expert["positions"][:, axis])
                       for axis in range(3)], dtype=float)


def train_student_rollout(model, schedule, scene, task: dict, mean: np.ndarray, std: np.ndarray,
                          device: torch.device) -> dict:
    """Deployment-visible R1 rollout; no expert or planner data is read."""
    env = ObstacleVelocityAviary(scene, SPEED_LIMIT, RAY_RANGE, gui=False)
    seed = stable_seed(task["task_id"])
    trace = []
    success = collision = ground = nonfinite = False
    error = ""
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(MAX_EPISODE_STEPS):
            state_before = np.asarray(env._getDroneStateVector(0), dtype=float).copy()
            if not np.isfinite(observation).all() or not np.isfinite(state_before[:16]).all():
                nonfinite = True
                break
            action, diag = predict_action(model, schedule, observation, mean, std, seed + step, device)
            if not np.isfinite(action).all():
                nonfinite = True
                break
            observation, _, _, _, _ = env.step(action)
            obstacle_now, ground_now = env.contact_counts()
            collision = collision or obstacle_now > 0
            ground = ground or ground_now > 0
            state = np.asarray(env._getDroneStateVector(0), dtype=float).copy()
            if not np.isfinite(state[:16]).all() or not np.isfinite(observation).all():
                nonfinite = True
                break
            distance = float(np.linalg.norm(state[:3] - scene.goal))
            trace.append({
                "task_id": task["task_id"], "scene_id": scene.scene_id, "family": scene.family,
                "step": int(step), "time": float((step + 1) / scene.control_hz),
                "position": state[0:3].astype(np.float32), "velocity": state[10:13].astype(np.float32),
                "quaternion": state[3:7].astype(np.float32), "angular_velocity": state[13:16].astype(np.float32),
                "goal": scene.goal.astype(np.float32), "observation": observation.astype(np.float32).copy(),
                "action": action.astype(np.float32).copy(), "action_norm": float(np.linalg.norm(action)),
                "support_violation_count": int(diag["support_violation_count"]),
                "success": bool(distance <= GOAL_TOLERANCE),
            })
            if distance <= GOAL_TOLERANCE:
                success = True
                break
            if collision or ground:
                break
    except Exception as exc:
        error = str(exc)
        nonfinite = True
    finally:
        env.close()
    action_array = np.asarray([item["action"] for item in trace], dtype=np.float32)
    outcome = "SUCCESS" if success else "COLLISION" if collision else "GROUND_CONTACT" if ground else "NONFINITE" if nonfinite else "TIMEOUT"
    return {"task_id": task["task_id"], "scene_id": scene.scene_id, "family": scene.family,
            "outcome": outcome, "success": bool(success), "collision": int(collision),
            "ground_contact": int(ground), "nonfinite": int(nonfinite),
            "timeout": bool(not success and not collision and not ground and not nonfinite and len(trace) >= MAX_EPISODE_STEPS),
            "steps": len(trace), "time": float(len(trace) / scene.control_hz),
            "action_trace_hash": hashlib.sha256(action_array.tobytes()).hexdigest(),
            "trace": trace, "error": error}


def select_first_anchor(rollout: dict, expert: dict[str, np.ndarray]) -> dict | None:
    """Post-hoc, deterministic first state beyond the frozen 0.30 m threshold."""
    for item in rollout["trace"]:
        reference = interpolate_expert_position(expert, float(item["time"]))
        error = float(np.linalg.norm(np.asarray(item["position"], dtype=float) - reference))
        if error > POSITION_DIVERGENCE_THRESHOLD:
            anchor = dict(item)
            anchor["expert_position"] = reference.astype(np.float32)
            anchor["position_error_to_expert"] = error
            return anchor
    return None


def _replace_vector(text: str, section: str, key: str, values: np.ndarray) -> str:
    pattern = rf"(?ms)(^\s*{re.escape(section)}:\s*\n.*?^\s*{re.escape(key)}:\s*)\[[^\]]+\]"
    replacement = rf"\1[{', '.join(f'{float(v):.10f}' for v in values)}]"
    result, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"missing {section}.{key} in scene YAML")
    return result


def write_recovery_yaml(scene_id: str, anchor: dict, goal: np.ndarray, path: Path) -> None:
    source = clean_reproduction_root() / "scenes" / "pybullet" / f"{scene_id}.yaml"
    text = source.read_text(encoding="utf-8")
    text = _replace_vector(text, "start", "position", np.asarray(anchor["position"], dtype=float))
    # The current frozen CLI records these real derivatives in the input audit,
    # but its C++ interface consumes only start.position/goal.position.
    text = _replace_vector(text, "start", "velocity", np.asarray(anchor["velocity"], dtype=float))
    text = _replace_vector(text, "goal", "position", np.asarray(goal, dtype=float))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def wsl_path(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    return f"/mnt/{drive}{path.as_posix().split(':', 1)[1]}" if path.drive else path.as_posix()


def planner_attempt(scene, task: dict, anchor: dict) -> dict:
    task_root = RECOVERY_ROOT / "planner" / task["task_id"]
    output_dir = task_root / "planner_output"
    coeff = output_dir / "reference_coefficients.csv"
    summary_path = output_dir / "planner_summary.json"
    yaml_path = task_root / "task.yaml"
    write_recovery_yaml(scene.scene_id, anchor, np.asarray(task["goal"], dtype=float), yaml_path)
    planner = PROJECT_ROOT / ".deps" / "gcopter_reference" / "gcopter_yaml_scene_planner"
    if not planner.exists():
        raise FileNotFoundError(f"GCOPTER planner missing: {planner}")
    if not (coeff.exists() and summary_path.exists()):
        output_dir.mkdir(parents=True, exist_ok=True)
        command = " ".join(shlex.quote(value) for value in (
            wsl_path(planner), wsl_path(yaml_path), wsl_path(output_dir),
            wsl_path(clean_reproduction_root())))
        completed = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command],
                                   capture_output=True, timeout=180, check=False)
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")[-1200:]
            return {"planner_success": False, "error": f"rc={completed.returncode}:{stderr}"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ok = bool(summary.get("setup_success") and summary.get("optimize_success") and summary.get("reference_pass"))
    except Exception as exc:
        return {"planner_success": False, "error": str(exc)}
    return {"planner_success": ok, "planner_summary": summary, "coefficients": str(coeff),
            "planner_interface": "position_only_static_zero_derivatives",
            "anchor_velocity_recorded": True, "error": "" if ok else "planner_reference_gate_failed"}


def _empty_recovery_array(name: str, length: int) -> np.ndarray:
    shapes = {"observations": (length, 34), "actions": (length, 3), "next_observations": (length, 34),
              "positions": (length, 3), "velocities": (length, 3), "goals": (length, 3), "times": (length,),
              "trajectory_progress": (length,), "collision": (length,), "success": (length,), "done": (length,),
              "episode_ids": (length,), "scene_indices": (length,), "task_indices": (length,)}
    shape = shapes[name]
    dtype = np.float32 if name not in {"collision", "success", "done", "episode_ids", "scene_indices", "task_indices"} else (
        np.bool_ if name in {"collision", "success", "done"} else np.int32)
    return np.zeros(shape, dtype=dtype)


def teacher_verify(scene, task: dict, anchor: dict, planner: dict, mean: np.ndarray) -> tuple[dict, dict | None]:
    """Re-run the exact student prefix, then switch at the actual anchor state."""
    trajectory = GcopterTrajectory.from_csv(planner["coefficients"])
    env = ObstacleVelocityAviary(scene, SPEED_LIMIT, RAY_RANGE, gui=False)
    anchor_step = int(anchor["step"])
    seed = stable_seed(task["task_id"])
    observations, actions, next_observations = [], [], []
    positions, velocities, goals, times, progress = [], [], [], [], []
    collisions, success = [], []
    clip_count = collision_count = ground_count = nonfinite_count = 0
    min_goal_distance = float("inf")
    error = ""
    switched_state_error = None
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(MAX_EPISODE_STEPS):
            state = np.asarray(env._getDroneStateVector(0), dtype=float).copy()
            if step <= anchor_step:
                action, _ = predict_action(_R2_CONTEXT["model"], _R2_CONTEXT["schedule"], observation,
                                            _R2_CONTEXT["mean"], _R2_CONTEXT["std"], seed + step, _R2_CONTEXT["device"])
            else:
                elapsed = (step - anchor_step - 1) / scene.control_hz
                _, v_ref, _, _, _ = trajectory.evaluate(min(elapsed, trajectory.total_duration))
                mapped = velocity_reference_to_action(v_ref, SPEED_LIMIT)
                action = mapped.raw_normalized.astype(np.float32)
            if not np.isfinite(action).all() or not np.isfinite(observation).all() or not np.isfinite(state[:16]).all():
                nonfinite_count += 1
                break
            if step == anchor_step + 1:
                switched_state_error = float(np.linalg.norm(state[:3] - np.asarray(anchor["position"], dtype=float)))
            observations.append(observation.astype(np.float32).copy())
            actions.append(action.astype(np.float32).copy())
            positions.append(state[:3].astype(np.float32))
            velocities.append(state[10:13].astype(np.float32))
            goals.append(scene.goal.astype(np.float32).copy())
            times.append(np.float32((step - anchor_step - 1) / scene.control_hz))
            progress.append(np.float32(min(1.0, max(0.0, (step - anchor_step - 1) / max(trajectory.total_duration, 1e-6)))) if step > anchor_step else np.float32(0.0))
            observation, _, _, _, _ = env.step(action)
            if env.last_velocity_action is not None and env.last_velocity_action.clipped:
                clip_count += 1
            obstacle_now, ground_now = env.contact_counts()
            collision_count += obstacle_now
            ground_count += ground_now
            collisions.append(bool(obstacle_now > 0))
            state_after = np.asarray(env._getDroneStateVector(0), dtype=float)
            if not np.isfinite(state_after[:16]).all() or not np.isfinite(observation).all():
                nonfinite_count += 1
                break
            min_goal_distance = min(min_goal_distance, float(np.linalg.norm(state_after[:3] - scene.goal)))
            success.append(bool(min_goal_distance <= GOAL_TOLERANCE))
            next_observations.append(observation.astype(np.float32).copy())
            if collision_count or ground_count:
                break
            if step > anchor_step and (step - anchor_step) >= int(np.ceil((trajectory.total_duration + SETTLE_S) * scene.control_hz)):
                break
    except Exception as exc:
        error = str(exc)
        nonfinite_count += 1
    finally:
        env.close()
    recovery_steps = max(0, len(actions) - (anchor_step + 1))
    reference_completed = recovery_steps >= int(np.ceil((trajectory.total_duration + SETTLE_S) * scene.control_hz))
    goal_reached = bool(min_goal_distance <= GOAL_TOLERANCE)
    accepted = bool(reference_completed and goal_reached and collision_count == 0 and ground_count == 0 and
                    nonfinite_count == 0 and clip_count == 0 and error == "")
    record = {"accepted": accepted, "reference_completed": reference_completed, "goal_reached": goal_reached,
              "collision_count": collision_count, "ground_contact_count": ground_count,
              "nonfinite_count": nonfinite_count, "clip_count": clip_count, "episode_length": len(actions),
              "recovery_steps": recovery_steps, "minimum_goal_distance": min_goal_distance if np.isfinite(min_goal_distance) else None,
              "switched_state_position_error": switched_state_error, "error": error,
              "planner_interface": planner["planner_interface"]}
    if not accepted:
        return record, None
    # Keep only the teacher suffix, starting with the action executed after the
    # exact anchor.  Prefix actions are student-generated and never enter R2.
    start = anchor_step + 1
    data = {name: np.asarray(values[start:], dtype=np.float32) for name, values in (
        ("observations", observations), ("actions", actions), ("next_observations", next_observations),
        ("positions", positions), ("velocities", velocities), ("goals", goals), ("times", times),
        ("trajectory_progress", progress))}
    data["collision"] = np.asarray(collisions[start:], dtype=np.bool_)
    data["success"] = np.zeros(len(data["actions"]), dtype=np.bool_)
    data["done"] = np.zeros(len(data["actions"]), dtype=np.bool_)
    data["success"][-1] = True
    data["done"][-1] = True
    if len(data["actions"]) < HORIZON:
        raise RuntimeError("accepted recovery trajectory shorter than H=16")
    return record, data


def write_recovery_npz(accepted: list[tuple[dict, dict]]) -> dict:
    total = sum(len(data["actions"]) for _, data in accepted)
    arrays = {name: _empty_recovery_array(name, total) for name in (
        "observations", "actions", "next_observations", "positions", "velocities", "goals", "times",
        "trajectory_progress", "collision", "success", "done", "episode_ids", "scene_indices", "task_indices")}
    offsets, lengths, cursor = [], [], 0
    for episode_id, (record, data) in enumerate(accepted):
        length = len(data["actions"])
        sl = slice(cursor, cursor + length)
        for name in ("observations", "actions", "next_observations", "positions", "velocities", "goals", "times",
                     "trajectory_progress", "collision", "success", "done"):
            arrays[name][sl] = data[name]
        scene_index = SCENE_IDS.index(record["scene_id"])
        task_index = int(record["task_id"].rsplit("_", 1)[1])
        arrays["episode_ids"][sl] = episode_id
        arrays["scene_indices"][sl] = scene_index
        arrays["task_indices"][sl] = task_index
        offsets.append(cursor); lengths.append(length); cursor += length
    arrays["episode_offsets"] = np.asarray(offsets, dtype=np.int64)
    arrays["episode_lengths"] = np.asarray(lengths, dtype=np.int32)
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise RuntimeError("BLOCKED_S3R2_RECOVERY_NONFINITE_DATA")
    RECOVERY_DATASET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECOVERY_DATASET, **arrays)
    return {"path": str(RECOVERY_DATASET), "sha256": sha256_file(RECOVERY_DATASET),
            "episodes": len(accepted), "transitions": total, "bytes": RECOVERY_DATASET.stat().st_size}


def train_model(train_obs, train_actions, val_obs, val_actions, mean, std, device, dataset_identity):
    model = ConditionalDiffusionMLP(**_R2_CONTEXT["model_config"]).to(device)
    schedule = DiffusionSchedule(DIFFUSION_STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(TRAIN_SEED)
    best_val, best_update, curve = float("inf"), 0, []
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    best_path, last_path = CHECKPOINT_ROOT / "best.pt", CHECKPOINT_ROOT / "last.pt"
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for update in range(1, 10_001):
        model.train()
        indices = torch.randint(0, len(train_obs), (BATCH_SIZE,), generator=generator, device=device)
        timesteps = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), generator=generator, device=device)
        noise = torch.randn(train_actions[indices].shape, generator=generator, device=device)
        noisy = schedule.add_noise(train_actions[indices], noise, timesteps)
        predicted = model(noisy, train_obs[indices], timesteps)
        loss = torch.nn.functional.mse_loss(predicted, train_actions[indices])
        if not torch.isfinite(loss):
            raise RuntimeError("BLOCKED_S3R2_RECOVERY_NUMERICS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError("BLOCKED_S3R2_RECOVERY_NUMERICS")
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("BLOCKED_S3R2_RECOVERY_NUMERICS")
        row = {"update": update, "train_loss": float(loss.item()), "val_loss": "", "gradient_norm": gradient_norm}
        if update % 500 == 0 or update == 1:
            model.eval(); losses = []
            eval_gen = torch.Generator(device=device).manual_seed(TRAIN_SEED + 991)
            with torch.no_grad():
                for start_i in range(0, len(val_obs), BATCH_SIZE):
                    batch = val_obs[start_i:start_i + BATCH_SIZE]
                    target = val_actions[start_i:start_i + BATCH_SIZE]
                    ts = torch.randint(0, DIFFUSION_STEPS, (len(batch),), generator=eval_gen, device=device)
                    noise_v = torch.randn(target.shape, generator=eval_gen, device=device)
                    value = torch.nn.functional.mse_loss(model(schedule.add_noise(target, noise_v, ts), batch, ts), target)
                    losses.append(float(value.item()) * len(batch))
            val_loss = sum(losses) / len(val_obs)
            row["val_loss"] = val_loss
            if val_loss < best_val:
                best_val, best_update = val_loss, update
                torch.save(checkpoint_payload(model, mean, std, dataset_identity, best_update, best_val, False), best_path)
        curve.append(row)
        if update % 1000 == 0:
            print(f"update={update}/10000 train_loss={loss.item():.6f} best_val={best_val:.6f}", flush=True)
    torch.save(checkpoint_payload(model, mean, std, dataset_identity, 10000, best_val, False), last_path)
    wall = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated(device))
    return best_path, last_path, curve, best_update, best_val, wall, peak


def checkpoint_payload(model, mean, std, identity, best_update, best_val, frozen):
    return {"model_state": model.state_dict(), "model_config": _R2_CONTEXT["model_config"],
            "observation_mean": mean, "observation_std": std,
            "dataset_sha256": identity["original"]["sha256"], "recovery_dataset_sha256": identity["recovery"].get("sha256"),
            "diffusion_steps": DIFFUSION_STEPS, "beta_schedule": "cosine", "prediction_target": "x0",
            "action_support": "unit_ball_squash", "ddim_steps": DDIM_STEPS, "ddim_eta": 0.0,
            "train_seed": TRAIN_SEED, "best_update": int(best_update), "best_val_loss": float(best_val),
            "model_frozen": bool(frozen)}


@torch.no_grad()
def offline_loss(model, schedule, obs_np, actions_np, mean, std, device) -> float:
    normalized = (obs_np - mean) / np.maximum(std, 1e-6)
    obs = torch.from_numpy(normalized.astype(np.float32)).to(device)
    actions = torch.from_numpy(actions_np.astype(np.float32)).to(device)
    generator = torch.Generator(device=device).manual_seed(TRAIN_SEED + 991)
    losses = []
    for start in range(0, len(obs), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        ts = torch.randint(0, DIFFUSION_STEPS, (len(obs[batch]),), generator=generator, device=device)
        noise = torch.randn(actions[batch].shape, generator=generator, device=device)
        losses.append(float(torch.nn.functional.mse_loss(model(schedule.add_noise(actions[batch], noise, ts), obs[batch], ts)).item()) * len(obs[batch]))
    return sum(losses) / len(obs)


def evaluate_task(model, schedule, scene, task, mean, std, device) -> dict:
    env = ObstacleVelocityAviary(scene, SPEED_LIMIT, RAY_RANGE, gui=False)
    seed = stable_seed(task["task_id"])
    collision = ground = nonfinite = executed_clip = support = 0
    support_total = 0
    action_trace = []
    success = False
    min_goal_distance = float("inf")
    error = ""
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(MAX_EPISODE_STEPS):
            action, diag = predict_action(model, schedule, observation, mean, std, seed + step, device)
            support += int(diag["support_violation_count"]); support_total += HORIZON * DDIM_STEPS
            if not np.isfinite(action).all() or not np.isfinite(observation).all():
                nonfinite += 1; break
            action_trace.append(action.copy())
            observation, _, _, _, _ = env.step(action)
            executed_clip += int(bool(env.last_velocity_action and env.last_velocity_action.clipped))
            obstacle_now, ground_now = env.contact_counts(); collision += obstacle_now; ground += ground_now
            state = np.asarray(env._getDroneStateVector(0), dtype=float)
            if not np.isfinite(state[:16]).all() or not np.isfinite(observation).all():
                nonfinite += 1; break
            distance = float(np.linalg.norm(state[:3] - scene.goal)); min_goal_distance = min(min_goal_distance, distance)
            if distance <= GOAL_TOLERANCE: success = True; break
            if collision or ground: break
    except Exception as exc:
        error = str(exc); nonfinite += 1
    finally:
        env.close()
    timeout = bool(not success and not collision and not ground and not nonfinite and len(action_trace) >= MAX_EPISODE_STEPS)
    array = np.asarray(action_trace, dtype=np.float32)
    return {"task_id": task["task_id"], "scene_id": scene.scene_id, "family": scene.family,
            "success": bool(success), "collision": int(collision > 0), "ground_contact": int(ground > 0),
            "timeout": timeout, "nonfinite": int(nonfinite > 0), "steps": len(action_trace),
            "time": len(action_trace) / scene.control_hz, "minimum_goal_distance": min_goal_distance if np.isfinite(min_goal_distance) else None,
            "final_goal_distance": None if not action_trace else float(np.linalg.norm(state[:3] - scene.goal)),
            "support_violation_count": support, "support_violation_fraction": support / max(1, support_total),
            "executed_action_clip_count": executed_clip, "executed_action_clip_fraction": executed_clip / max(1, len(action_trace)),
            "action_trace_hash": hashlib.sha256(array.tobytes()).hexdigest(), "error": error}


def family_gate(summary: dict) -> bool:
    return all(summary["by_family"][family]["success"] > 0 for family in ("OPEN", "BLOCK", "SBEND"))


def eval_gate(summary: dict, rows: list[dict], leakage: dict) -> bool:
    overall = summary["overall"]
    total = max(1, len(rows))
    unsafe = overall["collision"] + overall["ground_contact"]
    support = sum(row["support_violation_count"] for row in rows) / max(1, sum(row["steps"] * HORIZON * DDIM_STEPS for row in rows))
    clipping = sum(row["executed_action_clip_count"] for row in rows) / max(1, sum(row["steps"] for row in rows))
    return bool(overall["success"] >= 27 and family_gate(summary) and unsafe <= 13 and overall["nonfinite"] == 0 and
                support <= 0.01 and clipping <= 0.01 and leakage["passed"] and total == 54)


def rollout_summary(rows: list[dict]) -> dict:
    summary = summarize_rollouts(rows)
    summary["unsafe"] = summary["overall"]["collision"] + summary["overall"]["ground_contact"]
    summary["support_violation_fraction"] = sum(row["support_violation_count"] for row in rows) / max(1, sum(row["steps"] * HORIZON * DDIM_STEPS for row in rows))
    summary["executed_action_clipping_fraction"] = sum(row["executed_action_clip_count"] for row in rows) / max(1, sum(row["steps"] for row in rows))
    return summary


def teacher_independence() -> dict:
    source = inspect.getsource(evaluate_task).lower()
    forbidden = [token for token in ("expert", "trajectory", "planner", "v_ref", "corridor", "privileged") if token in source]
    return {"passed": not forbidden, "forbidden_runtime_tokens": forbidden, "policy_function": "evaluate_task"}


_R2_CONTEXT = {}


def main() -> None:
    global _R2_CONTEXT
    identity = assert_identity()
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_S3R2_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    set_deterministic(TRAIN_SEED)
    r1_model, r1_schedule, r1_payload = load_r1_model(device)
    train_data = load_npz(S2_DATASET / "train.npz")
    val_data = load_npz(S2_DATASET / "val.npz")
    mean = np.asarray(r1_payload["observation_mean"], dtype=np.float32)
    std = np.asarray(r1_payload["observation_std"], dtype=np.float32)
    # Recovery teacher verification replays the frozen R1 prefix.  Populate
    # this context before the planner loop; it is never used to generate the
    # student rollout or to select anchors.
    _R2_CONTEXT = {"model": r1_model, "schedule": r1_schedule, "mean": mean,
                   "std": std, "device": device,
                   "model_config": {"observation_dim": 34, "horizon": 16, "action_dim": 3,
                                    "observation_width": 128, "time_dim": 64, "width": 256,
                                    "residual_blocks": 4, "bounded_output": True}}
    train_expert = expert_episode_index(train_data)
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    manifest = [row for row in read_csv(S2_ROOT / "task_manifest.csv") if row["accepted"].lower() == "true" and row["split"] == "train"]
    manifest = sorted(manifest, key=lambda row: row["task_id"])
    train_rollouts = None
    if TRAIN_ROLLOUT_CACHE.exists():
        try:
            import pickle
            with TRAIN_ROLLOUT_CACHE.open("rb") as handle:
                cached = pickle.load(handle)
            if (cached.get("start_head") == START_HEAD and cached.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256 and
                    cached.get("train_sha256") == EXPECTED_DATASET_SHA256["train"] and len(cached.get("rows", [])) == 252):
                train_rollouts = cached["rows"]
                print("loaded temporary TRAIN rollout cache", flush=True)
        except Exception:
            train_rollouts = None
    if train_rollouts is None:
        train_rollouts = []
        for index, row in enumerate(manifest):
            scene, task = row_task(row)
            candidate_id = int(row["candidate_id"])
            result = train_student_rollout(r1_model, r1_schedule, scene, task, mean, std, device)
            result["anchor"] = select_first_anchor(result, train_expert[(row["scene_id"], candidate_id)])
            train_rollouts.append(result)
            print(f"train_rollout={index + 1}/252 task={row['task_id']} outcome={result['outcome']} anchor={result['anchor'] is not None}", flush=True)
        import pickle
        with TRAIN_ROLLOUT_CACHE.open("wb") as handle:
            pickle.dump({"start_head": START_HEAD, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                         "train_sha256": EXPECTED_DATASET_SHA256["train"], "rows": train_rollouts},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    anchors = [row for row in train_rollouts if row["anchor"] is not None]
    if len({row["task_id"] for row in anchors}) != len(anchors):
        raise RuntimeError("BLOCKED_S3R2_ANCHOR_DUPLICATE")
    recovery_manifest = []
    accepted = []
    planner_attempts = planner_successes = 0
    for index, rollout in enumerate(anchors):
        task_row = next(row for row in manifest if row["task_id"] == rollout["task_id"])
        scene, task = row_task(task_row)
        anchor = rollout["anchor"]
        planner_attempts += 1
        planner = planner_attempt(scene, task, anchor)
        planner_successes += int(planner.get("planner_success", False))
        record = {"task_id": task["task_id"], "scene_id": scene.scene_id, "family": scene.family,
                  "source_split": "TRAIN", "anchor_step": anchor["step"], "anchor_time": anchor["time"],
                  "anchor_position": anchor["position"], "anchor_velocity": anchor["velocity"],
                  "anchor_quaternion": anchor["quaternion"], "anchor_angular_velocity": anchor["angular_velocity"],
                  "anchor_position_error": anchor["position_error_to_expert"], "student_outcome": rollout["outcome"],
                  "planner_success": bool(planner.get("planner_success", False)),
                  "planner_interface": planner.get("planner_interface", "position_only_static_zero_derivatives"),
                  "planner_error": planner.get("error", "")}
        if planner.get("planner_success"):
            verified, data = teacher_verify(scene, task, anchor, planner, mean)
            record.update(verified)
            if verified["accepted"]:
                record["recovery_episode_id"] = len(accepted)
                accepted.append(({**record, "task_id": task["task_id"], "scene_id": scene.scene_id}, data))
        else:
            record.update({"accepted": False, "reference_completed": False, "goal_reached": False,
                           "collision_count": None, "ground_contact_count": None, "nonfinite_count": None,
                           "clip_count": None, "episode_length": None, "recovery_steps": None,
                           "minimum_goal_distance": None, "switched_state_position_error": None,
                           "error": planner.get("error", "planner_failed")})
        recovery_manifest.append(record)
        print(f"recovery={index + 1}/{len(anchors)} task={task['task_id']} accepted={record['accepted']}", flush=True)

    accepted_by_family = {family: sum(row["family"] == family for row in recovery_manifest if row.get("accepted")) for family in ("OPEN", "BLOCK", "SBEND")}
    if len(accepted) < 90 or not all(accepted_by_family[family] > 0 for family in accepted_by_family):
        write_csv(S3R2_ROOT / "recovery_manifest.csv", recovery_manifest)
        summary = {"task": "S3-R2-TRAIN-ONLY-RECOVERY-REPLAN-DIFFUSION-AUGMENTATION-V1",
                   "final_label": "BLOCKED_S3R2_RECOVERY_TEACHER_YIELD", "start_head": START_HEAD,
                   "branch": BRANCH, "dataset_identity": identity, "train_task_rollouts": 252,
                   "divergent_train_tasks": len(anchors), "recovery_anchors": len(anchors),
                   "recovery_planner_attempts": planner_attempts, "recovery_planner_success": planner_successes,
                   "accepted_recovery_trajectories": {"total": len(accepted), **accepted_by_family},
                   "recovery_source_split": "TRAIN_ONLY", "val_recovery_samples": 0, "test_recovery_samples": 0,
                   "current_blocker": "accepted recovery trajectories below 90 or family yield missing",
                   "formal_stage": "S3", "formal_progress": "30%", "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW",
                   "waiting_for_high_level_controller_audit": True}
        S3R2_ROOT.mkdir(parents=True, exist_ok=True)
        (S3R2_ROOT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError("BLOCKED_S3R2_RECOVERY_TEACHER_YIELD")

    recovery_identity = write_recovery_npz(accepted)
    write_csv(S3R2_ROOT / "recovery_manifest.csv", recovery_manifest)
    recovery_data = load_npz(RECOVERY_DATASET)
    original_train_obs, original_train_actions = build_windows(train_data)
    recovery_obs, recovery_actions = build_windows(recovery_data)
    original_val_obs, original_val_actions = build_windows(val_data)
    identity["original"] = dict(identity)
    identity["recovery"] = recovery_identity
    identity["recovery_source_split"] = "TRAIN_ONLY"
    identity["val_recovery_samples"] = 0
    identity["test_recovery_samples"] = 0
    identity["windows"] = {"original_train": len(original_train_obs), "recovery_train": len(recovery_obs),
                            "total_r2_train": len(original_train_obs) + len(recovery_obs), "val": len(original_val_obs)}
    r2_obs = np.concatenate([original_train_obs, recovery_obs], axis=0)
    r2_actions = np.concatenate([original_train_actions, recovery_actions], axis=0)
    _R2_CONTEXT = {"model": r1_model, "schedule": r1_schedule,
                   "model_config": {"observation_dim": 34, "horizon": 16, "action_dim": 3,
                                    "observation_width": 128, "time_dim": 64, "width": 256,
                                    "residual_blocks": 4, "bounded_output": True},
                   "mean": mean, "std": std, "device": device}
    train_obs = torch.from_numpy(((r2_obs - mean) / np.maximum(std, 1e-6)).astype(np.float32)).to(device)
    train_actions = torch.from_numpy(r2_actions.astype(np.float32)).to(device)
    val_obs = torch.from_numpy(((original_val_obs - mean) / np.maximum(std, 1e-6)).astype(np.float32)).to(device)
    val_actions = torch.from_numpy(original_val_actions.astype(np.float32)).to(device)
    best_path, last_path, curve, best_update, best_val, wall, peak = train_model(
        train_obs, train_actions, val_obs, val_actions, mean, std, device, identity)
    write_csv(S3R2_ROOT / "training_curve.csv", curve, ["update", "train_loss", "val_loss", "gradient_norm"])
    payload = torch.load(best_path, map_location=device, weights_only=False)
    payload["model_frozen"] = True
    payload["model_freeze_timestamp"] = "S3R2_POST_TRAINING_BEFORE_VAL"
    torch.save(payload, best_path)
    model = ConditionalDiffusionMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"]); model.eval()
    schedule = DiffusionSchedule(DIFFUSION_STEPS).to(device)
    val_rows = []
    val_manifest = [row for row in read_csv(S2_ROOT / "task_manifest.csv") if row["accepted"].lower() == "true" and row["split"] == "val"]
    for row in sorted(val_manifest, key=lambda item: item["task_id"]):
        scene, task = row_task(row); val_rows.append(evaluate_task(model, schedule, scene, task, mean, std, device))
    val_summary = rollout_summary(val_rows)
    leakage = teacher_independence()
    val_gate = eval_gate(val_summary, val_rows, leakage)
    write_csv(S3R2_ROOT / "val_rollout.csv", val_rows)
    r1 = {"success": 29, "collision": 9, "ground_contact": 6, "timeout": 10, "unsafe": 15}
    val_overall = val_summary["overall"]
    summary = {"task": "S3-R2-TRAIN-ONLY-RECOVERY-REPLAN-DIFFUSION-AUGMENTATION-V1",
               "final_label": "PENDING_VAL", "start_head": START_HEAD, "r1_policy_head": R1_POLICY_HEAD,
               "branch": BRANCH, "end_head": None, "remote_branch_head": None, "gpu": torch.cuda.get_device_name(0),
               "cpu": "24 cores", "pytorch": torch.__version__, "cuda": torch.version.cuda or "none",
               "train_device": str(device), "peak_gpu_memory_bytes": peak, "training_wall_time_s": wall,
               "r1_checkpoint_identity": {"sha256": EXPECTED_CHECKPOINT_SHA256, "path": str(R1_CHECKPOINT)},
               "original_dataset_identity": identity["original"], "train_task_rollouts": 252,
               "divergent_train_tasks": len(anchors), "recovery_anchors": len(anchors),
               "recovery_planner_attempts": planner_attempts, "recovery_planner_success": planner_successes,
               "accepted_recovery_trajectories": {"total": len(accepted), **accepted_by_family},
               "recovery_rejection_reasons": {}, "original_train_windows": len(original_train_obs),
               "recovery_train_windows": len(recovery_obs), "total_r2_train_windows": len(r2_obs),
               "recovery_window_fraction": len(recovery_obs) / len(r2_obs), "val_recovery_samples": 0,
               "test_recovery_samples": 0, "model_changed": False, "hyperparameters_changed": False,
               "model": {"architecture": "34-128-128; time64; width256; 4 residual MLP blocks",
                          "horizon": 16, "action_dim": 3, "prediction_target": "x0", "action_support": "unit_ball_squash",
                          "diffusion_steps": 100, "ddim_steps": 10, "ddim_eta": 0.0},
               "hyperparameters": {"train_seed": TRAIN_SEED, "batch_size": 512, "lr": 3e-4,
                                    "weight_decay": 1e-4, "gradient_clip": 1.0, "max_updates": 10000,
                                    "validation_interval": 500},
               "training": {"updates": 10000, "best_update": best_update, "best_val_loss": best_val, "all_finite": True,
                            "best_checkpoint_sha256": sha256_file(best_path)},
               "val": {**val_overall, "unsafe": val_summary["unsafe"], "support_violation_fraction": val_summary["support_violation_fraction"],
                       "executed_action_clipping_fraction": val_summary["executed_action_clipping_fraction"],
                       "by_family": val_summary["by_family"]}, "r1_to_r2_delta": {
                   "success": val_overall["success"] - r1["success"], "collision": val_overall["collision"] - r1["collision"],
                   "ground_contact": val_overall["ground_contact"] - r1["ground_contact"], "timeout": val_overall["timeout"] - r1["timeout"],
                   "unsafe": val_summary["unsafe"] - r1["unsafe"]},
               "val_gate": "PASS" if val_gate else "FAIL", "teacher_leakage": leakage,
               "r2_test_executed": False, "r2_test_run_count": 0, "determinism": None, "tests": None,
               "what_was_proven": ["TRAIN-only first-divergence recovery replans were collected and used for training" if val_gate else "TRAIN-only recovery training completed"],
               "what_was_not_proven": ["No TEST result before VAL Gate", "No PPO/BC comparison", "No cross-topology claim"],
               "current_blocker": "S3-R2 VAL Gate failed" if not val_gate else "awaiting TEST result",
               "formal_stage": "S3", "formal_progress": "30%", "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW",
               "waiting_for_high_level_controller_audit": True}
    if not val_gate:
        if val_summary["nonfinite"] > 0 or val_summary["support_violation_fraction"] > 0.01:
            summary["final_label"] = "BLOCKED_S3R2_RECOVERY_NUMERICS"
        elif val_summary["unsafe"] > 13 or val_summary["success"] < 27:
            summary["final_label"] = "BLOCKED_S3R2_RECOVERY_UNSAFE" if val_summary["unsafe"] > 13 else "BLOCKED_S3R2_RECOVERY_VAL_WEAK"
        S3R2_ROOT.mkdir(parents=True, exist_ok=True)
        (S3R2_ROOT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    test_data = load_npz(S2_DATASET / "test.npz")
    test_rows = []
    test_manifest = [row for row in read_csv(S2_ROOT / "task_manifest.csv") if row["accepted"].lower() == "true" and row["split"] == "test"]
    for row in sorted(test_manifest, key=lambda item: item["task_id"]):
        scene, task = row_task(row); test_rows.append(evaluate_task(model, schedule, scene, task, mean, std, device))
    test_summary = rollout_summary(test_rows)
    write_csv(S3R2_ROOT / "test_rollout.csv", test_rows)
    test_gate = eval_gate(test_summary, test_rows, leakage)
    summary.update({"final_label": "PASS_S3R2_RECOVERY_AUGMENTED_DIFFUSION_SANITY" if test_gate else "BLOCKED_S3R2_RECOVERY_UNSAFE",
                    "r2_test_executed": True, "r2_test_run_count": 1, "test_used_for_selection": False,
                    "test": test_summary, "current_blocker": "none" if test_gate else "S3-R2 TEST Gate failed"})
    (S3R2_ROOT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
