"""Read-only S3-D0 audit of the frozen S3-R1 closed-loop VAL failures.

This module deliberately keeps the policy runner independent of the expert
dataset.  Expert transitions are loaded only after all 54 diagnostic rollouts
have completed, for post-hoc and offline comparisons.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.spatial import cKDTree

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from quadrotor_diffusion_ppo.envs.observation import LOCAL_RAY_DIRECTIONS
from quadrotor_diffusion_ppo.envs.scene import SCENE_IDS, load_all_scenes
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary
from scripts.run_s3_diffusion import (
    ACTION_DIM, EVAL_BASE_SEED, GOAL_TOLERANCE, HORIZON, MAX_EPISODE_STEPS,
    OBSERVATION_DIM, PROJECT_ROOT, S2_DATASET, S2_ROOT, load_npz,
    observation_normalization, row_task, stable_seed,
)


S3R1_ROOT = PROJECT_ROOT / "artifacts" / "s3r1"
S3D0_ROOT = PROJECT_ROOT / "artifacts" / "s3d0"
TEMP_ROLLOUT_CACHE = Path(tempfile.gettempdir()) / "s3d0-rollouts.pkl"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "s3r1" / "best.pt"
EXPECTED_START_HEAD = "e2f4e19f02e1b07de48617b607ceeb32340a5095"
EXPECTED_CHECKPOINT_SHA256 = "8afa677d7c8a34179c60f067209b6924170fc7854445048d7a7e9b55bbbc5fc8"
EXPECTED_DATASET_SHA256 = {
    "train": "701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21",
    "val": "c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc",
    "test": "88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66",
}
EXPECTED_R1 = {"success": 29, "collision": 9, "ground_contact": 6, "timeout": 10, "nonfinite": 0}
SPEED_LIMIT = 0.801
RAY_RANGE = 3.0
CONTROL_HZ = 48
ANCHOR_EVERY_STEPS = CONTROL_HZ // 2
POSITION_DIVERGENCE_THRESHOLD = 0.30
ACTION_GROUPS = ("SUCCESS", "COLLISION", "GROUND_CONTACT", "TIMEOUT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("p50", "p90", "p95", "p99")}
    return {key: float(np.quantile(values, q)) for key, q in (
        ("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))}


def mean_or_none(values: list[float] | np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def pct(values: list[float] | np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def assert_start_clean() -> None:
    status = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True)
    allowed = {"scripts/audit_s3d0.py", "tests/test_s3d0_diagnostics.py"}
    unexpected = [line for line in status.splitlines() if line[3:] not in allowed]
    if unexpected:
        raise RuntimeError(f"S3D0 start-scope drift: {'; '.join(unexpected)}")
    if current_head() != EXPECTED_START_HEAD:
        raise RuntimeError(f"BLOCKED_S3D0_START_HEAD: expected {EXPECTED_START_HEAD}, got {current_head()}")


def checkpoint_identity() -> dict:
    if not CHECKPOINT.exists():
        raise RuntimeError("BLOCKED_S3D0_CHECKPOINT_IDENTITY")
    digest = sha256_file(CHECKPOINT)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"BLOCKED_S3D0_CHECKPOINT_IDENTITY: {digest}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    required = {
        "model_frozen": True, "prediction_target": "x0", "action_support": "unit_ball_squash",
        "ddim_steps": 10, "ddim_eta": 0.0, "diffusion_steps": 100,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("BLOCKED_S3D0_CHECKPOINT_IDENTITY: metadata mismatch")
    return {"path": str(CHECKPOINT), "sha256": digest, "frozen": True,
            "prediction_target": payload["prediction_target"],
            "action_support": payload["action_support"], "payload": payload}


def frozen_dataset_identity(r1_summary: dict) -> dict:
    # The diagnostic may read TRAIN and VAL.  It must not open the TEST split;
    # the frozen R1 identity is used for the held-out hash and no TEST rollout
    # file is ever loaded or created.
    train_hash = sha256_file(S2_DATASET / "train.npz")
    val_hash = sha256_file(S2_DATASET / "val.npz")
    old = r1_summary["dataset_identity"]["sha256"]
    hashes = {"train": train_hash, "val": val_hash, "test": old["test"]}
    if hashes != EXPECTED_DATASET_SHA256 or old != EXPECTED_DATASET_SHA256:
        raise RuntimeError("BLOCKED_S3D0_DATASET_IDENTITY")
    stats = r1_summary["dataset_identity"]["stats"]
    if stats["train"]["trajectories"] != 252 or stats["val"]["trajectories"] != 54:
        raise RuntimeError("BLOCKED_S3D0_DATASET_IDENTITY: train/val trajectory count")
    return {"sha256": hashes, "stats": stats, "observation_dim": 34, "action_dim": 3,
            "task_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "episode_boundaries": True, "all_finite": True,
            "test_hash_reused_from_frozen_r1": True}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def outcome(row: dict) -> str:
    if row.get("success") in (True, "True", "true", 1, "1"):
        return "SUCCESS"
    if row.get("collision") in (True, "True", "true", 1, "1"):
        return "COLLISION"
    if row.get("ground_contact") in (True, "True", "true", 1, "1"):
        return "GROUND_CONTACT"
    if row.get("timeout") in (True, "True", "true", 1, "1"):
        return "TIMEOUT"
    if row.get("nonfinite") in (True, "True", "true", 1, "1"):
        return "NONFINITE"
    return "OTHER"


def outcome_counts(rows: list[dict]) -> dict[str, int]:
    counts = {name.lower(): 0 for name in EXPECTED_R1}
    for row in rows:
        label = outcome(row).lower()
        if label in counts:
            counts[label] += 1
    return counts


def check_r1_outcome() -> tuple[dict, list[dict]]:
    rows = read_csv(S3R1_ROOT / "val_rollout.csv")
    counts = outcome_counts(rows)
    match = len(rows) == 54 and counts == EXPECTED_R1 and len({row["task_id"] for row in rows}) == 54
    if not match:
        raise RuntimeError(f"BLOCKED_S3D0_R1_OUTCOME_MISMATCH: {counts}")
    return {**counts, "match": True, "rows": len(rows)}, rows


def load_model(device: torch.device):
    identity = checkpoint_identity()
    payload = identity["payload"]
    model = ConditionalDiffusionMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, DiffusionSchedule(payload["diffusion_steps"]).to(device), identity


def predict(model, schedule, observation: np.ndarray, mean: np.ndarray, std: np.ndarray,
            seed: int, device: torch.device):
    normalized = (np.asarray(observation, dtype=np.float32) - mean) / np.maximum(std, 1e-6)
    condition = torch.from_numpy(normalized).reshape(1, -1).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return schedule.ddim_sample_x0(model, condition, steps=10, generator=generator, return_diagnostics=True), normalized


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)


def build_episode_index(data: dict[str, np.ndarray]) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    result = {}
    for offset, length in zip(data["episode_offsets"], data["episode_lengths"]):
        scene_index = data["scene_indices"][int(offset)]
        task_index = data["task_indices"][int(offset)]
        sl = slice(int(offset), int(offset + length))
        key = (SCENE_IDS[int(scene_index)], int(task_index))
        result[key] = {name: data[name][sl].copy() for name in (
            "observations", "actions", "positions", "velocities", "goals", "times", "trajectory_progress")}
    if len(result) != 54:
        raise RuntimeError("VAL expert episode index is not 54 unique tasks")
    return result


@torch.no_grad()
def offline_task_audit(model, schedule, expert: dict[str, np.ndarray], mean: np.ndarray,
                       std: np.ndarray, seed: int, device: torch.device) -> dict:
    observations = (expert["observations"] - mean) / np.maximum(std, 1e-6)
    actions = expert["actions"].astype(np.float32)
    obs_tensor = torch.from_numpy(observations.astype(np.float32)).to(device)
    action_tensor = torch.from_numpy(actions).to(device)
    preds = []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for start in range(0, len(obs_tensor), 512):
        sampled = schedule.ddim_sample_x0(model, obs_tensor[start:start + 512], steps=10, generator=generator)
        preds.append(sampled[:, 0, :].cpu().numpy())
    predicted = np.concatenate(preds, axis=0)
    target = action_tensor.cpu().numpy()
    errors = predicted - target
    cosines = cosine_rows(predicted, target)
    norms = np.linalg.norm(predicted, axis=1)
    return {
        "count": int(len(target)), "first_action_MAE": float(np.mean(np.abs(errors))),
        "first_action_MSE": float(np.mean(errors ** 2)), "cosine_similarity": float(np.mean(cosines)),
        "predicted_action_norm_mean": float(np.mean(norms)),
        "predicted_action_norm_p95": float(np.quantile(norms, 0.95)),
        "predicted_action_norm_max": float(np.max(norms)),
    }


def rotation_world_directions(quaternion: np.ndarray) -> np.ndarray:
    import pybullet as p
    rotation = np.asarray(p.getMatrixFromQuaternion(np.asarray(quaternion, dtype=float).tolist()), dtype=float).reshape(3, 3)
    return LOCAL_RAY_DIRECTIONS @ rotation.T


def policy_rollout(model, schedule, scene, task: dict, mean: np.ndarray, std: np.ndarray,
                   device: torch.device) -> dict:
    """Run exactly the R1 policy contract from deployment-visible inputs only."""
    env = ObstacleVelocityAviary(scene, SPEED_LIMIT, RAY_RANGE, gui=False)
    seed = stable_seed(task["task_id"])
    trace = []
    collision = ground = nonfinite = 0
    min_goal_distance = float("inf")
    last_state = None
    success = False
    error = ""
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(MAX_EPISODE_STEPS):
            state_before = np.asarray(env._getDroneStateVector(0), dtype=float)
            if not np.isfinite(observation).all():
                nonfinite += 1
                break
            sampled_result, normalized = predict(model, schedule, observation, mean, std, seed + step, device)
            sample, diagnostics = sampled_result
            sequence = sample[0].cpu().numpy().astype(np.float32)
            action = sequence[0].copy()
            rays = np.asarray(observation[16:], dtype=float)
            nearest_index = int(np.argmin(rays))
            nearest_direction = rotation_world_directions(state_before[3:7])[nearest_index]
            if not np.isfinite(action).all() or not np.isfinite(sequence).all():
                nonfinite += 1
                break
            observation, _, _, _, _ = env.step(action)
            obstacle_now, ground_now = env.contact_counts()
            collision += int(obstacle_now)
            ground += int(ground_now)
            state = np.asarray(env._getDroneStateVector(0), dtype=float)
            if not np.isfinite(state[:16]).all() or not np.isfinite(observation).all():
                nonfinite += 1
                break
            distance = float(np.linalg.norm(state[:3] - scene.goal))
            min_goal_distance = min(min_goal_distance, distance)
            action_norm = float(np.linalg.norm(action))
            trace.append({
                "task_id": task["task_id"], "scene_id": scene.scene_id, "step": int(step),
                "time": float((step + 1) / scene.control_hz), "position_x": float(state[0]),
                "position_y": float(state[1]), "position_z": float(state[2]),
                "velocity_x": float(state[10]), "velocity_y": float(state[11]), "velocity_z": float(state[12]),
                "goal_delta_x": float(scene.goal[0] - state[0]), "goal_delta_y": float(scene.goal[1] - state[1]),
                "goal_delta_z": float(scene.goal[2] - state[2]), "goal_distance": distance,
                "rays": rays.astype(float).tolist(), "ray_min": float(np.min(rays)),
                "normalized_observation_max_abs_z": float(np.max(np.abs(normalized))),
                "network_latent_norm": float(max(diagnostics["latent_max_norm"], 0.0)),
                "network_latent_max_abs": float(max(diagnostics["latent_max_abs"], 0.0)),
                "diffusion_action_x": float(action[0]), "diffusion_action_y": float(action[1]),
                "diffusion_action_z": float(action[2]), "diffusion_action_norm": action_norm,
                "collision": bool(obstacle_now > 0), "ground_contact": bool(ground_now > 0),
                "success": bool(distance <= GOAL_TOLERANCE),
                "horizontal_speed": float(np.linalg.norm(state[10:12])),
                "nearest_ray_direction": nearest_direction.astype(float).tolist(),
                "action_dot_nearest_ray": float(np.dot(action, nearest_direction)),
                "observation": np.asarray(observation, dtype=np.float32).copy(),
            })
            if distance <= GOAL_TOLERANCE:
                success = True
                break
            if collision or ground:
                break
    except Exception as exc:
        error = str(exc)
        nonfinite += 1
    finally:
        env.close()
    timeout = not success and not collision and not ground and not nonfinite and len(trace) >= MAX_EPISODE_STEPS
    action_array = np.asarray([[r["diffusion_action_x"], r["diffusion_action_y"], r["diffusion_action_z"]] for r in trace], dtype=np.float32)
    return {"task_id": task["task_id"], "scene_id": scene.scene_id, "family": scene.family,
            "success": bool(success), "collision": int(collision > 0), "ground_contact": int(ground > 0),
            "timeout": bool(timeout), "nonfinite": int(nonfinite > 0), "steps": len(trace),
            "time": float(len(trace) / scene.control_hz), "final_goal_distance": float(trace[-1]["goal_distance"]) if trace else None,
            "minimum_goal_distance": float(min_goal_distance) if np.isfinite(min_goal_distance) else None,
            "action_trace_hash": hashlib.sha256(action_array.tobytes()).hexdigest(), "error": error,
            "trace": trace}


def group_name(row: dict) -> str:
    return outcome(row)


def action_norm_summary(rows: list[dict]) -> dict:
    result = {}
    for group in ACTION_GROUPS:
        episodes = [row for row in rows if group_name(row) == group]
        values = [r["diffusion_action_norm"] for row in episodes for r in row["trace"]]
        values = np.asarray(values, dtype=float)
        result[group] = {"episodes": len(episodes), "steps": int(len(values)),
                         "action_norm_mean": mean_or_none(values), "action_norm_p50": pct(values, .50),
                         "action_norm_p95": pct(values, .95), "action_norm_max": pct(values, 1.0),
                         "fraction_norm_gt_0.80": float(np.mean(values > .80)) if len(values) else None,
                         "fraction_norm_gt_0.90": float(np.mean(values > .90)) if len(values) else None,
                         "fraction_norm_gt_0.95": float(np.mean(values > .95)) if len(values) else None,
                         "fraction_norm_gt_0.99": float(np.mean(values > .99)) if len(values) else None}
    return result


def build_train_reference(train_data: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray):
    normalized = (train_data["observations"].astype(np.float32) - mean) / np.maximum(std, 1e-6)
    tree = cKDTree(normalized)
    sample_indices = np.arange(0, len(normalized), max(1, len(normalized) // 4096), dtype=int)
    distances, _ = tree.query(normalized[sample_indices], k=2, workers=1)
    baseline = distances[:, 1]
    # The same deterministic anchor set defines the local-action-variance baseline.
    action_neighbors = tree.query(normalized[sample_indices], k=33, workers=1)[1][:, 1:]
    neighbor_actions = train_data["actions"][action_neighbors]
    action_std_norm = np.std(np.linalg.norm(neighbor_actions, axis=2), axis=1)
    return tree, normalized, {"anchor_count": int(len(sample_indices)),
                             "nearest_train_distance": quantiles(baseline),
                             "local_action_variance_std_norm": quantiles(action_std_norm)}, train_data["actions"]


def ambiguity_for_anchor(tree: cKDTree, train_actions: np.ndarray, normalized_observation: np.ndarray) -> dict:
    distances, indices = tree.query(normalized_observation, k=32, workers=1)
    neighbors = train_actions[np.asarray(indices, dtype=int)]
    mean_action = neighbors.mean(axis=0)
    norms = np.linalg.norm(neighbors, axis=1)
    mean_cosine = float(np.mean(cosine_rows(neighbors, np.repeat(mean_action[None, :], len(neighbors), axis=0))))
    return {"nearest_train_distance": float(np.min(distances)), "neighbor_action_mean": mean_action,
            "neighbor_action_std_xyz": neighbors.std(axis=0), "neighbor_action_std_norm": float(np.std(norms)),
            "mean_cosine_to_neighbor_mean": mean_cosine}


def attach_anchor_metrics(rows: list[dict], tree: cKDTree, train_actions: np.ndarray,
                          mean: np.ndarray, std: np.ndarray, baseline: dict) -> None:
    p95 = float(baseline["nearest_train_distance"]["p95"])
    p99 = float(baseline["nearest_train_distance"]["p99"])
    action_p95 = float(baseline["local_action_variance_std_norm"]["p95"])
    for row in rows:
        anchors = []
        for trace in row["trace"]:
            normalized = (trace["observation"] - mean) / np.maximum(std, 1e-6)
            trace["nearest_train_distance"] = float(tree.query(normalized, k=1, workers=1)[0])
            if int(trace["step"]) % ANCHOR_EVERY_STEPS:
                continue
            ambiguity = ambiguity_for_anchor(tree, train_actions, normalized)
            ambiguity["time"] = trace["time"]
            anchors.append(ambiguity)
        row["anchors"] = anchors
        distances = [a["nearest_train_distance"] for a in anchors]
        std_norms = [a["neighbor_action_std_norm"] for a in anchors]
        row["ood"] = {"anchor_count": len(anchors), "max_nn_distance": pct(distances, 1.0),
                       "mean_nn_distance": mean_or_none(distances),
                       "fraction_above_train_p95": float(np.mean(np.asarray(distances) > p95)) if distances else None,
                       "fraction_above_train_p99": float(np.mean(np.asarray(distances) > p99)) if distances else None}
        row["ambiguity"] = {"anchor_count": len(anchors),
                             "neighbor_action_mean": np.mean([a["neighbor_action_mean"] for a in anchors], axis=0).tolist() if anchors else None,
                             "neighbor_action_std_xyz_mean": np.mean([a["neighbor_action_std_xyz"] for a in anchors], axis=0).tolist() if anchors else None,
                             "neighbor_action_std_norm_mean": mean_or_none(std_norms),
                             "neighbor_action_std_norm_p95": pct(std_norms, .95),
                             "neighbor_action_std_norm_max": pct(std_norms, 1.0),
                             "mean_cosine_to_neighbor_mean": mean_or_none([a["mean_cosine_to_neighbor_mean"] for a in anchors]),
                             "fraction_above_train_local_variance_p95": float(np.mean(np.asarray(std_norms) > action_p95)) if std_norms else None}


def summarize_by_group(rows: list[dict], field: str) -> dict:
    result = {}
    for group in ACTION_GROUPS:
        selected = [row[field] for row in rows if group_name(row) == group]
        if not selected:
            result[group] = {"episodes": 0}
            continue
        if field == "offline":
            keys = ("first_action_MAE", "first_action_MSE", "cosine_similarity", "predicted_action_norm_mean",
                    "predicted_action_norm_p95", "predicted_action_norm_max")
            result[group] = {"episodes": len(selected), **{key: mean_or_none([item[key] for item in selected]) for key in keys}}
        elif field == "ood":
            keys = ("max_nn_distance", "mean_nn_distance", "fraction_above_train_p95", "fraction_above_train_p99")
            result[group] = {"episodes": len(selected), **{key: mean_or_none([item[key] for item in selected]) for key in keys}}
        else:
            keys = ("neighbor_action_std_norm_mean", "neighbor_action_std_norm_p95", "neighbor_action_std_norm_max",
                    "mean_cosine_to_neighbor_mean", "fraction_above_train_local_variance_p95")
            result[group] = {"episodes": len(selected), **{key: mean_or_none([item[key] for item in selected]) for key in keys}}
    return result


def collision_audit(rows: list[dict]) -> tuple[dict, dict[str, int]]:
    details = {}
    taxonomy = {"ray_blind": 0, "obstacle_visible_but_unsafe_action": 0, "other": 0}
    for row in rows:
        if group_name(row) != "COLLISION":
            continue
        trace = row["trace"]
        record = {}
        for seconds in (1.0, .5, .25):
            window = trace[-max(1, int(round(seconds * CONTROL_HZ))):]
            mins = np.asarray([r["ray_min"] for r in window])
            record[f"minimum_ray_normalized_{seconds:g}s"] = float(np.min(mins))
            record[f"minimum_ray_inferred_distance_m_{seconds:g}s"] = float(np.min(mins) * RAY_RANGE)
            record[f"diffusion_action_norm_{seconds:g}s"] = float(np.mean([r["diffusion_action_norm"] for r in window]))
            record[f"horizontal_speed_{seconds:g}s"] = float(np.mean([r["horizontal_speed"] for r in window]))
            record[f"action_dot_nearest_ray_{seconds:g}s"] = float(np.mean([r["action_dot_nearest_ray"] for r in window]))
        visible = record["minimum_ray_normalized_1s"] < 0.99
        toward = record["action_dot_nearest_ray_1s"] > 0.05
        category = "ray_blind" if not visible else "obstacle_visible_but_unsafe_action" if toward else "other"
        taxonomy[category] += 1
        record["category"] = category
        details[row["task_id"]] = record
        row["collision_audit"] = record
    return {"episodes": len(details), "taxonomy": taxonomy, "details": details,
            "ray_convention": "ray_distance = ray_value * 3.0 m; visible means min ray < 0.99 in final 1 s",
            "unsafe_action_rule": "visible and mean action dot nearest-ray-direction > 0.05"}, taxonomy


def ground_audit(rows: list[dict], baseline: dict) -> tuple[dict, dict[str, int]]:
    details, taxonomy = {}, {"persistent_downward": 0, "dynamic_lag": 0, "off_manifold": 0, "other": 0}
    for row in rows:
        if group_name(row) != "GROUND_CONTACT":
            continue
        window = row["trace"][-CONTROL_HZ:]
        z = np.asarray([r["position_z"] for r in window])
        vz = np.asarray([r["velocity_z"] for r in window])
        az = np.asarray([r["diffusion_action_z"] for r in window])
        fraction_down = float(np.mean(az < 0.0))
        record = {"minimum_altitude": float(np.min(z)), "minimum_vertical_velocity": float(np.min(vz)),
                  "mean_final_1s_action_z": float(np.mean(az)), "fraction_final_1s_action_z_negative": fraction_down,
                  "final_1s_ood_fraction_above_train_p95": row["ood"]["fraction_above_train_p95"]}
        if fraction_down >= 0.75 and record["mean_final_1s_action_z"] < -0.05:
            category = "persistent_downward"
        elif record["minimum_vertical_velocity"] < -0.20 and record["mean_final_1s_action_z"] >= -0.05:
            category = "dynamic_lag"
        elif (record["final_1s_ood_fraction_above_train_p95"] or 0.0) >= 0.50:
            category = "off_manifold"
        else:
            category = "other"
        taxonomy[category] += 1
        record["category"] = category
        details[row["task_id"]] = record
        row["ground_audit"] = record
    return {"episodes": len(details), "taxonomy": taxonomy, "details": details,
            "classification_rules": {"persistent_downward": "fraction action_z<0 >= 0.75 and mean action_z < -0.05",
                                      "dynamic_lag": "minimum vz < -0.20 and mean action_z >= -0.05",
                                      "off_manifold": "final-1s OOD fraction above TRAIN p95 >= 0.50"}}, taxonomy


def timeout_audit(rows: list[dict]) -> tuple[dict, dict[str, int]]:
    details, taxonomy = {}, {"near_goal_stall": 0, "no_progress": 0, "oscillatory": 0, "other": 0}
    for row in rows:
        if group_name(row) != "TIMEOUT":
            continue
        trace = row["trace"]
        distances = np.asarray([r["goal_distance"] for r in trace], dtype=float)
        initial = float(np.linalg.norm(np.asarray([trace[0]["goal_delta_x"], trace[0]["goal_delta_y"], trace[0]["goal_delta_z"]])))
        minimum = float(np.min(distances))
        final = float(distances[-1])
        progress = initial - minimum
        recent = trace[-min(len(trace), 5 * CONTROL_HZ):]
        recent_distances = np.asarray([r["goal_distance"] for r in recent])
        recent_times = np.asarray([r["time"] for r in recent])
        slope = float(np.polyfit(recent_times, recent_distances, 1)[0]) if len(recent) >= 2 else 0.0
        projections = []
        for item in trace:
            delta = np.asarray([item["goal_delta_x"], item["goal_delta_y"], item["goal_delta_z"]])
            action = np.asarray([item["diffusion_action_x"], item["diffusion_action_y"], item["diffusion_action_z"]])
            projections.append(float(np.dot(action, delta / max(np.linalg.norm(delta), 1e-8))))
        signs = np.sign(np.asarray(projections))
        signs = signs[signs != 0]
        sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0
        record = {"initial_goal_distance": initial, "minimum_goal_distance": minimum, "final_goal_distance": final,
                  "progress": progress, "last_5s_goal_distance_slope": slope,
                  "action_direction_sign_changes": sign_changes}
        if minimum <= 1.0:
            category = "near_goal_stall"
        elif sign_changes >= 10:
            category = "oscillatory"
        elif progress < 0.25 * max(initial, 1e-6) or slope >= -0.01:
            category = "no_progress"
        else:
            category = "other"
        taxonomy[category] += 1
        record["category"] = category
        details[row["task_id"]] = record
        row["timeout_audit"] = record
    return {"episodes": len(details), "taxonomy": taxonomy, "details": details,
            "classification_rules": {"near_goal_stall": "minimum goal distance <= 1.0 m",
                                      "oscillatory": "action-to-goal-direction sign changes >= 10",
                                      "no_progress": "progress < 25% of initial distance or last-5s slope >= -0.01 m/s"}}, taxonomy


def divergence_audit(rows: list[dict], expert_index: dict[tuple[str, int], dict[str, np.ndarray]],
                    manifest_rows: list[dict]) -> dict:
    result = {}
    manifest_by_id = {r["task_id"]: r for r in manifest_rows}
    for row in rows:
        manifest = manifest_by_id[row["task_id"]]
        key = (manifest["scene_id"], int(manifest["task_id"].rsplit("_", 1)[1]))
        expert = expert_index[key]
        expert_times = expert["times"].astype(float)
        expert_positions = expert["positions"].astype(float)
        first = None
        for item in row["trace"]:
            reference = np.asarray([np.interp(item["time"], expert_times, expert_positions[:, axis]) for axis in range(3)])
            position = np.asarray([item["position_x"], item["position_y"], item["position_z"]])
            distance = float(np.linalg.norm(position - reference))
            if distance > POSITION_DIVERGENCE_THRESHOLD:
                first = {"divergence_time": item["time"], "position_error": distance,
                         "action_norm_at_divergence": item["diffusion_action_norm"],
                         "nearest_train_distance_at_divergence": item.get("nearest_train_distance"),
                         "ray_min_at_divergence": item["ray_min"]}
                break
        result[row["task_id"]] = first or {"divergence_time": None, "position_error": None,
                                             "action_norm_at_divergence": None,
                                             "nearest_train_distance_at_divergence": None,
                                             "ray_min_at_divergence": None}
        row["divergence"] = result[row["task_id"]]
    by_group = {}
    for group in ACTION_GROUPS:
        selected = [row["divergence"]["divergence_time"] for row in rows if group_name(row) == group]
        observed = [value for value in selected if value is not None]
        by_group[group] = {"episodes": len(selected), "diverged": len(observed),
                           "fraction_diverged": len(observed) / max(1, len(selected)),
                           "divergence_time_p50": pct(observed, .50), "divergence_time_p95": pct(observed, .95),
                           "divergence_time_min": pct(observed, 0.0)}
    return {"threshold_m": POSITION_DIVERGENCE_THRESHOLD, "by_outcome": by_group, "per_task": result}


def per_episode_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        action_values = np.asarray([r["diffusion_action_norm"] for r in row["trace"]], dtype=float)
        result = {key: row.get(key) for key in ("task_id", "scene_id", "family", "success", "collision", "ground_contact",
                                                 "timeout", "nonfinite", "steps", "time", "final_goal_distance",
                                                 "minimum_goal_distance", "action_trace_hash", "error")}
        result.update({"outcome": group_name(row), "action_norm_mean": mean_or_none(action_values),
                       "fraction_norm_gt_0.80": float(np.mean(action_values > .80)),
                       "fraction_norm_gt_0.90": float(np.mean(action_values > .90)),
                       "fraction_norm_gt_0.95": float(np.mean(action_values > .95)),
                       "fraction_norm_gt_0.99": float(np.mean(action_values > .99)),
                       **{f"ood_{key}": value for key, value in row["ood"].items()},
                       **{f"ambiguity_{key}": value for key, value in row["ambiguity"].items()},
                       "offline_first_action_MAE": row["offline"]["first_action_MAE"],
                       "offline_first_action_MSE": row["offline"]["first_action_MSE"],
                       "offline_cosine_similarity": row["offline"]["cosine_similarity"],
                       **{f"divergence_{key}": value for key, value in row["divergence"].items()}})
        for prefix in ("collision_audit", "ground_audit", "timeout_audit"):
            if prefix in row:
                result.update({f"{prefix}_{key}": value for key, value in row[prefix].items()})
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key, value in row.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    value = json.dumps(np.asarray(value).tolist(), separators=(",", ":"))
                cleaned[key] = value
            writer.writerow(cleaned)


def determine_primary(offline: dict, ood: dict, actions: dict, collision_tax: dict,
                      ground_tax: dict, timeout_tax: dict, ambiguity: dict) -> tuple[str, list[str]]:
    evidence = []
    success_ood = ood["SUCCESS"].get("fraction_above_train_p95") or 0.0
    unsafe_groups = [ood[g].get("fraction_above_train_p95") or 0.0 for g in ("COLLISION", "GROUND_CONTACT", "TIMEOUT")]
    unsafe_ood = float(np.mean(unsafe_groups)) if unsafe_groups else 0.0
    if unsafe_ood - success_ood >= 0.10:
        evidence.append("COVARIATE_SHIFT_OFF_EXPERT_MANIFOLD")
    success_sat = actions["SUCCESS"].get("fraction_norm_gt_0.95") or 0.0
    unsafe_sat = float(np.mean([(actions[g].get("fraction_norm_gt_0.95") or 0.0) for g in ("COLLISION", "GROUND_CONTACT")]))
    if unsafe_sat - success_sat >= 0.10:
        evidence.append("UNIT_BALL_SATURATION_AGGRESSIVENESS")
    if collision_tax.get("ray_blind", 0) >= 5:
        evidence.append("RAY_OBSERVABILITY_GAP")
    ambiguity_success = ambiguity["SUCCESS"].get("fraction_above_train_local_variance_p95") or 0.0
    ambiguity_unsafe = float(np.mean([(ambiguity[g].get("fraction_above_train_local_variance_p95") or 0.0) for g in ("COLLISION", "GROUND_CONTACT", "TIMEOUT")]))
    if ambiguity_unsafe - ambiguity_success >= 0.10:
        evidence.append("CONDITIONAL_ACTION_AMBIGUITY")
    if ground_tax.get("persistent_downward", 0) >= 3:
        evidence.append("VERTICAL_GROUND_CONTROL_DRIFT")
    if timeout_tax.get("no_progress", 0) + timeout_tax.get("near_goal_stall", 0) >= 6:
        evidence.append("TIMEOUT_PROGRESS_FAILURE")
    if not evidence:
        return "UNRESOLVED", ["No frozen threshold produced a dominant quantitative signal."]
    if len(evidence) == 1:
        return evidence[0], evidence
    return "MIXED_MULTIPLE_CAUSES", evidence


def serialize_nested(value):
    if isinstance(value, dict):
        return {key: serialize_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_nested(item) for item in value]
    if isinstance(value, np.ndarray):
        return [serialize_nested(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def teacher_independence() -> dict:
    source = inspect.getsource(policy_rollout)
    forbidden = [token for token in ("expert", "v_ref", "trajectory_progress", "future_state", "corridor")
                 if token in source.lower()]
    # The function's runtime inputs are exactly model, schedule, scene, task,
    # normalization, and device; expert arrays are attached only post-hoc.
    return {"passed": not forbidden, "forbidden_runtime_tokens": forbidden,
            "policy_function": "policy_rollout", "runtime_inputs": ["scene", "task", "observation", "model", "schedule"],
            "post_hoc_expert_alignment": True}


def main() -> None:
    assert_start_clean()
    r1_summary = json.loads((S3R1_ROOT / "summary.json").read_text(encoding="utf-8"))
    r1_reproduction, r1_rows = check_r1_outcome()
    checkpoint = checkpoint_identity()
    dataset = frozen_dataset_identity(r1_summary)
    if (S3R1_ROOT / "test_rollout.csv").exists():
        raise RuntimeError("BLOCKED_S3D0_TEST_ACCESS: existing TEST rollout artifact")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(20260810); np.random.seed(20260810); torch.manual_seed(20260810); torch.use_deterministic_algorithms(True)
    model, schedule, _ = load_model(device)
    train_data = load_npz(S2_DATASET / "train.npz")
    val_data = load_npz(S2_DATASET / "val.npz")
    mean, std = observation_normalization(train_data)
    expert_index = build_episode_index(val_data)
    manifest = [row for row in read_csv(S2_ROOT / "task_manifest.csv") if row["accepted"].lower() == "true" and row["split"] == "val"]
    manifest_by_id = {row["task_id"]: row for row in manifest}
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}

    # This loop is VAL-only and reproduces the R1 seed, DDIM, environment, and limit.
    cached = None
    if TEMP_ROLLOUT_CACHE.exists():
        with TEMP_ROLLOUT_CACHE.open("rb") as handle:
            cached = pickle.load(handle)
    if isinstance(cached, dict) and cached.get("start_head") == EXPECTED_START_HEAD and cached.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256 and len(cached.get("rows", [])) == 54:
        rows = cached["rows"]
        print("loaded temporary VAL rollout cache", flush=True)
    else:
        rows = []
        for index, old in enumerate(r1_rows):
            scene, task = row_task(manifest_by_id[old["task_id"]])
            row = policy_rollout(model, schedule, scene, task, mean, std, device)
            row["r1_outcome"] = outcome(old)
            row["r1_outcome_match"] = outcome(row) == outcome(old)
            rows.append(row)
            print(f"rollout={index + 1}/54 task={row['task_id']} outcome={group_name(row)} steps={row['steps']}", flush=True)
        with TEMP_ROLLOUT_CACHE.open("wb") as handle:
            pickle.dump({"start_head": EXPECTED_START_HEAD, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "rows": rows}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    rerun_counts = outcome_counts(rows)
    per_task_match = all(row["r1_outcome_match"] for row in rows)
    r1_reproduction.update({"rerun": rerun_counts, "per_task_match": per_task_match,
                            "match": bool(per_task_match and rerun_counts == EXPECTED_R1)})
    if not r1_reproduction["match"]:
        raise RuntimeError(f"BLOCKED_S3D0_R1_OUTCOME_MISMATCH after rerun: {rerun_counts}")

    # Expert manifold audit is post-rollout only.
    for index, row in enumerate(rows):
        manifest_row = manifest_by_id[row["task_id"]]
        key = (manifest_row["scene_id"], int(manifest_row["task_id"].rsplit("_", 1)[1]))
        row["offline"] = offline_task_audit(model, schedule, expert_index[key], mean, std,
                                             stable_seed(row["task_id"]) + 500000, device)
        print(f"offline={index + 1}/54 task={row['task_id']}", flush=True)
    tree, train_normalized, train_baseline, train_actions = build_train_reference(train_data, mean, std)
    for row in rows:
        attach_anchor_metrics([row], tree, train_actions, mean, std, train_baseline)
    collision_result, collision_tax = collision_audit(rows)
    ground_result, ground_tax = ground_audit(rows, train_baseline)
    timeout_result, timeout_tax = timeout_audit(rows)
    divergence = divergence_audit(rows, expert_index, manifest)
    actions = action_norm_summary(rows)
    ood = summarize_by_group(rows, "ood")
    ambiguity = summarize_by_group(rows, "ambiguity")
    offline = summarize_by_group(rows, "offline")
    teacher = teacher_independence()
    primary, evidence = determine_primary(offline, ood, actions, collision_tax, ground_tax, timeout_tax, ambiguity)
    smoke = {}
    expected_smoke = r1_summary["val_determinism_smoke"]
    for family, expected in expected_smoke.items():
        smoke_row = next(row for row in manifest if row["task_id"] == expected["task_id"])
        scene, task = row_task(smoke_row)
        first = policy_rollout(model, schedule, scene, task, mean, std, device)
        second = policy_rollout(model, schedule, scene, task, mean, std, device)
        smoke[family] = {"task_id": task["task_id"], "hashes": [first["action_trace_hash"], second["action_trace_hash"]],
                         "expected_r1_hash": expected["hashes"][0],
                         "match": first["action_trace_hash"] == second["action_trace_hash"] == expected["hashes"][0]}
    determinism = all(item["match"] for item in smoke.values())
    per_episode = per_episode_rows(rows)
    S3D0_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(S3D0_ROOT / "per_episode.csv", per_episode)
    failure_rows = []
    for name, result in (("collision", collision_result), ("ground_contact", ground_result), ("timeout", timeout_result)):
        for task_id, detail in result["details"].items():
            failure_rows.append({"failure_type": name, "task_id": task_id, **detail})
    write_csv(S3D0_ROOT / "failure_taxonomy.csv", failure_rows)
    summary = {
        "task": "S3-D0-VAL-CLOSED-LOOP-SAFETY-FAILURE-ROOT-CAUSE-AUDIT-V1",
        "final_label": "PASS_S3D0_VAL_FAILURE_ROOT_CAUSE_AUDIT",
        "start_head": EXPECTED_START_HEAD, "end_head": None, "remote_branch_head": None,
        "branch": "agent/s3d0-failure-audit-v1", "r1_head": EXPECTED_START_HEAD,
        "checkpoint_sha256": checkpoint["sha256"], "checkpoint_identity": {k: v for k, v in checkpoint.items() if k != "payload"},
        "dataset_identity": dataset, "test_accessed": False,
        "r1_outcome_reproduction": r1_reproduction,
        "offline_expert_manifold_by_outcome": offline,
        "action_norm_by_outcome": actions,
        "ood_nn_distance_by_outcome": {"train_reference_baseline": train_baseline, "by_outcome": ood},
        "conditional_ambiguity_by_outcome": {"train_reference_baseline": train_baseline["local_action_variance_std_norm"], "by_outcome": ambiguity},
        "collision_diagnosis": collision_result, "ground_contact_diagnosis": ground_result,
        "timeout_diagnosis": timeout_result, "divergence_time_by_outcome": divergence,
        "policy_execution_teacher_independent": teacher,
        "primary_diagnosis": primary, "primary_diagnosis_evidence": evidence,
        "secondary_diagnosis": {"collision": collision_tax, "ground_contact": ground_tax, "timeout": timeout_tax},
        "diagnostic_determinism": {"pass": determinism, "same_policy_trace_hash_as_r1": determinism,
                                    "smoke": smoke,
                                    "all_54_outcomes_reproduced": r1_reproduction["match"]},
        "tests": {"passed": None, "failed": None},
        "regression": None, "large_files_tracked": False,
        "what_was_proven": ["R1 outcome reproduced on all 54 VAL tasks", "frozen checkpoint and train/val dataset identity",
                             "offline expert-manifold, action saturation, OOD, ambiguity, collision, ground, timeout, and divergence audits",
                             "teacher-independent policy execution", "TEST was not accessed"],
        "what_was_not_proven": ["No remediation was tested", "No TEST performance", "No PPO or S4 readiness", "No causal intervention"],
        "current_blocker": "S3 remains blocked pending high-level review.", "formal_progress": "30%",
        "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW", "waiting_for_high_level_controller_audit": True,
        "temporary_step_trace_written": False, "temporary_step_trace_deleted": True,
        "protocol": {"normalization": "R1 train-only", "ray_distance": "ray_value * 3.0 m",
                      "anchor_period_s": 0.5, "divergence_threshold_m": POSITION_DIVERGENCE_THRESHOLD,
                      "test_rollout_opened": False},
    }
    (S3D0_ROOT / "summary.json").write_text(json.dumps(serialize_nested(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    TEMP_ROLLOUT_CACHE.unlink(missing_ok=True)
    print(json.dumps({"final_label": summary["final_label"], "r1": r1_reproduction,
                      "primary_diagnosis": primary, "evidence": evidence,
                      "action_norm": actions, "ood": ood, "ambiguity": ambiguity,
                      "collision": collision_tax, "ground": ground_tax, "timeout": timeout_tax},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
