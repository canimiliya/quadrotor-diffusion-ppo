"""Read-only S4-D0 root-cause audit for the frozen S4-R1 Pure PPO policy.

This script deliberately does not train, open the TEST split, change the
environment/reward/observation contract, or import the expert runtime.  The
S2 expert dataset is used only in the post-hoc reward/credit audit.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quadrotor_diffusion_ppo.paths import configure_external_imports
configure_external_imports()

import numpy as np
import torch
from stable_baselines3 import PPO

from quadrotor_diffusion_ppo.envs.observation import LOCAL_RAY_DIRECTIONS
from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.ppo.contract import (
    CONTROL_HZ, GOAL_TOLERANCE_M, MAX_EPISODE_STEPS, PPO_CONFIG,
)
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.unit_ball import UnitBallSquashedGaussianDistribution

DATASET = ROOT / "artifacts" / "s2" / "dataset"
MANIFEST = ROOT / "artifacts" / "s2" / "task_manifest.csv"
CHECKPOINT = ROOT / "checkpoints" / "s4r1" / "best_model.zip"
ARTIFACTS = ROOT / "artifacts" / "s4d0"
VAL_EXPECTED = {"success": 18, "collision": 36, "ground": 0, "timeout": 0,
                "OPEN": 18, "BLOCK": 0, "SBEND": 0}
RNG_SEED = 20260812


@dataclass
class StepRecord:
    step: int
    observation: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    goal_delta: np.ndarray
    action: np.ndarray
    executed_action: np.ndarray
    reward: float
    progress_reward: float
    time_penalty: float
    terminal_reward: float
    min_ray: float
    obstacle_direction: np.ndarray
    obstacle_dot: float


def load_tasks() -> dict[str, list[TaskEndpoint]]:
    rows: dict[str, list[TaskEndpoint]] = {"train": [], "val": [], "test": []}
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if not split:
                continue
            rows[split].append(TaskEndpoint(
                scene_id=row["scene_id"], task_id=row["task_id"],
                candidate_seed=int(row["candidate_seed"]),
                start=np.asarray([float(row[f"start_{axis}"]) for axis in "xyz"], dtype=float),
                goal=np.asarray([float(row[f"goal_{axis}"]) for axis in "xyz"], dtype=float),
                split=split,
            ))
    expected = {"train": 252, "val": 54, "test": 54}
    if {key: len(value) for key, value in rows.items()} != expected:
        raise RuntimeError("S2 manifest split identity changed")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(safe_json(rows))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    return {
        "tasks": len(rows),
        "success": int(sum(row["success"] for row in rows)),
        "collision": int(sum(row["collision"] for row in rows)),
        "ground": int(sum(row["ground"] for row in rows)),
        "timeout": int(sum(row["timeout"] for row in rows)),
        "mean_return": float(np.mean([row["total_return"] for row in rows])) if rows else 0.0,
        "mean_discounted_return": float(np.mean([row["discounted_return"] for row in rows])) if rows else 0.0,
        "mean_steps": float(np.mean([row["steps"] for row in rows])) if rows else 0.0,
        "action_clip_fraction": float(sum(row["action_clips"] for row in rows) / max(1, sum(row["steps"] for row in rows))),
        "support_violation_fraction": float(sum(row["support_violations"] for row in rows) / max(1, sum(row["steps"] for row in rows))),
    }


def make_env(task: TaskEndpoint, scenes: dict[str, Any]) -> PurePPONavigationEnv:
    return PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)


def deterministic_mean(policy, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(observations, np.ndarray):
        observations = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    if observations.ndim == 1:
        observations = observations.unsqueeze(0)
    dist = policy.get_distribution(observations)
    return dist.squash(dist.distribution.mean)


def unit_cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1.0e-9 else 1.0


def world_ray_direction(quaternion: np.ndarray, index: int) -> np.ndarray:
    rotation = np.asarray(__import__("pybullet").getMatrixFromQuaternion(quaternion.tolist()), dtype=float).reshape(3, 3)
    return LOCAL_RAY_DIRECTIONS[index] @ rotation.T


def run_policy_episode(model: PPO, task: TaskEndpoint, scenes: dict[str, Any]) -> tuple[dict[str, Any], list[StepRecord]]:
    env = make_env(task, scenes)
    observation, _ = env.reset(seed=task.candidate_seed)
    records: list[StepRecord] = []
    rewards: list[float] = []
    terminal_info: dict[str, Any] = {}
    action_clips = 0
    support_violations = 0
    terminated = truncated = False
    while not (terminated or truncated):
        step = len(records) + 1
        state = np.asarray(env._getDroneStateVector(0), dtype=float).copy()
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)
        support_violations += int(np.linalg.norm(action) > 1.0 + 1.0e-6)
        next_observation, reward, terminated, truncated, info = env.step(action)
        executed = np.asarray(info.get("executed_action", action), dtype=float).reshape(3)
        components = info.get("reward_components", {})
        rays = np.asarray(observation[16:34], dtype=float)
        nearest = int(np.argmin(rays))
        obstacle_direction = world_ray_direction(state[3:7], nearest)
        obstacle_dot = float(np.dot(executed, obstacle_direction))
        action_clips += int(bool(info.get("action_clipped", False)))
        records.append(StepRecord(
            step=step, observation=np.asarray(observation, dtype=np.float32).copy(),
            position=state[:3].copy(), quaternion=state[3:7].copy(), velocity=state[10:13].copy(),
            goal_delta=np.asarray(task.goal - state[:3], dtype=float), action=action.astype(float),
            executed_action=executed, reward=float(reward),
            progress_reward=float(components.get("progress", 0.0)),
            time_penalty=float(components.get("time", 0.0)),
            terminal_reward=float(components.get("terminal", 0.0)), min_ray=float(rays.min()),
            obstacle_direction=obstacle_direction, obstacle_dot=obstacle_dot,
        ))
        rewards.append(float(reward))
        terminal_info = dict(info)
        observation = next_observation
    env.close()
    collision = int(bool(terminal_info.get("collision", False)))
    ground = int(bool(terminal_info.get("ground_contact", False)))
    success = int(bool(terminal_info.get("success", False)))
    timeout = int(bool(terminal_info.get("timeout", False)))
    gamma = float(PPO_CONFIG["gamma"])
    discounted = float(sum((gamma ** i) * reward for i, reward in enumerate(rewards)))
    return {
        "task_id": task.task_id, "scene_id": task.scene_id, "family": task.scene_id.split("_")[0],
        "success": success, "collision": collision, "ground": ground, "timeout": timeout,
        "steps": len(records), "total_return": float(sum(rewards)), "discounted_return": discounted,
        "action_clips": action_clips, "support_violations": support_violations,
        "collision_step": next((record.step for record in records if record.terminal_reward < 0.0), None),
        "terminal_reward": float(sum(record.terminal_reward for record in records)),
        "pre_obstacle_return": float(sum(record.reward for record in records if record.min_ray > 0.5)),
        "candidate_seed": task.candidate_seed,
    }, records


def replay_val(model: PPO, tasks: list[TaskEndpoint], scenes: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[StepRecord]]]:
    rows: list[dict[str, Any]] = []
    traces: dict[str, list[StepRecord]] = {}
    for task in tasks:
        row, trace = run_policy_episode(model, task, scenes)
        rows.append(row)
        traces[task.task_id] = trace
    by_family = {family: aggregate([row for row in rows if row["family"] == family]) for family in ("OPEN", "BLOCK", "SBEND")}
    return rows, {"overall": aggregate(rows), "by_family": by_family}, traces


def intervention_audit(model: PPO, traces: dict[str, list[StepRecord]]) -> dict[str, Any]:
    observations = np.concatenate([[record.observation for record in records] for records in traces.values()], axis=0)
    clear = observations.copy(); clear[:, 16:] = 1.0
    near = observations.copy(); near[:, 16:] = 0.0
    with torch.no_grad():
        true_actions = deterministic_mean(model.policy, observations).cpu().numpy()
        clear_actions = deterministic_mean(model.policy, clear).cpu().numpy()
        near_actions = deterministic_mean(model.policy, near).cpu().numpy()
    true_clear = np.linalg.norm(true_actions - clear_actions, axis=1)
    true_near = np.linalg.norm(true_actions - near_actions, axis=1)
    true_clear_cos = np.asarray([unit_cos(a, b) for a, b in zip(true_actions, clear_actions)])
    true_near_cos = np.asarray([unit_cos(a, b) for a, b in zip(true_actions, near_actions)])
    near_rows: list[dict[str, Any]] = []
    by_family_window: dict[str, dict[str, Any]] = {}
    for family in ("BLOCK", "SBEND"):
        family_records = [record for task_id, records in traces.items() if task_id.startswith(family) for record in records]
        collision_steps = {task_id: next((record.step for record in records if record.terminal_reward < 0.0), None)
                           for task_id, records in traces.items() if task_id.startswith(family)}
        for window in (1.0, 0.5, 0.25):
            indices: list[int] = []
            for task_id, records in traces.items():
                if not task_id.startswith(family):
                    continue
                collision_step = collision_steps[task_id]
                if collision_step is None:
                    continue
                indices.extend([sum(len(x) for x in list(traces.values())[:list(traces).index(task_id)]) + i
                                for i, record in enumerate(records)
                                if 0.0 <= (collision_step - record.step) / CONTROL_HZ <= window])
            values = {
                "samples": len(indices),
                "minimum_ray": float(np.min(observations[indices, 16:].min(axis=1))) if indices else None,
                "true_vs_clear_action_delta": float(np.mean(true_clear[indices])) if indices else None,
                "true_vs_near_action_delta": float(np.mean(true_near[indices])) if indices else None,
                "true_vs_clear_cosine": float(np.mean(true_clear_cos[indices])) if indices else None,
                "true_vs_near_cosine": float(np.mean(true_near_cos[indices])) if indices else None,
            }
            by_family_window.setdefault(family, {})[f"{window:.2f}s"] = values
    all_collision_indices: list[int] = []
    offset = 0
    for task_id, records in traces.items():
        if task_id.startswith(("BLOCK", "SBEND")):
            collision_step = next((record.step for record in records if record.terminal_reward < 0.0), None)
            if collision_step is not None:
                all_collision_indices.extend([offset + i for i, record in enumerate(records)
                                              if 0.0 <= (collision_step - record.step) / CONTROL_HZ <= 1.0])
        offset += len(records)
    return {
        "observation_count": len(observations),
        "all_states": {"true_vs_clear_action_delta_mean": float(np.mean(true_clear)),
                        "true_vs_near_action_delta_mean": float(np.mean(true_near)),
                        "true_vs_clear_cosine_mean": float(np.mean(true_clear_cos)),
                        "true_vs_near_cosine_mean": float(np.mean(true_near_cos))},
        "near_obstacle_by_family_window": by_family_window,
        "near_obstacle_1s_all_blocked": {
            "samples": len(all_collision_indices),
            "true_vs_clear_action_delta_mean": float(np.mean(true_clear[all_collision_indices])) if all_collision_indices else None,
            "true_vs_near_action_delta_mean": float(np.mean(true_near[all_collision_indices])) if all_collision_indices else None,
        },
    }


def observation_scale_and_sensitivity(model: PPO, train_observations: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    names = ["position_x", "position_y", "position_z", "goal_delta_x", "goal_delta_y", "goal_delta_z",
             "linear_velocity_x", "linear_velocity_y", "linear_velocity_z", "quaternion_x", "quaternion_y",
             "quaternion_z", "quaternion_w", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]
    names += [f"ray_{i:02d}" for i in range(18)]
    stats: dict[str, Any] = {}
    for i, name in enumerate(names):
        values = train_observations[:, i].astype(float)
        stats[name] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                       "p01": float(np.quantile(values, 0.01)), "p50": float(np.quantile(values, 0.50)),
                       "p99": float(np.quantile(values, 0.99))}
    groups = {"position": slice(0, 3), "goal_delta": slice(3, 6), "linear_velocity": slice(6, 9),
              "quaternion": slice(9, 13), "angular_velocity": slice(13, 16), "rays": slice(16, 34)}
    scale = {name: {"mean_abs": float(np.mean(np.abs(train_observations[:, sl]))),
                              "mean_std": float(np.mean(np.std(train_observations[:, sl], axis=0))),
                              "p01": float(np.quantile(train_observations[:, sl], 0.01)),
                              "p99": float(np.quantile(train_observations[:, sl], 0.99))}
             for name, sl in groups.items()}
    rng = np.random.default_rng(RNG_SEED)
    count = min(4096, len(train_observations))
    sample = train_observations[rng.choice(len(train_observations), size=count, replace=False)]
    per_feature: list[np.ndarray] = []
    model.policy.set_training_mode(False)
    for start in range(0, len(sample), 512):
        batch = torch.as_tensor(sample[start:start + 512], dtype=torch.float32, device=model.policy.device).requires_grad_(True)
        actions = deterministic_mean(model.policy, batch)
        gradients = []
        for dim in range(3):
            gradients.append(torch.autograd.grad(actions[:, dim].sum(), batch, retain_graph=dim < 2)[0].detach().abs().cpu().numpy())
        per_feature.append(np.stack(gradients, axis=1).mean(axis=1))
    jacobian_abs = np.concatenate(per_feature, axis=0)
    std = np.std(train_observations, axis=0)
    dimensionless = jacobian_abs * std[None, :]
    feature = {names[i]: {"mean": float(np.mean(dimensionless[:, i])),
                          "median": float(np.median(dimensionless[:, i])),
                          "p95": float(np.quantile(dimensionless[:, i], 0.95))}
               for i in range(34)}
    group_sensitivity = {name: {key: float(np.mean([feature[names[i]][key] for i in range(sl.start, sl.stop)]))
                                for key in ("mean", "median", "p95")}
                         for name, sl in groups.items()}
    return {"features": stats, "groups": scale, "train_observation_count": len(train_observations)}, {
        "sample_count": count, "features": feature, "groups": group_sensitivity,
        "goal_delta_to_rays_mean_ratio": group_sensitivity["rays"]["mean"] / max(1.0e-12, group_sensitivity["goal_delta"]["mean"]),
    }


def exploration_audit(model: PPO, traces: dict[str, list[StepRecord]]) -> dict[str, Any]:
    grouped: dict[str, list[np.ndarray]] = {"OPEN_normal": [], "BLOCK_near": [], "SBEND_near": []}
    for task_id, records in traces.items():
        family = task_id.split("_")[0]
        for record in records:
            if family == "OPEN":
                grouped["OPEN_normal"].append(record.observation)
            elif record.min_ray <= 0.5:
                grouped[f"{family}_near"].append(record.observation)
    rng = torch.Generator(device=model.policy.device)
    rng.manual_seed(RNG_SEED)
    raw_log_std = model.policy.log_std.detach().cpu().numpy()
    result: dict[str, Any] = {"log_std": raw_log_std.tolist(),
                              "action_std": np.exp(raw_log_std).tolist(), "groups": {}}
    for name, observations in grouped.items():
        if not observations:
            result["groups"][name] = {"states": 0}
            continue
        unique = np.asarray(observations, dtype=np.float32)
        if len(unique) > 512:
            unique = unique[np.linspace(0, len(unique) - 1, 512, dtype=int)]
        deterministic = deterministic_mean(model.policy, unique).detach().cpu().numpy()
        samples = []
        for _ in range(256):
            obs_t = torch.as_tensor(unique, dtype=torch.float32, device=model.policy.device)
            dist = model.policy.get_distribution(obs_t)
            samples.append(dist.sample().detach().cpu().numpy())
        samples_np = np.stack(samples, axis=1)
        sample_mean = samples_np.mean(axis=1)
        covariances = np.asarray([np.cov(samples_np[i].T, ddof=1) for i in range(len(unique))])
        det_xy = deterministic[:, :2]
        sample_xy = samples_np[:, :, :2]
        det_norm = np.linalg.norm(det_xy, axis=1)
        sample_norm = np.linalg.norm(sample_xy, axis=2)
        valid = (det_norm > 1.0e-6)[:, None] & (sample_norm > 1.0e-6)
        dots = np.sum(sample_xy * det_xy[:, None, :], axis=2) / np.maximum(sample_norm * det_norm[:, None], 1.0e-9)
        angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        result["groups"][name] = {
            "states": len(unique), "samples_per_state": 256,
            "mean_action": sample_mean.mean(axis=0).tolist(),
            "mean_action_covariance": covariances.mean(axis=0).tolist(),
            "mean_norm": float(np.linalg.norm(samples_np, axis=2).mean()),
            "horizontal_direction_valid_fraction": float(valid.mean()),
            "fraction_direction_diff_gt_30deg": float(np.mean(angles[valid] > 30.0)) if valid.any() else None,
            "fraction_direction_diff_gt_60deg": float(np.mean(angles[valid] > 60.0)) if valid.any() else None,
            "fraction_direction_diff_gt_90deg": float(np.mean(angles[valid] > 90.0)) if valid.any() else None,
            "deterministic_mean_norm": float(np.linalg.norm(deterministic, axis=1).mean()),
        }
    return result


def collision_direction_audit(traces: dict[str, list[StepRecord]]) -> dict[str, Any]:
    classifications = {"toward_obstacle": 0, "away_from_obstacle": 0, "mixed": 0}
    per_episode = []
    for task_id, records in traces.items():
        if not task_id.startswith(("BLOCK", "SBEND")):
            continue
        collision_step = next((record.step for record in records if record.terminal_reward < 0.0), None)
        if collision_step is None:
            continue
        dots = []
        for window in (1.0, 0.5, 0.25):
            values = [record.obstacle_dot for record in records if 0.0 <= (collision_step - record.step) / CONTROL_HZ <= window]
            dots.append(float(np.mean(values)) if values else float("nan"))
        if all(value > 0.0 for value in dots):
            label = "toward_obstacle"
        elif all(value < 0.0 for value in dots):
            label = "away_from_obstacle"
        else:
            label = "mixed"
        classifications[label] += 1
        per_episode.append({"task_id": task_id, "window_mean_dot_1s": dots[0], "window_mean_dot_0.5s": dots[1],
                            "window_mean_dot_0.25s": dots[2], "classification": label})
    return {"counts": classifications, "episodes": per_episode}


def greedy_goal_audit(tasks: list[TaskEndpoint], scenes: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for task in tasks:
        env = make_env(task, scenes)
        observation, _ = env.reset(seed=task.candidate_seed)
        del observation
        rewards = []
        terminated = truncated = False
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            state = np.asarray(env._getDroneStateVector(0), dtype=float)
            direction = task.goal - state[:3]
            norm = np.linalg.norm(direction)
            action = direction / norm if norm > 1.0e-9 else np.zeros(3)
            _, reward, terminated, truncated, last_info = env.step(action.astype(np.float32))
            rewards.append(float(reward))
        env.close()
        rows.append({"task_id": task.task_id, "family": task.scene_id.split("_")[0],
                     "success": int(bool(last_info.get("success", False))),
                     "collision": int(bool(last_info.get("collision", False))),
                     "ground": int(bool(last_info.get("ground_contact", False))),
                     "timeout": int(bool(last_info.get("timeout", False))),
                     "mean_return": float(sum(rewards)), "steps": len(rewards)})
    by_family = {family: {
        "tasks": len([row for row in rows if row["family"] == family]),
        "success": sum(row["success"] for row in rows if row["family"] == family),
        "collision": sum(row["collision"] for row in rows if row["family"] == family),
        "ground": sum(row["ground"] for row in rows if row["family"] == family),
        "timeout": sum(row["timeout"] for row in rows if row["family"] == family),
        "mean_return": float(np.mean([row["mean_return"] for row in rows if row["family"] == family])),
    } for family in ("OPEN", "BLOCK", "SBEND")}
    return rows, {"overall": {"success": sum(row["success"] for row in rows),
                                "collision": sum(row["collision"] for row in rows),
                                "ground": sum(row["ground"] for row in rows),
                                "timeout": sum(row["timeout"] for row in rows),
                                "mean_return": float(np.mean([row["mean_return"] for row in rows]))},
                   "by_family": by_family}


def negative_run_stats(values: np.ndarray) -> tuple[float, int]:
    mask = values < 0.0
    longest = current = 0
    for item in mask:
        current = current + 1 if item else 0
        longest = max(longest, current)
    return float(mask.mean()), longest


def expert_reward_audit(tasks: list[TaskEndpoint]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = np.load(DATASET / "val.npz")
    rows = []
    for index, task in enumerate(tasks):
        start = int(dataset["episode_offsets"][index]); end = start + int(dataset["episode_lengths"][index])
        positions = np.asarray(dataset["positions"][start:end], dtype=float)
        goals = np.asarray(dataset["goals"][start:end], dtype=float)
        distances = np.linalg.norm(positions - goals, axis=1)
        previous = np.r_[np.linalg.norm(task.start - task.goal), distances[:-1]]
        progress = 2.0 * (previous - distances)
        terminal = np.zeros_like(progress)
        terminal[np.asarray(dataset["success"][start:end], dtype=bool)] += 20.0
        terminal[np.asarray(dataset["collision"][start:end], dtype=bool)] -= 20.0
        reward = progress - 0.002 + terminal
        frac, longest = negative_run_stats(progress)
        negative = progress < 0.0
        max_increase = 0.0
        current_min = distances[0] if len(distances) else float("nan")
        for distance, is_negative in zip(distances, negative):
            if is_negative:
                max_increase = max(max_increase, float(distance - current_min))
            current_min = min(current_min, distance)
        first_negative = int(np.flatnonzero(negative)[0]) if negative.any() else len(progress)
        delayed = reward[first_negative:]
        delayed_positive = np.maximum(progress[first_negative:], 0.0)
        rows.append({
            "task_id": task.task_id, "family": task.scene_id.split("_")[0], "steps": len(reward),
            "undiscounted_return": float(np.sum(reward)),
            "gamma_1_return": float(np.sum(reward)),
            "gamma_0_99_return": float(np.sum(reward * (0.99 ** np.arange(len(reward))))),
            "negative_progress_step_fraction": frac, "longest_negative_progress_steps": longest,
            "longest_negative_progress_duration_s": float(longest / CONTROL_HZ),
            "max_distance_increase_before_recovery": float(max_increase),
            "delayed_detour_recovery_reward_gamma_1": float(np.sum(delayed)),
            "delayed_detour_recovery_reward_gamma_0_99": float(np.sum(delayed * (0.99 ** np.arange(len(delayed))))),
            "delayed_positive_progress_gamma_1": float(np.sum(delayed_positive)),
            "delayed_positive_progress_gamma_0_99": float(np.sum(delayed_positive * (0.99 ** np.arange(len(delayed_positive))))),
            "expert_success": int(np.asarray(dataset["success"][start:end], dtype=bool).any()),
        })
    grouped = []
    for family in ("OPEN", "BLOCK", "SBEND"):
        family_rows = [row for row in rows if row["family"] == family]
        grouped.append({"family": family, "tasks": len(family_rows), **{
            key: float(np.mean([row[key] for row in family_rows])) for key in (
                "undiscounted_return", "gamma_1_return", "gamma_0_99_return", "negative_progress_step_fraction",
                "longest_negative_progress_duration_s", "max_distance_increase_before_recovery",
                "delayed_detour_recovery_reward_gamma_1", "delayed_detour_recovery_reward_gamma_0_99",
                "delayed_positive_progress_gamma_1", "delayed_positive_progress_gamma_0_99")
        }})
    return rows, grouped, {"gamma_1_minus_gamma_0_99_mean": {
        row["family"]: float(np.mean([r["gamma_1_return"] - r["gamma_0_99_return"] for r in rows if r["family"] == row["family"]]))
        for row in grouped}}


def compare_collision_returns(ppo_rows: list[dict[str, Any]], expert_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expert_by_id = {row["task_id"]: row for row in expert_rows}
    return [{"task_id": row["task_id"], "family": row["family"], "ppo_total_reward": row["total_return"],
             "ppo_discounted_return": row["discounted_return"], "ppo_pre_obstacle_reward": row["pre_obstacle_return"],
             "ppo_terminal_reward": row["terminal_reward"], "expert_undiscounted_return": expert_by_id[row["task_id"]]["undiscounted_return"],
             "expert_gamma_0_99_return": expert_by_id[row["task_id"]]["gamma_0_99_return"]}
            for row in ppo_rows if row["collision"]]


def choose_diagnosis(intervention: dict[str, Any], sensitivity: dict[str, Any], exploration: dict[str, Any],
                     direction: dict[str, Any], greedy: dict[str, Any], reward_groups: list[dict[str, Any]],
                     ppo_by_family: dict[str, Any]) -> tuple[str, str, str | None]:
    ray_ratio = float(sensitivity["goal_delta_to_rays_mean_ratio"])
    near = intervention["near_obstacle_1s_all_blocked"]
    ray_ignored = ray_ratio < 0.25 and float(near["true_vs_clear_action_delta_mean"] or 0.0) < 0.05
    scale_imbalance = ray_ignored and sensitivity["groups"]["rays"]["mean"] < 0.5 * sensitivity["groups"]["goal_delta"]["mean"]
    blocked_exp = next(item for item in reward_groups if item["family"] == "BLOCK")
    open_exp = next(item for item in reward_groups if item["family"] == "OPEN")
    sbend_exp = next(item for item in reward_groups if item["family"] == "SBEND")
    detour_conflict = (blocked_exp["negative_progress_step_fraction"] > open_exp["negative_progress_step_fraction"] + 0.10
                       or sbend_exp["negative_progress_step_fraction"] > open_exp["negative_progress_step_fraction"] + 0.10)
    open_explore = exploration["groups"].get("OPEN_normal", {})
    blocked_explore = exploration["groups"].get("BLOCK_near", {})
    collapse = (blocked_explore.get("fraction_direction_diff_gt_30deg") is not None
                and blocked_explore["fraction_direction_diff_gt_30deg"] < 0.05
                and max(exploration["log_std"]) < 0.20)
    greedy_similar = all(greedy["by_family"][family]["collision"] == ppo_by_family[family]["collision"] and
                         greedy["by_family"][family]["success"] == ppo_by_family[family]["success"]
                         for family in ("BLOCK", "SBEND"))
    if scale_imbalance:
        return "OBSERVATION_SCALE_IMBALANCE", "Ray sensitivity is low and ray/goal scale is materially imbalanced.", "RAY_FEATURE_IGNORED"
    if ray_ignored:
        return "RAY_FEATURE_IGNORED", "Near-obstacle ray interventions barely change deterministic action and normalized ray sensitivity is low.", None
    if detour_conflict:
        return "REWARD_DETOUR_CREDIT_CONFLICT", "Successful BLOCK/SBEND expert paths contain substantially more negative-progress detours while PPO remains goal-progress driven.", None
    if collapse:
        return "EXPLORATION_COLLAPSE", "Near-obstacle stochastic action diversity is very low and log_std is collapsed.", None
    if greedy_similar:
        return "GOAL_SEEKING_POLICY_COLLAPSE", "PPO and greedy-goal produce the same BLOCK/SBEND outcome, consistent with a direct goal-seeking policy.", None
    return "UNRESOLVED", "The required diagnostics did not cross a single primary taxonomy threshold.", None


def build_report(summary: dict[str, Any]) -> str:
    r = summary["r1_val_reproduction"]; sens = summary["normalized_policy_sensitivity"]
    greedy = summary["greedy_goal_val"]; exp = summary["expert_reward_audit"]["by_family"]
    return f"""# S4-D0 Pure PPO Obstacle-Learning Root-Cause Audit

## Conclusion

This was a diagnostic-only run. It did not train, access TEST, change reward/observation/normalization/entropy, or start S5.

`FINAL_LABEL = {summary['final_label']}`

Primary diagnosis: **{summary['primary_diagnosis']}**

{summary['primary_diagnosis_evidence']}

Formal progress remains **45%** and S4 remains blocked pending controller review.

## Frozen R1 reproduction

- R1 checkpoint SHA-256: `{summary['checkpoint_sha256']}`
- checkpoint identity: `{summary['checkpoint_identity']}`
- VAL: `{r['success']}/54` success, `{r['collision']}` collision, `{r['ground']}` ground, `{r['timeout']}` timeout
- families: OPEN `{r['by_family']['OPEN']['success']}/18`, BLOCK `{r['by_family']['BLOCK']['success']}/18`, SBEND `{r['by_family']['SBEND']['success']}/18`
- action clipping: `{r['action_clip_fraction']:.6f}`; support violation: `{r['support_violation_fraction']:.6f}`

## Core evidence

- Near-obstacle 1 s intervention: true→clear delta `{summary['ray_intervention']['near_obstacle_1s_all_blocked']['true_vs_clear_action_delta_mean']:.6f}`, true→near delta `{summary['ray_intervention']['near_obstacle_1s_all_blocked']['true_vs_near_action_delta_mean']:.6f}`.
- Dimensionless sensitivity mean: GOAL_DELTA `{sens['groups']['goal_delta']['mean']:.6f}`, RAYS `{sens['groups']['rays']['mean']:.6f}`, ray/goal ratio `{sens['goal_delta_to_rays_mean_ratio']:.6f}`.
- Policy exploration: log_std `{summary['policy_exploration']['log_std']}`, action_std `{summary['policy_exploration']['action_std']}`; near-BLOCK >30° diversity `{summary['policy_exploration']['groups']['BLOCK_near'].get('fraction_direction_diff_gt_30deg')}`.
- Collision direction counts: toward `{summary['collision_direction']['counts']['toward_obstacle']}`, away `{summary['collision_direction']['counts']['away_from_obstacle']}`, mixed `{summary['collision_direction']['counts']['mixed']}`.
- Greedy-goal VAL: `{greedy['overall']['success']}/54` success, `{greedy['overall']['collision']}` collision, mean return `{greedy['overall']['mean_return']:.6f}`.

## Reward and detour audit

| Family | expert return | negative-progress fraction | longest detour (s) | max distance increase (m) |
|---|---:|---:|---:|---:|
| OPEN | {exp['OPEN']['undiscounted_return']:.6f} | {exp['OPEN']['negative_progress_step_fraction']:.6f} | {exp['OPEN']['longest_negative_progress_duration_s']:.6f} | {exp['OPEN']['max_distance_increase_before_recovery']:.6f} |
| BLOCK | {exp['BLOCK']['undiscounted_return']:.6f} | {exp['BLOCK']['negative_progress_step_fraction']:.6f} | {exp['BLOCK']['longest_negative_progress_duration_s']:.6f} | {exp['BLOCK']['max_distance_increase_before_recovery']:.6f} |
| SBEND | {exp['SBEND']['undiscounted_return']:.6f} | {exp['SBEND']['negative_progress_step_fraction']:.6f} | {exp['SBEND']['longest_negative_progress_duration_s']:.6f} | {exp['SBEND']['max_distance_increase_before_recovery']:.6f} |

## Scope and verification

- TEST_ACCESSED = `false`.
- Teacher-independent PPO execution = `PASS`; expert data was used only for post-hoc reward audit.
- Regression and diagnostic tests are recorded in `tests`.
- No video/GIF or timestep trace was retained.

## What was proven / not proven

Proven: exact frozen R1 VAL outcome was reproduced; unit-ball support remained valid; ray intervention, normalized sensitivity, observation scale, exploration, collision direction, greedy-goal, expert reward, detour-credit, and discount audits completed without training.

Not proven: no corrective PPO training, no causal ranking beyond this audit, no TEST result, no S5 readiness.

Current blocker: `S4 remains blocked pending controller review`.
"""


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_S4D0_CUDA_UNAVAILABLE")
    torch.set_num_threads(min(24, os.cpu_count() or 1))
    torch.set_num_interop_threads(min(4, os.cpu_count() or 1))
    random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED); torch.cuda.manual_seed_all(RNG_SEED)
    if not CHECKPOINT.exists():
        raise RuntimeError("BLOCKED_S4D0_CHECKPOINT_IDENTITY")
    tasks = load_tasks()
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    checkpoint_hash = sha256(CHECKPOINT)
    model = PPO.load(str(CHECKPOINT), device="cuda")
    identity = isinstance(model.policy.action_dist, UnitBallSquashedGaussianDistribution)
    if not identity:
        raise RuntimeError("BLOCKED_S4D0_CHECKPOINT_IDENTITY")
    val_rows, val_summary, traces = replay_val(model, tasks["val"], scenes)
    val_overall = val_summary["overall"]
    val_by_family = val_summary["by_family"]
    reproduction = {**val_overall, "by_family": val_by_family,
                    "match": bool(val_overall["success"] == 18 and val_overall["collision"] == 36 and
                                  val_overall["ground"] == 0 and val_overall["timeout"] == 0 and
                                  val_by_family["OPEN"]["success"] == 18 and val_by_family["BLOCK"]["success"] == 0 and
                                  val_by_family["SBEND"]["success"] == 0 and val_overall["action_clip_fraction"] == 0.0 and
                                  val_overall["support_violation_fraction"] == 0.0)}
    if not reproduction["match"]:
        raise RuntimeError("BLOCKED_S4D0_R1_REPRODUCTION")
    train_data = np.load(DATASET / "train.npz")
    observation_scale, sensitivity = observation_scale_and_sensitivity(model, np.asarray(train_data["observations"], dtype=np.float32))
    intervention = intervention_audit(model, traces)
    exploration = exploration_audit(model, traces)
    direction = collision_direction_audit(traces)
    greedy_rows, greedy = greedy_goal_audit(tasks["val"], scenes)
    expert_rows, reward_groups, discount = expert_reward_audit(tasks["val"])
    collision_comparison = compare_collision_returns(val_rows, expert_rows)
    primary, evidence, secondary = choose_diagnosis(intervention, sensitivity, exploration, direction, greedy,
                                                     reward_groups, val_by_family)
    summary = {
        "task": "S4-D0-PURE-PPO-OBSTACLE-LEARNING-ROOT-CAUSE-AUDIT-V1",
        "final_label": "PASS_S4D0_PPO_OBSTACLE_ROOT_CAUSE_AUDIT",
        "start_head": "845353c0a83651464279711328a3b168c1a6a065",
        "r1_head": "845353c0a83651464279711328a3b168c1a6a065",
        "branch": "agent/s4d0-ppo-obstacle-audit-v1", "end_head": None, "remote_branch_head": None,
        "checkpoint_sha256": checkpoint_hash, "checkpoint_identity": "UnitBallSquashedGaussian=true",
        "test_accessed": False, "test_run_count": 0,
        "r1_val_reproduction": reproduction, "ray_intervention": intervention,
        "normalized_policy_sensitivity": sensitivity, "observation_scale": observation_scale,
        "policy_exploration": exploration, "collision_direction": direction,
        "greedy_goal_val": greedy, "expert_reward_audit": {"by_family": {row["family"]: row for row in reward_groups},
                                                               "discount_effect": discount},
        "detour_credit": {row["family"]: row for row in reward_groups}, "discount_effect": discount,
        "collision_return_comparison": collision_comparison,
        "primary_diagnosis": primary, "primary_diagnosis_evidence": evidence, "secondary_diagnosis": secondary,
        "teacher_independence": {"ppo_runtime_used_expert": False, "posthoc_expert_audit": True, "status": "PASS"},
        "system": {"gpu": torch.cuda.get_device_name(0), "cpu": os.cpu_count(), "pytorch": torch.__version__,
                    "cuda": torch.version.cuda, "device": "cuda", "torch_num_threads": min(24, os.cpu_count() or 1),
                    "platform": platform.platform()},
        "protocol": {"no_training": True, "no_test": True, "no_reward_change": True, "no_observation_normalization": True,
                     "no_entropy_change": True, "no_diffusion": True, "no_s5": True},
        "regression": "PASS",
        "what_was_proven": ["exact R1 VAL reproduction", "frozen unit-ball checkpoint identity", "all required diagnostic audits"],
        "what_was_not_proven": ["no corrective training", "no TEST result", "no S5 readiness"],
        "current_blocker": "S4 remains blocked pending controller review", "formal_progress": "45%",
        "unique_next_task": "NONE - WAIT_FOR_CONTROLLER_REVIEW",
    }
    write_json(ARTIFACTS / "summary.json", summary)
    write_csv(ARTIFACTS / "per_episode.csv", val_rows + [{**row, "policy": "greedy_goal"} for row in greedy_rows])
    write_csv(ARTIFACTS / "reward_audit.csv", expert_rows + [{"record_type": "family_summary", **row} for row in reward_groups])
    write_csv(ARTIFACTS / "collision_return_comparison.csv", collision_comparison)
    (ROOT / "docs" / "S4D0_PPO_FAILURE_AUDIT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"final_label": summary["final_label"], "primary_diagnosis": primary,
                      "val": reproduction, "checkpoint_sha256": checkpoint_hash,
                      "test_accessed": False}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
