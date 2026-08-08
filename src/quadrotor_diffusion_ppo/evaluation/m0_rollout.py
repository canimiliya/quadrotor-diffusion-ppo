"""Deterministic one-rollout-per-scene M0 evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from quadrotor_diffusion_ppo.envs.action import velocity_reference_to_action
from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary
from quadrotor_diffusion_ppo.expert.trajectory import GcopterTrajectory, reference_path


def _finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def run_scene(scene, project_root: Path, speed_limit: float, ray_range: float,
              goal_tolerance: float, settle_s: float, trace_root: Path) -> dict:
    trajectory_file = reference_path(project_root, scene.scene_id)
    if not trajectory_file.exists():
        raise FileNotFoundError(f"missing generated GCOPTER reference: {trajectory_file}")
    trajectory = GcopterTrajectory.from_csv(trajectory_file)
    env = ObstacleVelocityAviary(scene, speed_limit, ray_range, gui=False)
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_file = trace_root / f"{scene.scene_id}_action_trace.csv"
    fields = ["step", "time", "v_ref_x", "v_ref_y", "v_ref_z", "raw_u_x", "raw_u_y", "raw_u_z",
              "post_u_x", "post_u_y", "post_u_z", "command_dir_x", "command_dir_y", "command_dir_z",
              "command_speed_ratio", "commanded_speed"]
    rows = []
    clip_count = 0
    nonfinite_observation = 0
    nonfinite_action = 0
    nonfinite_state = 0
    obstacle_collision_count = 0
    ground_contact_count = 0
    min_goal_distance = float("inf")
    min_obstacle_clearance = float("inf")
    max_velocity_norm = 0.0
    max_commanded_speed = 0.0
    max_raw_action_component = 0.0
    max_speed_limit_excess = 0.0
    start_time = time.perf_counter()
    try:
        observation, _ = env.reset(seed=0)
        total_duration = trajectory.total_duration
        rollout_duration = total_duration + settle_s
        total_steps = int(np.ceil(rollout_duration * scene.control_hz))
        for step in range(total_steps):
            elapsed = step / scene.control_hz
            _, v_ref, _, _, _ = trajectory.evaluate(min(elapsed, total_duration))
            if not np.isfinite(v_ref).all():
                nonfinite_action += 1
                raise FloatingPointError(f"{scene.scene_id} step={step} time={elapsed} variable=v_ref")
            mapped = velocity_reference_to_action(v_ref, speed_limit)
            if mapped.clipped:
                clip_count += 1
            raw = mapped.raw_normalized
            post = mapped.post_clipping
            command = mapped.velocity_aviary_command
            if not np.isfinite(np.r_[raw, post, command]).all():
                nonfinite_action += 1
                raise FloatingPointError(f"{scene.scene_id} step={step} time={elapsed} variable=action")
            max_raw_action_component = max(max_raw_action_component, float(np.max(np.abs(raw))))
            max_speed_limit_excess = max(max_speed_limit_excess, mapped.speed_limit_excess)
            max_commanded_speed = max(max_commanded_speed, mapped.speed_ratio * speed_limit)
            rows.append({"step": step, "time": elapsed, "v_ref_x": v_ref[0], "v_ref_y": v_ref[1], "v_ref_z": v_ref[2],
                         "raw_u_x": raw[0], "raw_u_y": raw[1], "raw_u_z": raw[2], "post_u_x": post[0], "post_u_y": post[1], "post_u_z": post[2],
                         "command_dir_x": command[0], "command_dir_y": command[1], "command_dir_z": command[2],
                         "command_speed_ratio": command[3], "commanded_speed": command[3] * speed_limit})
            observation, _, _, _, _ = env.step(raw.astype(np.float32))
            state = np.asarray(env._getDroneStateVector(0), dtype=float)
            if not np.isfinite(state[:16]).all():
                nonfinite_state += 1
                raise FloatingPointError(f"{scene.scene_id} step={step} time={elapsed} variable=state")
            if not np.isfinite(observation).all():
                nonfinite_observation += 1
                raise FloatingPointError(f"{scene.scene_id} step={step} time={elapsed} variable=observation")
            goal_distance = float(np.linalg.norm(state[0:3] - scene.goal))
            min_goal_distance = min(min_goal_distance, goal_distance)
            max_velocity_norm = max(max_velocity_norm, float(np.linalg.norm(state[10:13])))
            obstacle_contacts, ground_contacts = env.contact_counts()
            obstacle_collision_count += obstacle_contacts
            ground_contact_count += ground_contacts
            clearance = env.minimum_obstacle_clearance()
            if np.isfinite(clearance):
                min_obstacle_clearance = min(min_obstacle_clearance, clearance)
        final_state = np.asarray(env._getDroneStateVector(0), dtype=float)
        final_goal_distance = float(np.linalg.norm(final_state[0:3] - scene.goal))
        reference_completed = bool(total_steps >= int(np.ceil(total_duration * scene.control_hz)))
        goal_reached = bool(min_goal_distance <= goal_tolerance)
        result = {
            "scene_id": scene.scene_id, "family": scene.family,
            "reference_completed": reference_completed, "goal_reached": goal_reached,
            "trajectory_duration": total_duration, "rollout_duration": rollout_duration,
            "sampling_count": len(rows), "control_dt": 1.0 / scene.control_hz,
            "final_goal_distance": final_goal_distance, "minimum_goal_distance": min_goal_distance,
            "obstacle_collision_count": obstacle_collision_count, "ground_contact_count": ground_contact_count,
            "minimum_obstacle_clearance": _finite_or_none(min_obstacle_clearance),
            "nonfinite_observation_count": nonfinite_observation, "nonfinite_action_count": nonfinite_action,
            "nonfinite_state_count": nonfinite_state, "expert_action_clip_count": clip_count,
            "expert_action_clip_fraction": clip_count / len(rows) if rows else 0.0,
            "max_raw_action_component": max_raw_action_component, "max_velocity_norm": max_velocity_norm,
            "max_commanded_speed": max_commanded_speed, "max_speed_limit_excess": max_speed_limit_excess,
            "v_ref_finite": True, "observation_shape": [34],
            "trace_file": str(trace_file.relative_to(project_root)),
            "pass": bool(reference_completed and goal_reached and obstacle_collision_count == 0 and ground_contact_count == 0 and
                         nonfinite_observation == 0 and nonfinite_action == 0 and nonfinite_state == 0 and
                         max_speed_limit_excess <= speed_limit * 1.0e-6),
        }
    except Exception as exc:
        result = {
            "scene_id": scene.scene_id, "family": scene.family, "reference_completed": False, "goal_reached": False,
            "trajectory_duration": trajectory.total_duration, "rollout_duration": None, "sampling_count": len(rows),
            "control_dt": 1.0 / scene.control_hz, "final_goal_distance": None, "minimum_goal_distance": _finite_or_none(min_goal_distance),
            "obstacle_collision_count": obstacle_collision_count, "ground_contact_count": ground_contact_count,
            "minimum_obstacle_clearance": _finite_or_none(min_obstacle_clearance),
            "nonfinite_observation_count": nonfinite_observation, "nonfinite_action_count": nonfinite_action + 1,
            "nonfinite_state_count": nonfinite_state, "expert_action_clip_count": clip_count,
            "expert_action_clip_fraction": clip_count / len(rows) if rows else 0.0,
            "max_raw_action_component": max_raw_action_component, "max_velocity_norm": max_velocity_norm,
            "max_commanded_speed": max_commanded_speed, "max_speed_limit_excess": max_speed_limit_excess,
            "v_ref_finite": False, "observation_shape": [34], "error": str(exc), "pass": False,
        }
    finally:
        with trace_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        env.close()
    result["rollout_wall_time_s"] = time.perf_counter() - start_time
    return result


def evaluate_m0(project_root: Path) -> dict:
    contract = json.loads((project_root / "configs" / "m0_contract.json").read_text(encoding="utf-8"))
    scenes = load_all_scenes()
    speed_limit = float(contract["speed_limit_m_per_s"])
    results = [run_scene(scene, project_root, speed_limit, float(contract["ray_max_range_m"]),
                         float(contract["goal_tolerance_m"]), float(contract["post_reference_settle_s"]),
                         project_root / "artifacts" / "m0" / "action_traces") for scene in scenes]
    summary = {
        "final_label": "PASS_GCOPTER_PIVOT_M0" if all(row["pass"] for row in results) else "BLOCKED_M0_CLOSED_LOOP_TRACKING",
        "scene_count": len(results), "reference_completed_count": sum(row["reference_completed"] for row in results),
        "goal_reached_count": sum(row["goal_reached"] for row in results),
        "obstacle_collisions_total": sum(row["obstacle_collision_count"] for row in results),
        "ground_contacts_total": sum(row["ground_contact_count"] for row in results),
        "nonfinite_events_total": sum(row["nonfinite_observation_count"] + row["nonfinite_action_count"] + row["nonfinite_state_count"] for row in results),
        "expert_action_clip_count": sum(row["expert_action_clip_count"] for row in results),
        "expert_action_clip_fraction": sum(row["expert_action_clip_count"] for row in results) / max(1, sum(row["sampling_count"] for row in results)),
        "max_raw_action_component": max(row["max_raw_action_component"] for row in results),
        "max_velocity_norm": max(row["max_velocity_norm"] for row in results),
        "max_speed_limit_excess": max(row["max_speed_limit_excess"] for row in results),
        "action_contract": {"shape": [3], "range": [-1, 1], "speed_limit_m_per_s": speed_limit},
        "observation_contract": {"base_dim": 16, "ray_dim": 18, "total_dim": 34},
        "privileged_information_leakage": False,
        "scenes": results,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    out = project_root / "artifacts" / "m0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "per_scene.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(results[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    return summary

