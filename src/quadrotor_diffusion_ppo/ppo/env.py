"""Task-conditioned Pure PPO environment using the existing frozen flight path."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from quadrotor_diffusion_ppo.envs.scene import SceneSpec
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary

from .contract import CONTROL_HZ, GOAL_TOLERANCE_M, MAX_EPISODE_STEPS, compute_reward


@dataclass(frozen=True)
class TaskEndpoint:
    scene_id: str
    task_id: str
    candidate_seed: int
    start: np.ndarray
    goal: np.ndarray
    split: str


def task_scene(scene: SceneSpec, task: TaskEndpoint) -> SceneSpec:
    return replace(scene, start=np.asarray(task.start, dtype=float).copy(), goal=np.asarray(task.goal, dtype=float).copy())


class PurePPONavigationEnv(ObstacleVelocityAviary):
    """One frozen task episode with only 34D observation and 3D action."""

    def __init__(self, scene: SceneSpec, task: TaskEndpoint, speed_limit: float = 0.801,
                 ray_range: float = 3.0, train_tasks: tuple[TaskEndpoint, ...] = (),
                 rank: int = 0, train_mode: bool = False, gui: bool = False):
        self.base_scene = scene
        self.task = task
        self.train_tasks = tuple(train_tasks)
        self.rank = int(rank)
        self.train_mode = bool(train_mode)
        self._rng = np.random.default_rng(20260812 + self.rank)
        self.episode_steps = 0
        self.prev_distance = 0.0
        self.return_sum = 0.0
        self.action_clip_count = 0
        self.action_count = 0
        self.last_raw_action = np.zeros(3, dtype=np.float32)
        super().__init__(task_scene(scene, task), speed_limit, ray_range, gui=gui)

    def _select_train_task(self) -> TaskEndpoint:
        if not self.train_tasks:
            raise RuntimeError("TRAIN task list is empty")
        index = int(self._rng.integers(0, len(self.train_tasks)))
        return self.train_tasks[index]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.train_mode:
            if seed is not None:
                self._rng = np.random.default_rng(int(seed))
            self.task = self._select_train_task()
            self.scene = task_scene(self.base_scene, self.task)
        else:
            self.scene = task_scene(self.base_scene, self.task)
        observation, info = super().reset(seed=seed, options=options)
        self.episode_steps = 0
        self.prev_distance = float(np.linalg.norm(self._getDroneStateVector(0)[:3] - self.scene.goal))
        self.return_sum = 0.0
        self.action_clip_count = 0
        self.action_count = 0
        self.last_raw_action = np.zeros(3, dtype=np.float32)
        info = dict(info)
        info.update({"task_id": self.task.task_id, "split": self.task.split,
                     "train_task_sampling": self.train_mode})
        return observation.astype(np.float32), info

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(3)
        self.last_raw_action = raw.copy()
        if not np.isfinite(raw).all():
            raise FloatingPointError("nonfinite Pure PPO action")
        observation, _, _, _, info = super().step(raw)
        self.episode_steps += 1
        self.action_count += 1
        mapped = self.last_velocity_action
        clipped = bool(mapped.clipped) if mapped is not None else False
        self.action_clip_count += int(clipped)
        state = self._getDroneStateVector(0)
        obstacle_contacts, ground_contacts = self.contact_counts()
        collision = obstacle_contacts > 0
        ground = ground_contacts > 0
        finite = bool(np.isfinite(observation).all() and np.isfinite(state[:16]).all())
        distance = float(np.linalg.norm(state[:3] - self.scene.goal)) if finite else float("nan")
        success = bool(finite and distance <= GOAL_TOLERANCE_M and not collision and not ground)
        reward, components = compute_reward(self.prev_distance if finite else distance, distance,
                                            success=success, collision=collision, ground=ground)
        self.prev_distance = distance
        self.return_sum += reward
        timeout = bool(self.episode_steps >= MAX_EPISODE_STEPS and not success and not collision and not ground and finite)
        terminated = bool(success or collision or ground or not finite)
        truncated = bool(timeout)
        info = dict(info)
        info.update({
            "task_id": self.task.task_id, "split": self.task.split,
            "success": success, "collision": collision, "ground_contact": ground,
            "nonfinite": not finite, "timeout": timeout,
            "raw_action": raw.copy(),
            "executed_action": None if mapped is None else mapped.post_clipping.copy(),
            "action_clipped": clipped, "action_clip_count": self.action_clip_count,
            "action_count": self.action_count, "reward_components": components,
            "distance": distance, "episode_steps": self.episode_steps,
        })
        return observation.astype(np.float32), float(reward), terminated, truncated, info

    def episode_diagnostics(self) -> dict[str, Any]:
        return {"action_clip_count": int(self.action_clip_count),
                "action_count": int(self.action_count),
                "action_clip_fraction": self.action_clip_count / max(1, self.action_count)}
