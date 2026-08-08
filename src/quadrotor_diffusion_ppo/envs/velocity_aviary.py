"""Obstacle-aware wrapper over the standard gym-pybullet-drones VelocityAviary path."""
from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .action import VelocityAction, map_normalized_velocity
from .observation import build_observation
from .scene import SceneSpec


class ObstacleVelocityAviary(VelocityAviary):
    """CF2X environment with a shared (3,) normalized world-velocity input."""

    def __init__(self, scene: SceneSpec, speed_limit_m_per_s: float, ray_max_range_m: float,
                 gui: bool = False):
        self.scene = scene
        self.speed_limit_m_per_s = float(speed_limit_m_per_s)
        self.ray_max_range_m = float(ray_max_range_m)
        self.obstacle_ids: list[int] = []
        self.last_velocity_action: VelocityAction | None = None
        super().__init__(drone_model=DroneModel.CF2X, num_drones=1,
                         initial_xyzs=scene.start.reshape(1, 3), initial_rpys=np.zeros((1, 3)),
                         physics=Physics.PYB, pyb_freq=scene.physics_hz, ctrl_freq=scene.control_hz,
                         gui=gui, record=False, obstacles=False, user_debug_gui=False)
        self.SPEED_LIMIT = self.speed_limit_m_per_s
        self._add_scene_obstacles()

    def _actionSpace(self):
        return spaces.Box(low=-np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    def _observationSpace(self):
        low = np.full(34, -np.inf, dtype=np.float32)
        high = np.full(34, np.inf, dtype=np.float32)
        low[3:7] = -np.inf
        high[3:7] = np.inf
        low[16:] = 0.0
        high[16:] = 1.0
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _add_scene_obstacles(self) -> None:
        self.obstacle_ids = []
        for obstacle in self.scene.obstacles:
            half = (np.asarray(obstacle["size"], dtype=float) / 2.0).tolist()
            center = np.asarray(obstacle["center"], dtype=float).tolist()
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.75, 0.2, 0.2, 1.0], physicsClientId=self.CLIENT)
            body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision,
                                     baseVisualShapeIndex=visual, basePosition=center, physicsClientId=self.CLIENT)
            self.obstacle_ids.append(int(body))

    def reset(self, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._add_scene_obstacles()
        return self._computeObs(), info

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        return build_observation(state, self.scene.goal, self.obstacle_ids, self.CLIENT, self.ray_max_range_m)

    def _preprocessAction(self, action):
        mapped = map_normalized_velocity(np.asarray(action, dtype=float).reshape(-1), self.speed_limit_m_per_s)
        self.last_velocity_action = mapped
        if not np.isfinite(mapped.velocity_aviary_command).all():
            raise FloatingPointError("nonfinite mapped VelocityAviary command")
        # This is the standard VelocityAviary controller path; only the input
        # contract is changed from its legacy (direction, speed-ratio) shape.
        return super()._preprocessAction(mapped.velocity_aviary_command.reshape(1, 4))

    def contact_counts(self) -> tuple[int, int]:
        drone = int(self.DRONE_IDS[0])
        ground = len(p.getContactPoints(bodyA=drone, bodyB=self.PLANE_ID, physicsClientId=self.CLIENT))
        obstacle = sum(len(p.getContactPoints(bodyA=drone, bodyB=body, physicsClientId=self.CLIENT)) for body in self.obstacle_ids)
        return int(obstacle), int(ground)

    def minimum_obstacle_clearance(self) -> float:
        if not self.obstacle_ids:
            return float("nan")
        drone = int(self.DRONE_IDS[0])
        distances = [float(row[8]) for body in self.obstacle_ids for row in p.getClosestPoints(bodyA=drone, bodyB=body, distance=100.0, physicsClientId=self.CLIENT)]
        return min(distances) if distances else float("nan")

