"""34-D deployment-observable state plus deterministic local obstacle rays."""
from __future__ import annotations

import numpy as np
import pybullet as p

LOCAL_RAY_DIRECTIONS = np.asarray([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
], dtype=float)
LOCAL_RAY_DIRECTIONS /= np.linalg.norm(LOCAL_RAY_DIRECTIONS, axis=1, keepdims=True)


def obstacle_rays(position: np.ndarray, quaternion: np.ndarray, obstacle_ids: list[int], client: int,
                  max_range_m: float) -> np.ndarray:
    """Return ray hit fractions, with 0 near and 1 no obstacle in range."""
    pos = np.asarray(position, dtype=float)
    quat = np.asarray(quaternion, dtype=float)
    if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[pos, quat]).all():
        raise ValueError("ray origin and quaternion must be finite")
    rotation = np.asarray(p.getMatrixFromQuaternion(quat.tolist()), dtype=float).reshape(3, 3)
    world_dirs = LOCAL_RAY_DIRECTIONS @ rotation.T
    ray_from = np.repeat(pos[None, :], len(world_dirs), axis=0)
    ray_to = ray_from + float(max_range_m) * world_dirs
    hits = p.rayTestBatch(ray_from.tolist(), ray_to.tolist(), physicsClientId=client)
    obstacle_set = set(int(x) for x in obstacle_ids)
    values = []
    for hit in hits:
        object_id, fraction = int(hit[0]), float(hit[2])
        values.append(float(np.clip(fraction, 0.0, 1.0)) if object_id in obstacle_set else 1.0)
    result = np.asarray(values, dtype=float)
    if result.shape != (18,) or not np.isfinite(result).all():
        raise FloatingPointError("nonfinite obstacle rays")
    return result


def build_observation(state: np.ndarray, goal: np.ndarray, obstacle_ids: list[int], client: int,
                      max_range_m: float) -> np.ndarray:
    """Build only student-deployable fields; teacher/planner fields are absent."""
    state = np.asarray(state, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if state.shape[0] < 16 or goal.shape != (3,) or not np.isfinite(state[:16]).all() or not np.isfinite(goal).all():
        raise ValueError("state and goal must be finite")
    base = np.concatenate((state[0:3], goal - state[0:3], state[10:13], state[3:7], state[13:16]))
    rays = obstacle_rays(state[0:3], state[3:7], obstacle_ids, client, max_range_m)
    observation = np.concatenate((base, rays))
    if observation.shape != (34,) or not np.isfinite(observation).all():
        raise FloatingPointError("nonfinite 34-D observation")
    return observation

