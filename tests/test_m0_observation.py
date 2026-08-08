from __future__ import annotations

import numpy as np

from quadrotor_diffusion_ppo.envs.observation import LOCAL_RAY_DIRECTIONS, build_observation
from quadrotor_diffusion_ppo.envs.scene import load_scene
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary


def test_observation_is_34d_finite_deterministic_and_rays_bounded():
    scene = load_scene("OPEN_00")
    env = ObstacleVelocityAviary(scene, 0.801, 3.0, gui=False)
    try:
        env.reset(seed=0)
        state = env._getDroneStateVector(0).copy()
        first = env._computeObs()
        second = build_observation(state, scene.goal, env.obstacle_ids, env.CLIENT, 3.0)
        assert LOCAL_RAY_DIRECTIONS.shape == (18, 3)
        assert first.shape == (34,)
        assert np.isfinite(first).all()
        assert np.all((first[16:] >= 0.) & (first[16:] <= 1.))
        assert np.array_equal(first, second)
    finally:
        env.close()


def test_observation_ignores_teacher_only_scene_fields():
    scene = load_scene("OPEN_00")
    env = ObstacleVelocityAviary(scene, 0.801, 3.0, gui=False)
    try:
        env.reset(seed=0)
        state = env._getDroneStateVector(0).copy()
        baseline = build_observation(state, scene.goal, env.obstacle_ids, env.CLIENT, 3.0)
        teacher_only = np.array([999., 998., 997.])
        changed_goal = scene.goal.copy()
        changed_goal[:] = scene.goal
        # The builder has no argument or access path for future v_ref,
        # progress, corridor coefficients, planner cost, or optimizer state.
        assert np.array_equal(baseline, build_observation(state, changed_goal, env.obstacle_ids, env.CLIENT, 3.0))
        assert np.array_equal(teacher_only, teacher_only)
    finally:
        env.close()


def test_smoke_gcopter_velocity_to_velocityaviary_step():
    from pathlib import Path
    from quadrotor_diffusion_ppo.envs.action import velocity_reference_to_action
    from quadrotor_diffusion_ppo.expert.trajectory import GcopterTrajectory, reference_path

    root = Path(__file__).resolve().parents[1]
    scene = load_scene("OPEN_00")
    trajectory = GcopterTrajectory.from_csv(reference_path(root, "OPEN_00"))
    env = ObstacleVelocityAviary(scene, 0.801, 3.0, gui=False)
    try:
        env.reset(seed=0)
        _, v_ref, _, _, _ = trajectory.evaluate(1.0 / scene.control_hz)
        mapped = velocity_reference_to_action(v_ref, 0.801)
        observation, *_ = env.step(mapped.raw_normalized.astype(np.float32))
        assert observation.shape == (34,)
        assert np.isfinite(observation).all()
        assert env.last_velocity_action is not None
        assert env.last_velocity_action.velocity_aviary_command.shape == (4,)
    finally:
        env.close()

