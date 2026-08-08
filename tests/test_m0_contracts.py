from __future__ import annotations

import numpy as np
import pytest

from quadrotor_diffusion_ppo.envs.action import map_normalized_velocity, velocity_reference_to_action
from quadrotor_diffusion_ppo.envs.scene import load_all_scenes


def test_action_contract_zero_axes_diagonal_boundary_and_determinism():
    for action in [np.zeros(3), np.array([1., 0., 0.]), np.array([0., -1., 0.]), np.ones(3), -np.ones(3)]:
        result = map_normalized_velocity(action, 0.801)
        assert result.raw_normalized.shape == (3,)
        assert result.velocity_aviary_command.shape == (4,)
        assert np.isfinite(result.velocity_aviary_command).all()
        assert np.linalg.norm(result.post_clipping) <= 1.0 + 1e-12
        assert np.allclose(result.velocity_aviary_command, map_normalized_velocity(action, 0.801).velocity_aviary_command)
    assert map_normalized_velocity(np.zeros(3), 0.801).speed_ratio == 0.0
    assert np.allclose(velocity_reference_to_action(np.array([0.801, 0., 0.]), 0.801).post_clipping, [1., 0., 0.])


def test_action_rejects_nonfinite_and_wrong_shape():
    with pytest.raises(ValueError):
        map_normalized_velocity(np.array([np.nan, 0., 0.]), 0.801)
    with pytest.raises(ValueError):
        map_normalized_velocity(np.zeros(4), 0.801)


def test_scene_import_exactly_nine_and_geometry_finite():
    scenes = load_all_scenes()
    assert len(scenes) == 9
    assert len({scene.scene_id for scene in scenes}) == 9
    for scene in scenes:
        assert np.isfinite(scene.start).all() and np.isfinite(scene.goal).all()
        for obstacle in scene.obstacles:
            assert np.isfinite(obstacle["center"]).all() and np.isfinite(obstacle["size"]).all()
            assert np.all(np.asarray(obstacle["size"]) > 0)

