"""Fast, non-training installation smoke test for this repository."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quadrotor_diffusion_ppo.paths import configure_external_imports

configure_external_imports()


def main() -> int:
    import gymnasium
    import gym_pybullet_drones
    import numpy as np
    import pybullet
    import torch
    import quadrotor_diffusion_ppo
    from quadrotor_diffusion_ppo.envs.action import map_normalized_velocity
    from quadrotor_diffusion_ppo.envs.observation import LOCAL_RAY_DIRECTIONS
    from quadrotor_diffusion_ppo.ppo.contract import ACTION_DIM, OBSERVATION_DIM
    from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
    from quadrotor_diffusion_ppo.diffusion.unet1d import ConditionalUnet1D
    from quadrotor_diffusion_ppo.bc.model import MatchedSequenceBC
    from quadrotor_diffusion_ppo.ppo import unit_ball
    import stable_baselines3

    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda or "none",
        "cuda_visible": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "pybullet": getattr(pybullet, "__version__", "imported"),
        "gymnasium": gymnasium.__version__,
        "gym_pybullet_drones": str(getattr(gym_pybullet_drones, "__file__", "imported")),
        "stable_baselines3": stable_baselines3.__version__,
        "project_package": str(getattr(quadrotor_diffusion_ppo, "__file__", "imported")),
        "observation_dim": int(OBSERVATION_DIM),
        "action_dim": int(ACTION_DIM),
        "ray_direction_count": int(LOCAL_RAY_DIRECTIONS.shape[0]),
    }
    assert OBSERVATION_DIM == 34
    assert ACTION_DIM == 3
    assert LOCAL_RAY_DIRECTIONS.shape == (18, 3)
    assert map_normalized_velocity(np.zeros(3), 0.801).velocity_aviary_command.shape == (4,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        observation = torch.zeros(1, OBSERVATION_DIM, device=device)
        actions = torch.zeros(1, 16, ACTION_DIM, device=device)
        timesteps = torch.zeros(1, dtype=torch.long, device=device)
        mlp = ConditionalDiffusionMLP().to(device).eval()
        mlp_output = mlp(actions, observation, timesteps)
        unet = ConditionalUnet1D().to(device).eval()
        unet_output = unet(actions, observation, timesteps)
        bc = MatchedSequenceBC().to(device).eval()
        bc_output = bc(observation)
    assert mlp_output.shape == unet_output.shape == bc_output.shape == (1, 16, 3)
    checks["diffusion_mlp_forward"] = list(mlp_output.shape)
    checks["temporal_unet_forward"] = list(unet_output.shape)
    checks["bc_forward"] = list(bc_output.shape)
    checks["ppo_module_import"] = unit_ball.__name__

    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
