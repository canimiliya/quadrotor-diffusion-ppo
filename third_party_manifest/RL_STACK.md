# S4 RL stack

Frozen for `S4-R0-CANONICAL-MAIN-AND-PURE-PPO-BASELINE-V1`.

| component | version / value |
|---|---|
| stable-baselines3 | 2.6.0 |
| PyTorch | 2.7.1+cu128 |
| Gymnasium | 1.1.1 |
| CUDA runtime | 12.8 |
| training device | CUDA (`NVIDIA GeForce RTX 5060 Ti`) |
| PPO implementation | upstream Stable-Baselines3, not copied into this repository |

## gym-pybullet-drones

- Upstream: https://github.com/learnsyslab/gym-pybullet-drones.git
- Frozen source commit used by the formal S8-R4 route:
  `e712698a05a80728b06572819dcf044596707754`
- License: MIT.
- Install/recreate with `scripts/bootstrap_windows.ps1`, which clones the
  commit into the ignored `.deps/gym-pybullet-drones` directory and installs it
  editable with `--no-deps`; the project dependencies remain declared here.
  The `--no-deps` choice is intentional: this frozen commit's metadata asks for
  newer NumPy/SciPy/Gymnasium/SB3 versions than the formal S8-R4 stack, while
  the imported source is compatible with the pinned stack and is verified by
  the fresh-environment test.
- It is an external read-only input. No duplicate third-party source mirror is
  committed to this repository.
