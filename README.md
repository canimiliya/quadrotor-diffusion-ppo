# Planning-Guided Diffusion Prior for Sample-Efficient Quadrotor PPO

This repository currently contains only the approved M0 bootstrap: a frozen
GCOPTER polynomial-velocity bridge, a shared normalized 3-D velocity action
contract, a 34-D obstacle-ray observation candidate, and a deterministic
9-scene CF2X PyBullet sanity rollout.

No Diffusion, PPO, BC, dataset generation, or training code is implemented.

Run the M0 tests and evaluation with the existing project environment:

```powershell
$py = "D:\Desktop\research_progress_management\single_quad_ppo_diffusion\.venv\m1_r2_f2\python.exe"
$env:PYTHONPATH = "src;D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones"
& $py -m pytest -q
& $py scripts/run_m0.py
```

The clean GCOPTER reproduction and generated reference coefficients are
external/gitignored inputs. Their provenance is recorded in
`third_party_manifest/GCOPTER.md` and `docs/M0_CONTRACT.md`.

