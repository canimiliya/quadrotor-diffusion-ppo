# Planning-Guided Diffusion Prior for Sample-Efficient Quadrotor PPO

This repository contains the approved M0 bootstrap plus the completed S2-R0
balanced GCOPTER expert-dataset pipeline: deterministic task sampling from
the same nine frozen scene topologies, GCOPTER planning, unchanged M0
closed-loop validation, task-level train/validation/test splits, and compact
NPZ export.

No Diffusion, PPO, BC training, or policy training is implemented.

Run the M0 tests and evaluation with the existing project environment:

```powershell
$py = "D:\Desktop\research_progress_management\single_quad_ppo_diffusion\.venv\m1_r2_f2\python.exe"
$env:PYTHONPATH = "src;D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones"
& $py -m pytest -q
& $py scripts/run_m0.py

# Generate/audit the frozen-scope S2 dataset (NPZ files remain gitignored)
& $py scripts/run_s2_dataset.py
& $py scripts/audit_s2_dataset.py
```

The clean GCOPTER reproduction and generated reference coefficients are
external/gitignored inputs. Their provenance is recorded in
`third_party_manifest/GCOPTER.md` and `docs/M0_CONTRACT.md`.
