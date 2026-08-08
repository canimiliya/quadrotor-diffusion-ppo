# S4-R1 Report

This report records the isolated `S4-R1-UNIT-BALL-PPO-ACTION-SUPPORT-REPAIR-V1`
milestone. S4-R0 evidence remains under `artifacts/s4/` and is not overwritten.

## Status

```text
FINAL_LABEL = PENDING_TRAINING
FORMAL_STAGE = S4
FORMAL_PROGRESS = 45%
S5 = FROZEN
```

The implementation replaces only the SB3 action distribution with
`UnitBallSquashedGaussian`. Reward, observation, scene, split, PPO optimizer
configuration, seed, N_ENVS=8, and the exact 500,000 environment-step budget
are frozen. The CUDA run uses the `smd-blackwell` environment, RTX 5060 Ti,
and 24 PyTorch CPU threads with eight PyBullet workers.

Final training and audit values are written to
`artifacts/s4r1/summary.json`, `artifacts/s4r1/learning_curve.csv`, and
`artifacts/s4r1/val_best_rollout.csv`. `test_rollout.csv` is created only when
the VAL gate passes. The independent protocol result is written to
`artifacts/s4r1/independent_verification.json`.
