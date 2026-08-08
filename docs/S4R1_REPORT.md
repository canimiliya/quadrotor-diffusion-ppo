# S4-R1 Report

This report records the isolated `S4-R1-UNIT-BALL-PPO-ACTION-SUPPORT-REPAIR-V1`
milestone. S4-R0 evidence remains under `artifacts/s4/` and is not overwritten.

## Status

```text
FINAL_LABEL = BLOCKED_S4R1_PPO_NO_OBSTACLE_LEARNING
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

## Final result

The exact CUDA run completed 500,000 environment steps in 493.98 s on the RTX
5060 Ti with eight environments and 24 CPU threads. The best VAL checkpoint was
at 400,000 steps: 18/54 success, 36 collisions, zero ground contacts, zero
timeouts, zero nonfinite episodes, and mean return -1.8289. Family results were
OPEN 18/18, BLOCK 0/18, and SBEND 0/18.

The action-support repair passed: policy support violation was 0/9,569
evaluated VAL actions and environment action clipping was 0/9,569 actions. R0's clipping
was approximately 99.872%, so the distribution/environment mismatch is fixed.
The VAL usability gate nevertheless failed because BLOCK and SBEND had no
success. TEST was not executed (`PPO_R1_TEST_RUN_COUNT=0`) as required. No
reward, observation, task split, scene, PPO hyperparameter, GCOPTER, Diffusion,
or teacher dependency was changed.
