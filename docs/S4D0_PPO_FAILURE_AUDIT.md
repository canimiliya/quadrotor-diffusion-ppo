# S4-D0 Pure PPO Obstacle-Learning Root-Cause Audit

## Conclusion

This was a diagnostic-only run. It did not train, access TEST, change reward/observation/normalization/entropy, or start S5.

`FINAL_LABEL = PASS_S4D0_PPO_OBSTACLE_ROOT_CAUSE_AUDIT`

Primary diagnosis: **OBSERVATION_SCALE_IMBALANCE**

Ray sensitivity is low and ray/goal scale is materially imbalanced.

Formal progress remains **45%** and S4 remains blocked pending controller review.

Branch: `agent/s4d0-ppo-obstacle-audit-v1`; evidence-generation head: `0a4a1cb5152635c16e5d2a1b114b797a5dc82993`; provenance metadata was finalized afterward.

## Frozen R1 reproduction

- R1 checkpoint SHA-256: `3dea0fc92274bb34a542a5f2077dd233475a49cc05171dd4ffa661f0eb0c224b`
- checkpoint identity: `UnitBallSquashedGaussian=true`
- VAL: `18/54` success, `36` collision, `0` ground, `0` timeout
- families: OPEN `18/18`, BLOCK `0/18`, SBEND `0/18`
- action clipping: `0.000000`; support violation: `0.000000`

## Core evidence

- Near-obstacle 1 s intervention: true→clear delta `0.030620`, true→near delta `0.067296`.
- Dimensionless sensitivity mean: GOAL_DELTA `0.161290`, RAYS `0.001675`, ray/goal ratio `0.010384`.
- Policy exploration: log_std `[-0.2187143862247467, -0.1332813799381256, -0.3104768991470337]`, action_std `[0.8035512566566467, 0.8752188086509705, 0.73309725522995]`; near-BLOCK >30° diversity `0.2323150634765625`.
- Collision direction counts: toward `36`, away `0`, mixed `0`.
- Greedy-goal VAL: `18/54` success, `36` collision, mean return `-2.386559`.

## Reward and detour audit

| Family | expert return | negative-progress fraction | longest detour (s) | max distance increase (m) |
|---|---:|---:|---:|---:|
| OPEN | 26.565680 | 0.150617 | 0.575231 | 0.010907 |
| BLOCK | 27.255026 | 0.119638 | 0.702546 | 0.011502 |
| SBEND | 32.492676 | 0.081971 | 0.645833 | 0.010858 |

## Scope and verification

- TEST_ACCESSED = `false`.
- Teacher-independent PPO execution = `PASS`; expert data was used only for post-hoc reward audit.
- Regression and diagnostic tests are recorded in `tests`.
- No video/GIF or timestep trace was retained.

## What was proven / not proven

Proven: exact frozen R1 VAL outcome was reproduced; unit-ball support remained valid; ray intervention, normalized sensitivity, observation scale, exploration, collision direction, greedy-goal, expert reward, detour-credit, and discount audits completed without training.

Not proven: no corrective PPO training, no causal ranking beyond this audit, no TEST result, no S5 readiness.

Current blocker: `S4 remains blocked pending controller review`.
