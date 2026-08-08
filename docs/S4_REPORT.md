# S4-R0 Pure PPO baseline report

## Result

`S4-R0` is **BLOCKED**, not PASS. The canonical main fast-forward and the complete Pure PPO protocol are valid, but the usability gate fails because the selected VAL model learned only `OPEN` tasks: `OPEN=18/18`, `BLOCK=0/18`, `SBEND=0/18`.

Formal label:

```text
BLOCKED_S4_PPO_NO_LEARNING
```

Formal progress remains `45%`. S5 is not authorized by this result and was not started.

## Canonical main

- `main` was fast-forwarded from `8a315379c018f9ae617ca7801b96653a73a70432` to the audited S3-R2 commit `b10c8edf2fc834aac9136626a969c0c3108820ae`.
- No merge commit, rebase, squash, history rewrite, or force push was used.
- `origin/main` matched `b10c8edf2fc834aac9136626a969c0c3108820ae` before S4 work began.

## Frozen run

- Branch: `agent/s4-pure-ppo-v1`
- GPU: NVIDIA GeForce RTX 5060 Ti
- CPU: 24 cores; eight PyBullet worker processes
- PyTorch: `2.7.1+cu128`; CUDA runtime `12.8`
- Stable-Baselines3: `2.6.0`
- Train device: CUDA
- Observation/action: 34D / 3D shared normalized velocity action
- Training: exactly `500,000` environment steps, `N_ENVS=8`
- TRAIN/VAL/TEST: `252/54/54`
- Reward, environment, network, PPO hyperparameters, reset distribution, and TEST boundary were unchanged after training started.

## Evidence

| checkpoint | VAL success | unsafe | mean return |
|---:|---:|---:|---:|
| 0 | 0/54 | 0 | -1.920 |
| 250,000 | 12/54 | 34 | -4.148 |
| 450,000 | 18/54 | 36 | -1.732 |
| 500,000 selected | 18/54 | 36 | -1.703 |

Selected VAL by family: `OPEN 18/18`, `BLOCK 0/18`, `SBEND 0/18`.

The frozen model was evaluated on TEST exactly once: `18/54` success, `36` collision, `0` ground contact, `0` nonfinite, and `0` timeout. TEST was not used for selection.

Protocol audit and the 41-test regression pass. The usability gate fails only on the required per-family VAL success condition; therefore no claim is made that Pure PPO is a usable all-family navigation baseline.
