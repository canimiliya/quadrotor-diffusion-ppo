# S4-R2 TRAIN-only Observation Standardization Report

## Formal result

```text
FINAL_LABEL = BLOCKED_S4R2_PPO_NO_OBSTACLE_LEARNING
FORMAL_PROGRESS = 45%
S4 = IN_PROGRESS / BLOCKED
S5 = NOT_STARTED
TEST_ACCESSED = false
MAIN_CHANGED = false
```

The authorized 500,000-step Pure PPO run completed, but the usability Gate failed. The best checkpoint achieved `14/54` VAL success at 500k, all in OPEN: OPEN `14/18`, BLOCK `0/18`, SBEND `0/18`. Because both obstacle families remained at zero, TEST was not executed and `main` was not modified.

## Frozen intervention and leakage audit

The environment still emits the same raw 34D observation. Inside the shared actor/critic policy, each feature is transformed by `(o - mean_train) / scale_train`. Mean and population standard deviation were derived once from the 129,458 observations in S2 `train.npz` (SHA-256 `701a1d36...125db21`) using float64, then frozen as float32 checkpoint buffers. VAL and TEST contributed no statistics, online running statistics were not used, and checkpoint save/load preserved the buffers exactly.

TRAIN feature 20 was constant (`std=0`). Its fixed scale is `1.0`, so rollout-time variation remains finite and is not amplified by an epsilon divisor. Normalization contract SHA-256: `0dfbff8e68bd751bfab379c156e1c7ff92718bc1c7253c8eee3395490e7e1b93`.

Reward SHA-256 remained `92afefbf...5ae3ec`; UnitBallSquashedGaussian, `[256,256]` actor/critic, seed, PPO optimizer parameters, eight environments, scene/task split, control rate, horizon and 500k budget were unchanged. Pure PPO imported neither Diffusion nor expert runtime dependencies.

## Training and VAL evidence

Hardware: RTX 5060 Ti, CUDA 12.8, PyTorch 2.7.1+cu128, SB3 2.6.0, 24 CPU threads and eight PyBullet environments. The valid training wall time was `806.82 s`; peak allocated GPU memory was `90,076,672 bytes`.

| Steps | Success | Unsafe | Mean return |
|---:|---:|---:|---:|
| 0 | 0/54 | 0 | -1.920 |
| 50k | 0/54 | 11 | -3.030 |
| 100k | 6/54 | 4 | 1.534 |
| 150k | 10/54 | 2 | 5.102 |
| 200k | 12/54 | 11 | 2.768 |
| 250k | 11/54 | 23 | -2.070 |
| 300k | 6/54 | 15 | -1.152 |
| 350k | 7/54 | 16 | -1.011 |
| 400k | 7/54 | 32 | -7.232 |
| 450k | 13/54 | 30 | -3.552 |
| 500k | 14/54 | 32 | -3.394 |

At the selected checkpoint: collision `32`, ground `0`, timeout `8`, nonfinite `0`; policy support violation `0`, environment clipping `0`. Mean return also failed the strict comparison (`-3.394 <= -1.920` step-0), although family success already hard-failed.

## R1 to R2 interpretation

The dimensionless ray/goal sensitivity ratio increased from `0.01038` to `0.28841` (about `27.8x`). This demonstrates that fixed standardization materially changed obstacle-ray use, but it did not produce successful obstacle navigation within the frozen 500k budget. Compared with R1, total success changed `18 -> 14`; BLOCK remained `0 -> 0`; SBEND remained `0 -> 0`; SBEND collisions improved `18 -> 14` with four timeouts, while BLOCK collisions remained `18`.

Therefore S4-R2 proves the normalization implementation and sensitivity effect, not a usable Pure PPO obstacle-navigation baseline. It does not prove that larger budgets would fail. Per controller guidance, a future separately authorized study may consider `5,000k` followed by `20,000k`; neither was run here.

## Engineering incident

The first attempt was invalidated at 200k by a native Windows worker crash (`VCRUNTIME140.dll`, `0xc0000005`, parent BrokenPipe/EOFError). It never accessed TEST and its checkpoint was excluded. Worker BLAS/OpenMP oversubscription was then constrained while retaining eight environments and 24 main-process PyTorch threads. The full rerun completed from random initialization under the unchanged scientific contract. Evidence is preserved in `artifacts/s4r2/engineering_failure_01.json`.

## Scope closure

`WHAT_WAS_PROVEN`: TRAIN-only fixed normalization is finite, checkpoint-persistent, shared by actor and critic, and increases ray sensitivity; a valid 500k run and full VAL curve completed with unchanged reward/action/PPO contracts.

`WHAT_WAS_NOT_PROVEN`: no usable BLOCK/SBEND policy, no TEST result, no 5M/20M budget result, no PPO+Diffusion comparison, no S4 closure, and no S5 readiness.

`CURRENT_BLOCKER = BLOCKED_S4R2_PPO_NO_OBSTACLE_LEARNING`

`UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW`

`WAITING_FOR_HIGH_LEVEL_CONTROLLER_AUDIT`
