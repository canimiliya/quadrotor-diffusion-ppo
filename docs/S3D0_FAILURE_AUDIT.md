# S3-D0 VAL Closed-Loop Safety Failure Root-Cause Audit

## 结论

本轮是诊断，不是算法通过，也没有推进正式进度。冻结的 S3-R1 policy 在 54 个 VAL task 上逐任务重放一致：`29 success / 9 collision / 6 ground contact / 10 timeout / 0 nonfinite`。因此输出：

```text
FINAL_LABEL = PASS_S3D0_VAL_FAILURE_ROOT_CAUSE_AUDIT
PRIMARY_DIAGNOSIS = MIXED_MULTIPLE_CAUSES
FORMAL_PROGRESS = 30%
CURRENT_BLOCKER = S3 remains blocked pending high-level review.
UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW
```

这不是 S3 safety gate 通过。S3 仍为 `IN_PROGRESS / BLOCKED`，本任务没有授权 S3-R2、PPO 或 S4。

## 冻结身份

```text
START_HEAD = e2f4e19f02e1b07de48617b607ceeb32340a5095
R1_HEAD = e2f4e19f02e1b07de48617b607ceeb32340a5095
BRANCH = agent/s3d0-failure-audit-v1
CHECKPOINT_SHA256 = 8afa677d7c8a34179c60f067209b6924170fc7854445048d7a7e9b55bbbc5fc8
CHECKPOINT_IDENTITY = frozen, prediction_target=x0, action_support=unit_ball_squash
TRAIN = 701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21
VAL = c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc
TEST = 88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66
TEST_ACCESSED = false
```

TRAIN/VAL 使用 R1 train-only normalization；TEST 仅沿用冻结 R1 identity，未打开 TEST dataset rollout，也没有 `test_rollout.csv`。

## R1 结果复核

原始 `artifacts/s3r1/val_rollout.csv` 与本轮 frozen-policy 重放逐 task 一致：

| outcome | 原始 R1 | 本轮重放 |
|---|---:|---:|
| success | 29 | 29 |
| collision | 9 | 9 |
| ground contact | 6 | 6 |
| timeout | 10 | 10 |
| nonfinite | 0 | 0 |

3 个 R1 determinism smoke 任务的 action trace hash 也全部双跑一致，并与 R1 记录一致。policy runtime source audit 通过：运行时输入仅为 scene、task、当前 observation、frozen model、frozen schedule；teacher independence 为 `true`。

## 量化诊断

### Offline expert-manifold

四组在 VAL expert observations 上的误差都很低，不能解释闭环 outcome：

| outcome | first-action MAE | MSE | cosine | predicted norm mean |
|---|---:|---:|---:|---:|
| SUCCESS (29) | 0.00903 | 0.000165 | 0.7864 | 0.6266 |
| COLLISION (9) | 0.00944 | 0.000178 | 0.8148 | 0.6736 |
| GROUND_CONTACT (6) | 0.00920 | 0.000159 | 0.7963 | 0.6398 |
| TIMEOUT (10) | 0.00902 | 0.000162 | 0.8065 | 0.6549 |

这证明“expert-manifold 上会模仿”与“离开 expert trajectory 后能安全闭环”是两个不同性质的问题。

### Action saturation

| outcome | action norm mean | `norm > 0.95` | `norm > 0.99` |
|---|---:|---:|---:|
| SUCCESS | 0.7756 | 60.64% | 23.61% |
| COLLISION | 0.9102 | 83.18% | 37.14% |
| GROUND_CONTACT | 0.7886 | 59.43% | 4.59% |
| TIMEOUT | 0.4733 | 35.43% | 16.51% |

Collision episodes clearly use the bounded action support more aggressively than SUCCESS. This supports `UNIT_BALL_SATURATION_AGGRESSIVENESS` as one cause, but the audit did not modify the squash or test a remedy.

### OOD and conditional ambiguity

TRAIN deterministic nearest-neighbor baseline (normalized 34-D observation) was:

```text
NN distance p50/p90/p95/p99 = 0.6693 / 1.0081 / 1.1393 / 1.5097
local action std_norm p50/p90/p95/p99 = 0.01384 / 0.13193 / 0.16105 / 0.21694
```

Rollout anchors were sampled every 0.5 s. Fraction above TRAIN NN p95 was SUCCESS `81.19%`, COLLISION `91.40%`, GROUND `83.29%`, TIMEOUT `88.00%`. All groups are substantially OOD under this diagnostic, so OOD is present but does not separate the outcomes strongly enough to be the primary taxonomy label.

The fraction above TRAIN local-action-variance p95 was SUCCESS `6.58%`, COLLISION `0.56%`, GROUND `11.85%`, TIMEOUT `6.50%`. This does not support conditional-action ambiguity as the dominant explanation.

### Collision

All 9/9 collision episodes were classified as:

```text
ray_blind = 0
obstacle_visible_but_unsafe_action = 9
other = 0
```

The classification used the final 1 s window, `ray_distance = ray_value * 3.0 m`, visible when minimum ray value `< 0.99`, and unsafe action when mean `action · nearest-ray-direction > 0.05`. The 9 collision windows had minimum inferred ray distances roughly `0.021–0.068 m`; thus this is not evidence of an 18-ray blind spot. It is evidence that an obstacle was observed, but the closed-loop action continued toward it, usually with high action norm.

### Ground contact

All 6/6 ground-contact episodes were classified as `persistent_downward`; dynamic lag, off-manifold-only, and other were each `0`. Across the final 1 s windows, every episode had `fraction(action_z < 0) = 1.0`; mean final-1 s `action_z` ranged from `-0.566` to `-0.804`, minimum altitude was about `0.011–0.014 m`, and minimum vertical velocity ranged from `-0.493` to `-0.682 m/s`. This supports `VERTICAL_GROUND_CONTROL_DRIFT` as a separate cause.

### Timeout

All 10/10 timeout episodes were classified as `near_goal_stall`; `no_progress`, `oscillatory`, and `other` were each `0` under the registered exclusive rules. Every timeout reached minimum goal distance `<= 1.0 m` but failed to enter the frozen `0.30 m` success tolerance. This supports `TIMEOUT_PROGRESS_FAILURE` as a separate cause; it is not evidence that the policy made no initial progress.

### Divergence time

Using the post-hoc same-task, time-aligned expert position and the diagnostic threshold `0.30 m`, divergence occurred in SUCCESS `26/29`, COLLISION `9/9`, GROUND `6/6`, TIMEOUT `10/10`. Median divergence times were respectively `1.135 s`, `1.083 s`, `1.510 s`, and `1.406 s`. This confirms early closed-loop departure from the expert trajectory, while remaining post-hoc only; no expert state or action was supplied to policy execution.

## What was and was not proven

Proven:

- frozen checkpoint identity and TRAIN/VAL identity;
- R1 outcome reproduction on all 54 VAL tasks;
- low offline expert-manifold error in all four outcome groups;
- action saturation separation for collision episodes;
- no ray-blind collision classification under the registered rule;
- persistent downward ground-contact classification;
- near-goal timeout classification;
- deterministic policy execution and teacher independence;
- existing regression: `33 passed, 2 warnings`.

Not proven:

- no causal intervention or remediation;
- no new training, no parameter tuning, no action-scale/squash change;
- no TEST performance;
- no PPO, S4, deployment, or safety qualification.

The only permitted next action is high-level controller review. Stop here.
