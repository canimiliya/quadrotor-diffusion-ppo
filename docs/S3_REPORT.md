# S3-R0 Conditional Diffusion Action Prior Sanity

## Result

The fixed S3-R0 experiment was executed completely and independently audited.
It is blocked:

```text
TASK: S3-R0-CONDITIONAL-DIFFUSION-ACTION-PRIOR-SANITY-V1
FINAL_LABEL: BLOCKED_S3_DIFFUSION_ACTION_SUPPORT
```

The primary failure is action support: the frozen diffusion sampler produces
out-of-range raw actions on `99.950685%` of TEST control steps, versus the
required maximum of `1%`. The same rollout also fails the safety gate: 8/54
collision tasks plus 18/54 ground-contact tasks, or `48.15%`, versus the
maximum `25%`. TEST closed-loop success is `0/54`; no family has a success.

This is a genuine S3 failure of the fixed prior sanity experiment. No PPO,
BC, reward training, test-driven checkpoint selection, hyperparameter sweep,
or S4 work was started.

## Reproducibility and data identity

```text
START_HEAD: dd1ad5043dc83d3fc4841855318604ce3d38b069
BRANCH: agent/s3-diffusion-sanity-v1
S2_AUDIT_HEAD: dd1ad5043dc83d3fc4841855318604ce3d38b069
S2_IMPLEMENTATION_HEAD: 1a05975cf0bf8018683f123f4a77ebea2bf39b22

train.npz SHA256: 701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21
val.npz SHA256:   c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc
test.npz SHA256:  88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66

TRAIN_TRAJECTORIES: 252
VAL_TRAJECTORIES: 54
TEST_TRAJECTORIES: 54
TRAIN_WINDOWS: 125678
VAL_WINDOWS: 27127
TEST_WINDOWS: 26666
OBSERVATION_DIM: 34
ACTION_DIM: 3
ACTION_HORIZON: 16
TRAIN/VAL/TEST TASK OVERLAP: 0 / 0 / 0
EPISODE_BOUNDARIES: PASS
ALL_DATA_FINITE: PASS
TEST_ACCESSED_BEFORE_MODEL_FREEZE: false
```

## Fixed model and training

```text
MODEL: ConditionalDiffusionMLP
PARAMETER_COUNT: 621360
TRAINABLE_PARAMETER_COUNT: 621360
ARCHITECTURE: 34-128-128 observation encoder; sinusoidal time64; width256; 4 residual MLP blocks; output 16x3

DIFFUSION_TRAIN_STEPS: 100
BETA_SCHEDULE: cosine
PREDICTION_TARGET: epsilon
DIFFUSION_INFERENCE_STEPS: 10
DDIM_ETA: 0

TRAIN_SEED: 20260810
EVAL_BASE_SEED: 20260811
BATCH_SIZE: 512
LEARNING_RATE: 3e-4
WEIGHT_DECAY: 1e-4
MAX_GRADIENT_UPDATES: 10000
VALIDATION_INTERVAL: 500

DEVICE: cpu
PYTORCH: 2.13.0+cpu
CUDA: none
GPU: none

BEST_UPDATE: 9500
BEST_VAL_LOSS: 0.016388932385638755
FINAL_TRAIN_LOSS: 0.013001623563468456
ALL_FINITE: true
BEST_CHECKPOINT_SHA256: 9c5ef8d1e0f5f04ca453f8c226a4531ab505c9a040f3a7d8fb86ff754bd4f7d7
MODEL_FROZEN: true
```

Normalization used train observations only. The test split was accessed only
after the checkpoint was marked frozen; TEST was not used for any selection.

## Offline metrics

```text
VAL  denoising_loss=0.0163889324  first_action_MAE=5.2365007401  MSE=59.7279472354  cosine=0.5416975617
TEST denoising_loss=0.0167624851  first_action_MAE=5.2494425774  MSE=60.4207344055  cosine=0.5407333374
TEST offline raw-action clip fraction=0.9942248556
```

## Closed-loop metrics

Each task used 48 Hz, a maximum of 960 control steps, deterministic task-
derived sampling, only the 34D student observation as policy condition, and
the unchanged M0 action mapping.

```text
                 success  collision  ground_contact  timeout  nonfinite
VAL  (54 tasks)       0         8          13            33          0
TEST (54 tasks)       0         8          18            28          0
```

TEST family results:

| Family | Success | Collision | Ground contact | Timeout |
|---|---:|---:|---:|---:|
| OPEN | 0/18 | 0 | 12 | 6 |
| BLOCK | 0/18 | 5 | 6 | 7 |
| SBEND | 0/18 | 3 | 0 | 15 |

TEST scene results are retained in
`artifacts/s3/test_rollout.csv`; no timestep traces were persisted.

```text
RAW_ACTION_CLIP_COUNT: 38509
RAW_ACTION_CLIP_FRACTION: 0.9995068522
MAX_ABS_RAW_ACTION: 655992.75
PRIVILEGED_INFORMATION_LEAKAGE: PASS
```

Determinism smoke passed for one frozen TEST task in each family. The retained
hashes are in `artifacts/s3/summary.json`:

```text
OPEN  81e897ab5c3f82332e8c6b7d94fb708b237a847e9c987928ed54e4487c288032
BLOCK 68f00b1e51b8214d80130bb93b121b491dbb946352c4831c3f9ad8127239f93e
SBEND ccff1e98a764f63bd85c5578dbdb52d8c577ba612a4794d1d93dfb7075e93733
```

## Gate audit

```text
TEST success >= 27/54: FAIL (0/54)
Each family has a success: FAIL (OPEN=0, BLOCK=0, SBEND=0)
Collision + ground-contact <= 25%: FAIL (26/54 = 48.15%)
TEST nonfinite events = 0: PASS
Executed action nonfinite = 0: PASS
Raw action clip fraction <= 1%: FAIL (99.950685%)
Teacher-information leakage absent: PASS
TEST not accessed before model freeze: PASS
Determinism smoke: PASS
S2 dataset identity: PASS
Training numerical finiteness: PASS
Checkpoint freeze and VAL-only selection: PASS
M0/S0-S2 regression: PASS, 19 passed, 1 warning
```

## Interpretation and stop boundary

The model learned a low validation denoising loss under the fixed objective,
but this does not translate into a usable bounded velocity prior. S3 does not
pass and formal progress must not be advanced automatically. The correct next
action is controller review of the failure mode; the execution agent stops
here. No repair, sweep, PPO, or S4 is authorized by this result.

```text
WHAT_WAS_PROVEN:
The frozen S2 data identity, fixed diffusion training protocol, checkpoint
freeze boundary, leakage audit, deterministic inference, and closed-loop
evaluation pipeline are runnable and auditable; this particular diffusion
prior is not a usable S3 prior under the declared gate.

WHAT_WAS_NOT_PROVEN:
Diffusion success, Diffusion vs BC, Diffusion vs PPO, PPO + Diffusion,
sample efficiency, cross-topology generalization, or the paper hypothesis.

CURRENT_BLOCKER: BLOCKED_S3_DIFFUSION_ACTION_SUPPORT
FORMAL_STAGE: S3
FORMAL_PROGRESS: 30% pending controller audit; no automatic advancement
UNIQUE_NEXT_TASK: NONE — WAIT_FOR_CONTROLLER_REVIEW
WAITING_FOR_HIGH_LEVEL_CONTROLLER_AUDIT
```
