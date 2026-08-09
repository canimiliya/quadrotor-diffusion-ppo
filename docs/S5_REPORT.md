# S5-R0 report

## Formal result

`FINAL_LABEL = PASS_S5_DIFFUSION_PRIOR_RESIDUAL_PPO_SANITY`

S5-R0 completed the authorized single-seed 500,000-step sanity experiment and
passes its integration Gate. Formal project progress is 70%. S6 was not
started.

The result must be read with an important limitation: the selected checkpoint
is step 0, where the deterministic zero residual exactly reproduces the frozen
S3-R2 prior. No trained residual checkpoint exceeded that prior. This is a
PASS for minimal integration and equal-budget sanity, not evidence that PPO
improved the diffusion prior.

## Frozen provenance and contracts

- S4 canonical main / S5 start: `6eb478aa22f46d63896324e36d4d3a6dffe6de11`
- branch: `agent/s5-diffusion-residual-ppo-v1`
- frozen S3-R2 checkpoint SHA256: `98de9a5d765ec1aeb48648497ac44ec98a6b6a071a607a01b4e49f3e13ab76cd`
- observation: raw 34D environment observation; S2 TRAIN-only fixed normalization
- prior: bounded x0, unit-ball support, T=100, DDIM=10, eta=0, eval/inference mode, no gradients
- composition: `project_unit_ball(a_prior + 0.25 * delta_a)`
- residual initialization: actor mean exactly 0, latent `log_std=-2.0`
- PPO: unchanged S4 `[256,256]`, learning rate 3e-4, 1024 rollout steps,
  batch 512, 10 epochs, gamma 0.99, GAE 0.95, clip 0.2, 8 envs
- reward, scenes, task split, 48 Hz, 960-step cap, and 0.30 m goal tolerance unchanged

The optimized eight-item training inference matched the S3 reference at fixed
inputs with maximum absolute error `1.1920928955078125e-07` against tolerance
`2e-06`. Evaluation retained batch-one S3 inference per trajectory. The frozen
prior reference reproduced VAL 37/54 with zero task-outcome mismatches.

## Hardware and execution

- GPU: NVIDIA GeForce RTX 5060 Ti
- CPU: 24 logical processors
- PyTorch: 2.7.1+cu128
- CUDA runtime: 12.8
- Stable-Baselines3: 2.6.0
- vector backend: eight independent in-process PyBullet clients (`DummyVecEnv`)
- valid training wall time: 4105.2294413 s
- total TRAIN environment steps: exactly 500,000

The in-process backend was an engineering recovery from Windows paging-file
pressure while another user-owned 20-worker experiment was running. It kept
eight environments and all scientific settings unchanged. Two incomplete
engineering runs are preserved separately and excluded from the valid result.

## Learning curve

| env steps | success | unsafe | mean return | projection |
|---:|---:|---:|---:|---:|
| 0 | 37/54 | 9 | 17.910 | 0.00% |
| 50k | 29/54 | 14 | 12.126 | 2.82% |
| 100k | 32/54 | 15 | 12.698 | 1.81% |
| 150k | 34/54 | 15 | 13.930 | 1.20% |
| 200k | 35/54 | 15 | 14.075 | 2.40% |
| 250k | 33/54 | 16 | 12.873 | 5.64% |
| 300k | 23/54 | 16 | 8.696 | 6.53% |
| 350k | 26/54 | 15 | 10.493 | 6.59% |
| 400k | 18/54 | 20 | 5.072 | 7.32% |
| 450k | 20/54 | 20 | 5.832 | 9.95% |
| 500k | 17/54 | 20 | 4.557 | 8.89% |

The trained-policy peak was 35/54 at 200k. Performance then degraded as the
mean residual norm and projection fraction grew. Across all 500,000 TRAIN
actions, mean prior norm was 0.8323, mean residual norm 0.2301, mean scaled
residual norm 0.05753, mean final norm 0.8393, and projection fraction 27.684%.
Support violations remained zero. Projection is part of the formal method and
is distinct from environment clipping.

## Selected VAL and TEST

Selection used highest success, fewer unsafe outcomes, higher return, then the
earlier step. The selected checkpoint is step 0:

- VAL: 37/54, unsafe 9, return 17.9095
- OPEN: 17/18
- BLOCK: 9/18
- SBEND: 11/18
- support violations: 0
- environment action clipping: 0

After 500k completed and the model was frozen, exactly one development TEST
was executed:

- TEST: 38/54, unsafe 9, return 18.3401
- OPEN: 18/18
- BLOCK: 8/18
- SBEND: 12/18
- support violations: 0
- environment action clipping: 0
- TEST used for selection or retraining: false

## Comparison and interpretation

The frozen Pure PPO result remains a valid weak baseline: 14/54 at 500k,
OPEN 14/18, BLOCK 0/18, SBEND 0/18. The integrated method's selected policy
has successful BLOCK and SBEND behavior and exceeds Pure PPO under the same
interaction budget, satisfying the S5 sanity Gate.

What was proven:

- frozen diffusion prior plus residual PPO can be trained for the full 500k contract;
- action support, gradient isolation, normalization, reward, and TEST protocol are correct;
- the integrated policy escapes the Pure PPO OPEN-only baseline through the frozen prior.

What was not proven:

- the learned residual improves the diffusion prior;
- multi-seed superiority or final sample-efficiency significance;
- cross-topology, ROS/PX4, real-flight, or S6 claims.

## Verification and frozen files

The independent verifier recomputed the selection order, VAL and TEST totals,
family totals, action-support counts, TEST protocol, frozen-prior identity,
normalization identity, PPO/reward contracts, and runtime dependency boundary.
It passed every check. Full regression reported `74 passed, 1 skipped, 4
warnings` with zero failures.

- best residual-PPO checkpoint SHA256: `c1d2cd6490dd0d61e49df0d8c0d86e019f4fe8b86e68b2bc58324ea00cde5583`
- last residual-PPO checkpoint SHA256: `325c3ce4df958bc58c0fdabff5e0127ef7ba3c58ea02a6b96c26481bfcdd935e`
- learning curve SHA256: `7ca899aa5b02ce85b89d2eb33a939bb1024ad6ef15cf5f9dc9fe58a4b5e71463`
- selected VAL rollout SHA256: `a8038bde62605de2d6e4aed75b29772fc75fb715cb23573785f8580842f04cd1`
- TEST rollout SHA256: `f36fcbc9d76081f6d4bedd9f624b747cbb1b2ad2022e8b16d7f0f0db26565603`

`UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW`

`WAITING_FOR_HIGH_LEVEL_CONTROLLER_AUDIT`
