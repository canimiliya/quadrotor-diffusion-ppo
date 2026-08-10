# S7 matched-BC and fresh-topology preregistered contract

This contract was frozen before formal BC training or any S7 policy evaluation.

## Matched sequence BC

- Input: the same deployment-visible 34-D observation used by S3-R2.
- Output: a deterministic `16 x 3` action sequence; rollout executes only the
  first action and predicts a new sequence at the next environment step.
- Every three-dimensional action is mapped by the same open-unit-ball squash.
- Architecture: `34 -> 128 -> 128 -> 256`, four width-256 residual MLP blocks,
  then `256 -> 48`. No RNN, Transformer, map, corridor, planner coefficient,
  future state, or diffusion feature is available.
- The architecture has 592,688 parameters versus 621,360 in the frozen
  S3-R2 denoiser (95.38%). It will not be changed after observing VAL results.
- Loss and optimization: SmoothL1, AdamW, learning rate `3e-4`, weight decay
  `1e-4`, batch size 512, gradient clip 1.0, 10,000 updates, validation every
  500 updates, deterministic seed 20260810. Best checkpoint is selected only
  by original S2 VAL sequence loss.

## Offline information identity

Training windows are exactly the S3-R2 offline budget: S2 TRAIN plus the
accepted S3-R2 TRAIN-only recovery trajectories, same H=16 no-cross-episode
window construction. Normalization is derived only from original S2 TRAIN,
matching S3-R2. Original S2 VAL is used only for checkpoint selection. S2 TEST
and the S7 fresh holdout do not participate in training or selection.

## Online comparison

BC-prior PPO uses `project(a_bc + 0.25 * delta_ppo)` and otherwise reuses the
frozen S6 PPO, reward, normalization, initialization, 8-environment, 500k-step,
seed, and deferred `0,50k,...,500k` VAL-evaluation contract. S6 Pure PPO and
Diffusion-prior PPO evidence is reused without retraining.

Diffusion-specific advantage is supported only if all three paired
`AUC_DIFFUSION_PPO - AUC_BC_PPO` values are positive and their mean is at least
0.05. Results are reported honestly otherwise.

## Fresh topology holdout

The holdout contains six new static-box structural families and 54 tasks. Its
scene definitions, task endpoints, generation seed, and feasibility evidence
must be generated and frozen before any policy is evaluated. No policy result
may remove, alter, or tune a topology or task. GCOPTER may be used only for
offline feasibility certification; its trajectory is never given to a tested
policy. The holdout is not used for architecture, checkpoint, or hyperparameter
selection.

The S6 safety conclusion remains conservative unless broad, stable evidence
supports a change. S7 does not claim that PPO improves a prior when the selected
checkpoint is step zero.
