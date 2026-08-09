# S5 residual PPO contract

This contract was frozen before any S5 VAL evaluation or formal PPO training.
No VAL or TEST result was used to choose the residual design.

## Frozen prior

- checkpoint: `checkpoints/s3r2/best.pt`
- SHA256: `98de9a5d765ec1aeb48648497ac44ec98a6b6a071a607a01b4e49f3e13ab76cd`
- model: frozen `ConditionalDiffusionMLP`, 34D condition, horizon 16, action dimension 3
- inference: bounded x0, unit-ball support, cosine T=100, deterministic DDIM 10 steps, eta 0
- seed: exact S3-R2 `stable_seed(task_id) + episode_step`
- gradients: disabled; eval and inference mode only

The optimized training path batches the eight independently seeded initial
noise tensors and performs each DDIM model call as a batch. Before formal
training it must match the original S3-R2 batch-one reference path within the
documented numerical tolerance.

## Observation normalization

Both branches use statistics derived only from
`artifacts/s2/dataset/train.npz::observations`. The S3-R2 checkpoint mean and
standard deviation are float32-identical to the independently recomputed S2
TRAIN statistics. Feature 20 is constant at exactly 1 in S2 and at runtime;
therefore S3's epsilon floor and S4's safe unit scale both map it to exactly
zero. PPO retains the checkpoint-persistent S4 fixed standardizer.

## Residual decision variable and composition

PPO samples a residual action `delta_a` from the audited
`UnitBallSquashedGaussian`, so `||delta_a|| < 1`, and its log probability is
computed for that residual variable.

The frozen residual scale is:

`alpha = 0.25`

Composition is the Euclidean unit-ball projection:

```text
z = a_prior + alpha * delta_a
a_exec = z                 when ||z|| <= 1
a_exec = z / ||z||         otherwise
```

Consequently, zero residual reproduces every in-support prior exactly. At the
0.801 m/s physical speed limit, the maximum unprojected correction magnitude
is 0.20025 m/s.

## Initialization

- actor final mean weights and bias: exactly zero
- latent Gaussian `log_std`: `-2.0`
- latent standard deviation: `exp(-2) = 0.135335...`

Deterministic step-0 behavior therefore exactly reproduces the frozen prior.
The small nonzero stochastic exploration gives a typical scaled initial
residual norm of roughly 0.05 in normalized action units, selected from the
action scale and TRAIN-only contract rather than a VAL sweep.

All S4 PPO optimizer settings, network widths, reward, task split, eight-env
layout, seed 20260812, and 500,000-step interaction budget remain frozen.
