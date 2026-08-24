# Network architecture index

This index points to the implemented neural networks and policy components. It
does not redefine any frozen model, action contract, or scientific result.

## At a glance

| model | implementation | principal classes | input | output | current research use |
|---|---|---|---|---|---|
| MLP Diffusion | `src/quadrotor_diffusion_ppo/diffusion/model.py` | `ConditionalDiffusionMLP`, `ResidualMLPBlock`, `SinusoidalTimeEmbedding` | noisy action sequence `(B,16,3)`, observation `(B,34)`, timestep `(B,)` | predicted action sequence `(B,16,3)` | S3-R2 recovery diffusion and S8-R3 MLP diffusion training-sufficiency baseline |
| Temporal 1-D U-Net | `src/quadrotor_diffusion_ppo/diffusion/unet1d.py` | `ConditionalUnet1D`, `ConditionalResidualBlock1D` | noisy action sequence `(B,16,3)`, observation `(B,34)`, timestep `(B,)` | bounded clean-action prediction `(B,16,3)` | S8-R4 conditional temporal U-Net architecture audit; `pass_10.pt` is the retained selected checkpoint |
| Behavioral Cloning | `src/quadrotor_diffusion_ppo/bc/model.py` | `MatchedSequenceBC` | observation `(B,34)` | bounded action sequence `(B,16,3)` | deterministic matched-scale BC baseline used in S7, S8-R1A, and S8-R3 comparisons |
| PPO policy components | `src/quadrotor_diffusion_ppo/ppo/` | `UnitBallActorCriticPolicy`, `UnitBallSquashedGaussianDistribution`, `FixedObservationStandardizer`, `FrozenDiffusionPrior`, `FrozenBCPrior` | 34-D observation; SB3 actor latent/mean and log standard deviation | one normalized 3-D velocity action inside the open unit ball | Pure PPO and prior/residual PPO evidence routes; no new PPO training is implied by this index |

## MLP Diffusion

`ConditionalDiffusionMLP` encodes the 34-D observation, embeds the diffusion
timestep with `SinusoidalTimeEmbedding`, flattens the 16-by-3 noisy action
sequence, and processes the concatenated condition through residual MLP blocks.
`predict_raw` returns the denoiser prediction. `forward` optionally applies the
shared per-step unit-ball squash according to the frozen configuration.

## Temporal 1-D U-Net

`ConditionalUnet1D` keeps the same observation, timestep, and action-sequence
interface as the MLP denoiser while making the horizon an explicit temporal
axis. Its 256-to-512-to-1024 encoder/bottleneck/decoder uses skip connections
and FiLM-conditioned `ConditionalResidualBlock1D` modules. The S8-R4 contract
predicts bounded clean actions (`x0`), not epsilon or velocity targets.

## Behavioral Cloning

`MatchedSequenceBC` maps one 34-D observation directly to a 16-by-3 action
sequence using the same residual-MLP scale as the MLP diffusion baseline. Its
forward output is always mapped into the per-step open 3-D unit ball.

## PPO components

The PPO route uses Stable-Baselines3's PPO/actor-critic network rather than a
second project-local full PPO network implementation. Project-local code
supplies the frozen interfaces around it:

- `unit_ball.py` defines the actor-critic policy and the change-of-variables
  corrected Gaussian-to-unit-ball action distribution;
- `normalization.py` defines frozen TRAIN-only 34-D observation statistics and
  feature standardization;
- `env.py` defines the 34-D observation / 3-D action navigation environment;
- `residual.py` and `bc.py` expose frozen Diffusion and BC priors for the
  prior/residual evidence routes;
- `contract.py` and `runtime.py` hold the frozen reward/training contract and
  checkpoint/evaluation orchestration.

These paths document where the implementations live; they do not claim that
PPO, BC, MLP Diffusion, or Temporal U-Net has passed any experiment beyond the
stage-specific evidence recorded in the corresponding reports.
