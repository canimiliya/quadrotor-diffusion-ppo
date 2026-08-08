# S3-R0 Conditional Diffusion Contract

S3-R0 tests whether a small conditional diffusion model can produce a useful
3D velocity action prior from the frozen S2 student-visible 34D observation.

The experiment is deliberately fixed: a 16-step action horizon, epsilon
prediction with 100 cosine diffusion steps, deterministic 10-step DDIM
sampling, `34 -> 128 -> 128` observation encoding, a width-256 four-block
residual MLP, AdamW at `3e-4`, batch size 512, and 10,000 updates.  The model
condition is only the normalized current observation; the normalization
statistics come from train observations only.  The raw first action is logged
before the unchanged M0 action mapping/clipping path.

Train windows never cross an episode boundary and windows at an episode tail
shorter than 16 steps are omitted.  Validation loss alone selects
`checkpoints/s3/best.pt`; `test.npz` is opened only after that checkpoint is
marked frozen.  The test rollout uses 54 frozen held-out tasks, 48 Hz, a 960
step budget, and no GCOPTER trajectory or planner state.

The S3 Go/No-Go gate requires at least 27/54 test successes, at least one
success in each OPEN/BLOCK/SBEND family, collision plus ground-contact
failures at most 25%, no nonfinite events, raw-action clipping at most 1%,
leakage audit PASS, and deterministic OPEN/BLOCK/SBEND smoke PASS.
