# S3-R2 recovery contract

S3-R2 keeps the S3-R1 bounded-x0 Diffusion model, observation contract, action
support, horizon, DDIM schedule, and training hyperparameters unchanged. The
only algorithmic change is the training distribution.

Recovery sources are limited to the 252 accepted S2 TRAIN tasks. Each task may
contribute at most one deterministic first state whose post-hoc position error
from its own expert trajectory exceeds 0.30 m. No VAL or TEST state is used to
create labels or recovery windows.

The local frozen GCOPTER command-line interface accepts only start and goal
positions and uses static zero boundary derivatives internally. The recovery
record therefore preserves the actual anchor position, velocity, quaternion,
angular velocity, observation, scene, goal, and time, while the planner input
audit records the actual anchor velocity separately. The teacher is verified by
replaying the deterministic student prefix to the exact anchor and then
switching to the GCOPTER reference in the same PyBullet environment. A
trajectory enters training only if the full reference and settle window
completes with goal reached, zero collision, zero ground contact, zero
nonfinite events, and zero action clipping.

The VAL Gate is evaluated before TEST. TEST is opened at most once and is never
used for model selection or retraining.
