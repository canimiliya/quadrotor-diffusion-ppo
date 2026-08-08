# M0 contracts

## Action

The only student/expert action is a finite `float` array with shape `(3,)`
and range `[-1, 1]^3`. Its meaning is world-frame normalized target velocity:
`u = v_ref / 0.801 m/s`. The shared mapper uniformly clips the normalized
vector to the unit ball, retains raw and post-clipping values, then passes
`[direction_x, direction_y, direction_z, speed_ratio]` through the standard
`VelocityAviary._preprocessAction` DSLPID controller path. RPM values are only
internal actuator commands and never expert labels.

The single frozen environment speed limit is `0.801 m/s`, taken from the
maximum reference speed contract in every imported clean-reproduction scene.

## Observation

The candidate is 34-D: position (3), goal minus position (3), linear velocity
(3), quaternion (4), angular velocity (3), and 18 deterministic obstacle rays.
Ray directions are defined in the drone body-local frame, rotated to world
frame by the current quaternion, queried with `rayTestBatch`, and normalized
to `[0,1]` where 0 is an immediate obstacle and 1 means no imported obstacle
within the fixed 3.0 m range.

No future trajectory, reference velocity, trajectory progress, corridor
coefficients, obstacle list, planner cost, optimizer state, or hidden map
parameter enters the observation builder. The imported scene is used only to
provide the deployment goal and actual PyBullet obstacle bodies for ray tests.

## Success and diagnostics

The reference is complete only after the full analytic trajectory duration has
been executed. Goal success is tracked independently as the actual CF2X state
being within the frozen 0.30 m Euclidean goal tolerance during the deterministic
rollout. A two-second zero-reference settle period is included after reference
completion so completion is not confused with success. Collision counts come
from PyBullet contacts with obstacle bodies and the plane separately.

