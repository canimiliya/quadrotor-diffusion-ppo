# GCOPTER-PIVOT-M0 report

**Final label:** `PASS_GCOPTER_PIVOT_M0`

The approved M0 bridge was implemented and independently rerun. The clean
reproduction source tree remained unchanged. Its published release retained
the nine frozen scene YAML files but not the generated PyBullet reference
coefficient files, so the existing frozen planner was compiled read-only
against official source commit `e0444f6d47b84f972ced91746b05feb36ce1fd4f` and
its small outputs were generated under gitignored `.deps/`.

## Frozen contracts

- Action: finite `float`, shape `(3,)`, range `[-1, 1]^3`, world-frame
  normalized target velocity.
- Global speed limit: `0.801 m/s`.
- Mapping: `v_ref / 0.801 -> uniform unit-ball clip ->
  [direction_x, direction_y, direction_z, speed_ratio] -> standard
  `VelocityAviary._preprocessAction`.
- Observation: 16-D base state plus 18 body-local rays rotated to world frame,
  ray range 3.0 m, `[0,1]` with 0 near and 1 clear, total 34-D.
- Goal contract: actual state within 0.30 m at some deterministic rollout
  sample. Reference completion is separately defined by full reference
  duration; a 2.0 s zero-reference settle is included afterward.

## Nine-case result

| Scene | Reference | Goal | Final distance (m) | Min clearance (m) |
|---|---:|---:|---:|---:|
| OPEN_00 | PASS | PASS | 0.046014 | n/a |
| OPEN_01 | PASS | PASS | 0.062931 | n/a |
| OPEN_02 | PASS | PASS | 0.020584 | n/a |
| BLOCK_00 | PASS | PASS | 0.063565 | 0.186114 |
| BLOCK_01 | PASS | PASS | 0.029253 | 0.138388 |
| BLOCK_02 | PASS | PASS | 0.083734 | 0.224822 |
| SBEND_00 | PASS | PASS | 0.103839 | 0.323423 |
| SBEND_01 | PASS | PASS | 0.067042 | 0.526380 |
| SBEND_02 | PASS | PASS | 0.079299 | 0.338310 |

Aggregate: reference `9/9`, goal `9/9`, obstacle collisions `0`, ground
contacts `0`, nonfinite events `0`, expert clipping `0/5784 = 0.0`, maximum
raw action component `0.9999973425`, maximum commanded speed `0.8009998663 m/s`,
and maximum speed-limit excess `0.0 m/s`.

The reported maximum actual state velocity was `0.8256376532 m/s`; this is a
plant response diagnostic and is distinct from the commanded target-speed
contract, whose maximum was below `0.801 m/s`.

## Tests and reproducibility

```text
pytest -q: 6 passed, 1 warning
run_m0.py: PASS_GCOPTER_PIVOT_M0 scenes=9 reference=9/9 goal=9/9
independent action-trace hash rerun: DETERMINISTIC_TRACE_HASHES=True
```

Runtime: Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, PyBullet 3.2.7,
Gymnasium 1.3.0, gym-pybullet-drones 2.1.0 from local snapshot commit
`e712698a05a80728b06572819dcf044596707754`.

Committed M0 artifacts are `artifacts/m0/summary.json` and
`artifacts/m0/per_scene.csv`. Nine raw action traces remain local and ignored
under `artifacts/m0/action_traces/`; they are small diagnostics, not videos,
GIFs, checkpoints, or a training dataset. Generated planner binaries and
reference outputs remain local and ignored under `.deps/`.

## Scope boundary

This proves only that the frozen GCOPTER reference can enter the shared
normalized velocity interface and drive CF2X through the obstacle-aware
VelocityAviary bridge on these nine deterministic scenes. It does not prove
Diffusion, PPO, BC, data-scale readiness, generalization, paper hypotheses,
PX4/ROS 2, real flight, or dynamic replanning.
