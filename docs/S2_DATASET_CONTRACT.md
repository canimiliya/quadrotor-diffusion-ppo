# S2 expert dataset contract

This stage produces a compact, task-isolated expert dataset from the nine
frozen M0 scene topologies. Endpoints are sampled from the interiors of the
first and final frozen corridor boxes with NumPy `PCG64` using seed
`20260808`. Each candidate is planned by the approved local GCOPTER planner
and accepted only after the unchanged M0 shared action and 34-D observation
closed-loop rollout passes all quality gates.

The accepted target is 40 tasks per scene, or 360 trajectories total. The
deterministic per-scene split uses seed `20260809`: 28 train, 6 validation,
and 6 test tasks. No timestep is shared between splits because splits are made
at whole-task/whole-episode granularity.

Each compressed NPZ contains float32 observations/actions/state arrays,
episode identifiers, and explicit `episode_offsets`/`episode_lengths`.
`trajectory_progress` is indexing metadata and is not part of the 34-D
student observation. Planner coefficients, corridor data, future reference
velocities, and planner state never enter the student arrays.

Large NPZ files and temporary planner outputs are gitignored. Only the task
manifest, split manifest, summary, and this contract/report are committed.
