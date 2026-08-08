# S3-R2 report

## Result

`FINAL_LABEL = PASS_S3R2_RECOVERY_AUGMENTED_DIFFUSION_SANITY`

The run used the frozen R1 checkpoint and the CUDA-first `smd-blackwell`
environment on an NVIDIA GeForce RTX 5060 Ti. All 252 TRAIN tasks were rolled
out; 242 produced one first-divergence anchor. GCOPTER accepted 148 planning
attempts, and 146 teacher trajectories passed the full closed-loop acceptance
contract: OPEN 57, BLOCK 61, SBEND 28.

The recovery data was TRAIN-only. Original TRAIN windows: 125,678; recovery
windows: 66,939; total R2 windows: 192,617. VAL/TEST recovery samples were
zero.

VAL passed with 37/54 successes, 9 unsafe outcomes, 0 nonfinite outcomes, zero
support violations, and zero executed clipping. Relative to R1, this is
`Δsuccess=+8`, `Δcollision=0`, `Δground_contact=-6`, `Δtimeout=-2`, and
`Δunsafe=-6`. The single TEST run passed with 38/54 successes and 9 unsafe
outcomes. OPEN, BLOCK, and SBEND each had at least one success.

Training used PyTorch `2.7.1+cu128`, CUDA `12.8`, 10,000 updates, and 78.74 s
wall time. Independent verification and three-family VAL determinism smoke
passed. Regression: `36 passed, 4 warnings`.

The formal project progress remains 30% until the high-level controller
accepts this complete S3-R2 Gate. This result does not prove superiority over
PPO or BC, sample-efficiency improvement, or cross-topology generalization.
No S3-R3, S4, or PPO work is authorized by this report.
