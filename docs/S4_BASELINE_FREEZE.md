# S4 Pure PPO baseline freeze

## Formal stage decision

S4 is closed as `CLOSED_WITH_VALID_WEAK_PPO_BASELINE` at formal project
progress 55%.

The frozen 500,000-step Pure PPO experiment is a valid weak baseline. Its
environment, task split, reward, action support, TRAIN-only observation
normalization, PPO configuration, interaction budget, checkpoint selection,
and VAL evaluation protocol were completed and independently audited.

This management decision does not rewrite the S4-R2 experiment result:

`BLOCKED_S4R2_PPO_NO_OBSTACLE_LEARNING`

The scientifically correct interpretation is:

> Pure PPO failed to acquire obstacle-navigation behavior within the 500k
> interaction budget.

It is not evidence that the PPO implementation failed. The frozen best VAL
result remains 14/54: OPEN 14/18, BLOCK 0/18, SBEND 0/18, with 32 unsafe
episodes, mean return -3.394123992411189, zero policy support violations, and
zero environment action clipping.

## Frozen provenance

- S4-R2 source commit: `acfabed2dd28b7f23aee7007236a2bb58cb5ffba`
- Prior canonical main: `b10c8edf2fc834aac9136626a969c0c3108820ae`
- S4-R2 summary SHA256: `02e6e5002e376bac9ba962c555416032ae4b29ecc459a8a092ab120e6156f548`
- S4-R2 learning curve SHA256: `0583c7933bb049419b017ba78326c5d070fd44105834ecec619c0b85efb2c136`
- S4-R2 best VAL rollout SHA256: `d7352a1e56a85dab392d86f4907129d49e11947e2b7d3e94013d5ecf19db1e23`

S5 may use this frozen curve for an equal-budget comparison. This freeze does
not authorize changing the S4-R2 result, running a larger Pure PPO budget, or
claiming sample-efficiency superiority before the later multi-seed gate.
