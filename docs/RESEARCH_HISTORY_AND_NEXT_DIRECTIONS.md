# Research history, retained result, and next directions

## Purpose and archival decision

This document is the compact long-term record of the local experimental history
that preceded the reproducible-release freeze.  The formal code release remains
in Git.  Large historical checkpoints, interrupted runs, duplicate validation
clones, local worktrees, caches, and raw logs were approved for local deletion
after their scientific content was reviewed and summarized here.

The only retained model weight is the formally selected S8-R4 Temporal U-Net
checkpoint:

| field | retained value |
|---|---|
| path | `checkpoints/s8r4/unet/pass_10.pt` |
| size | 188,488,387 bytes |
| SHA256 | `8ab962ea18e818d9fe6ef481f4f0f3162b39a30ba75fa87c3aacdeb27f66545d` |
| selection metric | frozen VAL offline diffusion loss |
| selected pass | 10 |
| selected before closed loop | yes |
| training seed | 20260820 |
| TEST accessed in S8-R4 | false |

Pass 20 and pass 30 had higher closed-loop VAL success, but they were not the
pre-registered offline-loss selection.  They are therefore historical
checkpoints, not substitutes for the selected pass-10 model.

## Frozen interfaces

- Student observation: 34 dimensions: position, goal-relative position,
  linear velocity, quaternion, angular velocity, and 18 local obstacle rays.
- Action: three-dimensional normalized world-frame target velocity in the
  Euclidean unit ball; physical speed limit 0.801 m/s.
- Sequence contract: one observation conditions an H=16 sequence of 3-D
  actions; only the first action is executed before replanning.
- Diffusion contract used by S8-R4: bounded x0, cosine schedule, T=100,
  DDIM=10, eta=0, MSE-x0, eager-exact inference.
- External expert source: GCOPTER static-known-map trajectories executed in
  gym-pybullet-drones/PyBullet.  This is not dynamic replanning, ROS 2, PX4,
  real flight, or hardware qualification.

## Dataset lineage retained locally

All formal datasets remain on disk.  Their relative locations and identities
are indexed in `docs/DATA_AND_CHECKPOINT_MANIFEST.md`.

| stage | retained data | scientific role |
|---|---|---|
| S2 | `artifacts/s2/dataset/{train,val,test}.npz` | original nine-topology expert dataset, 252/54/54 trajectories |
| S3-R2 | `artifacts/s3r2/recovery_dataset/` | TRAIN-only recovery augmentation used to make the bounded Diffusion prior usable |
| S7 | `artifacts/s7/fresh_holdout/` | frozen six-family fresh-topology development holdout and GCOPTER feasibility evidence |
| S8-R2 | `artifacts/s8r2_10k/{train,val,test}.npz` plus manifests | 1,000 maps, 10,000 trajectories, 6,120,725 transitions, map-isolated 8k/1k/1k split |
| S8-R3 | `artifacts/s8r3/train_obs_normalization.npz` | TRAIN-only normalization used by S8-R3/S8-R4 |

The S8-R2 generation cache is not a dataset of record.  The three concatenated
NPZ files, map/task manifests, summaries, and hashes are the frozen dataset;
the cache is reproducible intermediate storage.

## Experimental development history

### M0 and S2: expert bridge and first dataset

M0 established the GCOPTER-to-PyBullet interface.  All nine frozen scenes
passed reference and goal checks, with no collision, ground contact,
non-finite action, or expert clipping.  S2 then generated the first balanced
360-trajectory expert dataset.  These stages proved the data and control
interface, not any learning-method advantage.

### S3: Diffusion failure, repair, diagnosis, and recovery

S3-R0 failed decisively.  Epsilon prediction produced extreme raw actions,
99.95% rollout clipping, 0/54 TEST success, and unsafe outcomes.  S3-R1 changed
the output contract to bounded x0 with unit-ball squash.  This eliminated
support violations and reached 29/54 VAL success, but narrowly failed the
safety gate at 15/54 collision plus ground contact.

The S3-D0 audit found mixed causes: collision episodes used near-saturated
actions despite visible obstacles, all ground contacts showed persistent
downward commands, timeouts were near-goal stalls, and rollout observations
became strongly off the expert manifold.  S3-R2 added TRAIN-only recovery data
and passed the sanity gate at 37/54 VAL and 38/54 one-time TEST success, with
unsafe count 9.  This established the first usable frozen Diffusion prior.

### S4: Pure PPO baseline remained obstacle-blind

Pure PPO never became a usable all-family baseline under the frozen 500k
budget.  S4-R0/R1 reached 18/54 but all successes were OPEN; BLOCK and SBEND
remained 0/18.  The unit-ball distribution repair removed the action-space
mismatch but did not create obstacle learning.  The root-cause audit identified
severe observation-scale imbalance: ray/goal sensitivity ratio was about
0.0104.  TRAIN-only standardization increased the ratio about 27.8 times, but
S4-R2 still achieved only 14/54, again with zero obstacle-family success.

### S5 and S6: prior improves online sample efficiency; residual PPO does not

S5 integrated a frozen Diffusion prior with a PPO residual.  The integration
was correct, but the selected checkpoint was step 0.  Trained residual policies
never beat the frozen prior and eventually degraded it.

S6 repeated Pure PPO and Diffusion+PPO with three seeds and 500k interactions
per run.  Diffusion+PPO improved normalized success AUC over Pure PPO for all
three seeds; mean paired improvement was 0.29753 with bootstrap 95% interval
[0.26204, 0.33889].  However, residual training again degraded the frozen
37/54 prior, and collision/unsafe AUC did not improve.  The defensible result is
that a planning-guided offline prior improves online environment-interaction
sample efficiency, not that PPO improves the prior or makes it safer.

The S6-P0 engineering stage accelerated exact batch-one Diffusion inference
with a CUDA Graph while preserving bitwise action and closed-loop identity.
Batched DDIM was rejected because small numerical differences changed 26 VAL
outcomes.

### S7: BC explains most prior benefit; Diffusion advantage is limited

Matched BC alone achieved 31/54 on the original VAL compared with Diffusion
37/54.  BC+PPO mean success AUC was 0.41204, Pure PPO 0.15926, and
Diffusion+PPO 0.45679.  Diffusion exceeded BC for all three paired seeds, but
the mean delta 0.04475 was below the preregistered 0.05 threshold.  Therefore
the online benefit is primarily an expert-prior effect; a Diffusion-specific
advantage was not established at the registered strength.

On frozen fresh topologies, Diffusion achieved 10/54, BC 4/54, and Pure PPO
0/54.  Both learned priors lost 0.50 success rate relative to the original
distribution, and Diffusion had 30 unsafe outcomes.  This is limited
cross-topology evidence, not robust generalization or deployment readiness.

### S8-R1A: low-data study favored BC competitiveness

Across 18 offline runs, Diffusion did not gain a reliable low-data advantage.
At 25% data it led BC by 6.17 percentage points, below the 10-point threshold;
at 50% and 100% BC was stronger by 4.94 and 19.14 points.  The correct result is
`BC competitive or better`, not a low-data Diffusion superiority claim.

### S8-R2 and S8-R3: 10K expert base and training-sufficiency audit

S8-R2 created the retained 10K procedural GCOPTER dataset: 1,000 unique maps,
10 families, 10 trajectories per map, 10,000 accepted trajectories, and
6,120,725 transitions.  Only two of 10,002 candidates failed planning.  All
accepted trajectories passed collision, ground, finite, action-support, and
clipping gates.  Splits are map-isolated and TEST remains sealed for S8-R4.

S8-R3 showed that old 10k-update training was only about 1.07 effective passes
over the new TRAIN set.  BC rose from 62.7% at pass 1 to 84.2% at pass 30 but
was highly non-monotonic.  MLP Diffusion was 82.4% at pass 1, peaked at 95.5%
at pass 5, and ended at 82.5%; it was neither proven under-trained nor proven
saturated.  At pass 30 BC exceeded MLP Diffusion by 1.7 points.

### S8-R4: Temporal U-Net is promising under a limited audit

The Conditional 1-D U-Net has 47,110,915 parameters, versus 621,360 for the
MLP Diffusion model and 592,688 for BC.  Under one seed and the frozen 10K
TRAIN/VAL contract, U-Net success was 97.6%, 98.7%, 99.1%, 99.6%, and 99.8%
at passes 1/5/10/20/30.  The offline-loss-selected pass-10 model reached 99.1%,
18.3 percentage points above the corresponding offline-selected MLP result.
Late-pass U-Net behavior was also more stable than MLP.

This supports `PASS_S8R4_UNET_PROMISING`, but it is not a parameter-matched
comparison, uses only one seed, does not prove superiority over BC, and does
not include sealed TEST evaluation.

### S8-R5: visualization only

S8-R5/R5B generated advisor-facing figures from frozen S8-R2 TRAIN data.  They
are data visualizations, not new model evidence.  No training or TEST model
evaluation occurred.

## Consolidated scientific conclusions

1. The GCOPTER expert interface and retained datasets are valid and auditable.
2. Pure PPO under the tested 500k budget does not learn obstacle navigation.
3. Bounded action support and recovery data are essential for the original
   Diffusion prior.
4. A strong offline expert prior materially improves online sample efficiency.
5. The tested PPO residual usually erodes rather than improves either prior.
6. Matched BC explains most of the prior benefit; MLP Diffusion's extra benefit
   is consistent but below the preregistered S7 strength threshold.
7. Neither BC nor MLP Diffusion demonstrates robust fresh-topology safety.
8. Temporal U-Net is the strongest observed architecture under the S8-R4
   single-seed audit, but its capacity is about 76 times the MLP's and the
   comparison is not final.

## Recommended next research directions

These are recommendations only; none is automatically authorized by cleanup.

1. Run a preregistered three-seed confirmation of the retained U-Net contract,
   with selection fixed before closed-loop evaluation.
2. Add a parameter/capacity-matched control or scaling curve before attributing
   the gain specifically to temporal convolution rather than model size.
3. Compare U-Net Diffusion against a sufficiently trained matched BC under the
   same effective passes and seeds.
4. Use a genuinely untouched topology/test protocol only after architecture,
   seeds, and selection rules are frozen; do not tune on the S7 development
   holdout or sealed S8-R2 TEST.
5. Treat safety and distribution shift as separate objectives.  Investigate
   conservative action magnitude, recovery-state coverage, uncertainty/OOD
   detection, and obstacle-aware safety layers before claiming deployment.
6. Do not continue the current residual-PPO formulation as though it improves
   the prior; any future online adaptation should include an explicit
   prior-preservation constraint and a no-degradation gate.

## Claim boundary for papers and presentations

The strongest supported statement is: planning-expert-trained offline action
priors improve PPO's online environment-interaction sample efficiency; matched
BC captures most of this benefit, while MLP Diffusion shows a limited extra
advantage and Temporal U-Net is promising in a single-seed, non-parameter-
matched architecture audit.  The project does not establish robust safety,
dynamic replanning, real-world transfer, PX4/ROS 2 deployment, or real flight.

