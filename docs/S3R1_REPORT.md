# S3-R1 formal report

## Result

`FINAL_LABEL = BLOCKED_S3R1_BOUNDED_X0_VAL_WEAK`

This is a bounded-x0 action-support repair, not a pass. Training, dataset isolation, deterministic sampling, numerical finiteness, and the action-support gate passed. The VAL safety gate failed because collision plus ground contact was 15/54 = 27.78%, above the frozen 25% limit. Per protocol, TEST was not opened.

## Provenance and frozen data

- Task: `S3-R1-BOUNDED-X0-DIFFUSION-ACTION-SUPPORT-REPAIR-V1`
- Start head / S3-R0 audit head: `5c5dc9443aa96bc40ac7881aaf63182097ef31b1`
- S3-R0 implementation/report head: `754255746f262ed961f260ff7829af1446ac229a`
- Branch: `agent/s3r1-bounded-x0-v1`
- S2 dataset SHA256: train `701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21`; val `c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc`; test `88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66`.
- Trajectories: 252 / 54 / 54; observation/action dimensions: 34 / 3; task overlap: zero; boundaries and finite checks: pass.

## R0 diagnosis and R1 repair

At DDIM `t=99`, `alpha_bar=2.4285407107527135e-7`, `sqrt_alpha_bar=0.0004928022636669513`, and `1/sqrt(alpha_bar)=2029.2114580785008`. R0 had best VAL loss `0.016388932385638755`, offline VAL first-action MAE/MSE/cosine `5.2365007401 / 59.7279472354 / 0.5416975617`, TEST max raw action `655992.75`, and raw clip fraction `0.9995068521594684`.

R1 predicts clean `x0` directly and applies `unit_ball_squash` before M0 mapping. The sampler oracle passed with maximum reconstruction error below `2.4e-7`; zero, small, large, extreme, and random squash inputs were finite and within the unit ball.

## Training

- Seed `20260810`; CPU PyTorch `2.13.0+cpu`.
- 10,000 updates; batch 512; AdamW `3e-4`, weight decay `1e-4`; gradient clip `1.0`; VAL every 500.
- Best update `10000`; best VAL x0 loss `0.00027222433963381267`; final logged train loss `0.000236`; all finite.
- Model parameters: 621,360; checkpoint SHA256: `8afa677d7c8a34179c60f067209b6924170fc7854445048d7a7e9b55bbbc5fc8`.
- Train-only normalization matches R0.

## VAL Gate

Overall: 29/54 success (53.70%), 9 collision, 6 ground contact, 10 timeout, 0 nonfinite. By family: OPEN 13/18, BLOCK 10/18, SBEND 6/18. Network latent max abs/norm: `6.8216943741 / 6.8468871117`. Diffusion x0 max abs/norm: `0.9997767806 / 0.9999932647`; support violations: `0`, fraction `0`. Executed action clip count/fraction: `0 / 0`.

The success and family gates pass, but collision plus ground contact is `15/54 = 27.78%`; therefore VAL Gate fails. VAL determinism smoke passes for OPEN, BLOCK, and SBEND. Teacher-independence audit passes.

`R1_TEST_EXECUTED = false`, `R1_TEST_RUN_COUNT = 0`, `R1_TEST_USED_FOR_SELECTION = false`. No `test_rollout.csv` is produced.

## Scope conclusion

This run proves that bounded-x0/unit-ball parameterization removes the R0 action explosion: VAL diffusion support violation and executed clipping are both zero, while closed-loop success reaches 29/54. It does not prove the frozen S3 safety gate, TEST performance, final generalization, PPO compatibility, or any S4 result. S3-R2, sweeps, BC, PPO, and S4 remain frozen pending controller review.
