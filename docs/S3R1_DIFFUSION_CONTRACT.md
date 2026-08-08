# S3-R1 bounded-x0 diffusion contract

This document freezes the one authorized S3-R1 repair relative to R0 audit head `5c5dc9443aa96bc40ac7881aaf63182097ef31b1`.

- The 34D student observation, 16-step horizon, 3D normalized velocity action, 100-step cosine forward process, 10-step deterministic DDIM (`eta=0`), and all M0/S2 environment contracts remain unchanged.
- The model keeps `34 -> 128 -> 128`, timestep embedding 64, width 256, four residual MLP blocks, and 621,360 parameters.
- Training target is the clean expert sequence `x0`, with only `MSE(predicted_x0, expert_action)`.
- The final model output is `unit_ball_squash(z) = z * tanh(||z||) / max(||z||, 1e-8)`, with a `1e-6` interior margin for float32 safety. It is applied before the shared M0 mapping.
- DDIM reconstructs `epsilon_hat` from the bounded `x0_hat` and uses the frozen deterministic update. The environment mapping remains unchanged and its clipping is logged separately.
- Normalization is computed from train observations only. Test data is not opened until after the best checkpoint is frozen and the VAL Gate is evaluated.

The S3-R1 PASS threshold is unchanged from R0. Failure at the VAL Gate terminates this task and does not authorize S3-R2, PPO, or hyperparameter tuning.
