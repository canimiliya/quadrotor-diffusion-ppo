# S4 Pure PPO contract

This contract freezes the fairness boundary for the Pure PPO baseline and all later PPO-based comparisons.

- Observation: 34D, consisting of the existing 16D base state and 18 obstacle rays.
- Action: `(3,)`, world-frame normalized target velocity in `[-1, 1]^3`, passed through the existing shared M0 mapping.
- Environment: the existing CF2X `ObstacleVelocityAviary`, 48 Hz control, 960-step maximum episode, 0.30 m goal tolerance.
- Reset distribution: uniformly sampled S2 TRAIN tasks only during gradient training; VAL and TEST are never training resets.
- Reward: `2.0 * (d_prev - d_now) - 0.002`, plus exactly `+20` on success, `-20` on obstacle collision, or `-20` on ground contact.
- Termination: success, collision, ground contact, nonfinite state/action, or timeout.
- Policy: upstream Stable-Baselines3 `MlpPolicy`, actor and critic `34 -> 256 -> 256`.
- PPO hyperparameters: seed `20260812`, 500,000 environment steps, 8 environments, `learning_rate=3e-4`, `n_steps=1024`, `batch_size=512`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.20`, `ent_coef=0.0`, `vf_coef=0.5`, `max_grad_norm=0.5`, normalized advantage enabled.
- Observation normalization: none (`VecNormalize` is forbidden in S4).
- Checkpoint selection: VAL only, ordered by success count, unsafe count, mean return, then earlier step.
- TEST: only after 500,000 steps and model freeze, exactly once, never for selection.

The Pure PPO implementation has no runtime dependency on the Diffusion model, expert action, GCOPTER trajectory, recovery teacher, future waypoint, corridor, scene ID, or split ID as an observation feature.
