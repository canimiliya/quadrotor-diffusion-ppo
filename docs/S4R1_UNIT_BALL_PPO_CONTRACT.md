# S4-R1 Unit-Ball Pure PPO Contract

S4-R1 repairs only the Pure PPO action distribution exposed to SB3. The
environment, reward, observation, scene geometry, train/VAL/TEST split, PPO
hyperparameters, seed, and 500,000-step budget remain the S4-R0 contract.

The policy samples an explicit latent

```text
u ~ Normal(mu, diag(sigma^2))
a = u * tanh(||u||) / ||u||
```

with a stable zero-radius branch. Its inverse is

```text
rho = ||a||
r = atanh(clamp(rho, 0, 1-eps))
u = a * r / max(rho, eps)
```

For the three-dimensional radial transform, `log_prob(a)` is computed as the
Gaussian latent log probability minus

```text
log(sech^2(r)) + 2 * (log(tanh(r)) - log(r))
```

The custom distribution subclasses SB3's diagonal Gaussian distribution, so
the stock `stable_baselines3.PPO` algorithm is unchanged. `forward()`,
`predict()`, rollout storage, and `evaluate_actions()` all use the same
transformed action and inverse log probability. The environment's shared
`map_normalized_velocity()` remains the only physical mapping; a valid policy
action therefore reaches the simulator without a second projection.

Required oracle checks are forward support, forward/inverse round trip,
autograd Jacobian agreement, finite deterministic log probability, and SB3
rollout/evaluation consistency. TEST is executed exactly once only after the
VAL usability gate passes.
