# F1-RL PPO model input contracts

`PPOModel` accepts a single `F1RLEnv` for a single-observation episode, or a registered SB3 `VecEnv` for vectorized observations and actions. The `PPOConfig` sets `n_steps`, `batch_size`, `n_epochs` and device; the selected rollout granularity is `n_envs * n_steps`.

Legacy vector tests demonstrate the public API, but neither the `F1RLEnv` nor `VecEnv` interface alone grants permission to run new Physics-v4 scientific PPO training. Research environments must pass the separate versioned scientific admission guard; prediction-only checkpoint loads are not training approval.

A requested 200000-step budget with eight environments and 256 rollout steps per environment implies 98 full rollouts and an actual final checkpoint at 200704. This arithmetic is a draft-only technical reconciliation, not a signed policy change.
