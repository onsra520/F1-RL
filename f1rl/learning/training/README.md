# `f1rl.learning.training`

## Purpose

Coordinates repeatable RL training inputs without owning physics, rewards, policy evaluation, or experiment claims.

## Vectorized PPO baseline

`make_vector_env(factory, config)` keeps the existing same-track behavior: it uses one `F1RLEnvFactory` to create fresh, independent environment slots.

`VectorEnvConfig` selects an explicit backend:

- `dummy` — `DummyVecEnv` in one process;
- `subproc` — `SubprocVecEnv` with `spawn`.

Checked-in defaults live in `configs/training/vectorized-ppo.yml` and start at eight `dummy` environments. That value is a deterministic baseline, not an optimum.

`make_vector_env()` returns an SB3 VecEnv. The caller owns its lifecycle and must call `close()`.

## Multi-track sampling

Use `TrainingScenario(track_id, factory)` and a non-empty `TrainingScenarioSet` with `make_multi_track_vector_env(scenarios, config, sampling_seed)` to sample authored training tracks at episode resets. The set requires unique IDs, one value-equal non-track contract across all factories (`vehicle_parameters`, `simulation_config`, `observation_encoder`, `reward_composer`, and `episode_config`), and compatible canonical action and observation spaces. Only track, track-dependent initial state, and ID vary.

The wrapper exposes canonical spaces before its first reset and does not select an episode during construction. Each reset creates a new fixed-track `F1RLEnv`; the prior environment is closed first, and a failed new reset closes the partial environment and propagates the error. Wrapper `close()` closes its active environment.

`sampling_seed` exclusively controls selection. Each vector rank receives an independent stream derived by SHA-256 from the seed and rank. Gym/SB3 `reset(seed=...)` does not restart or alter that stream. `track_id` appears in reset and transition `info` (including terminal transitions), never in policy observations.

## Windows subprocess rule

Any executable entrypoint that constructs the `subproc` backend must use a `__main__` guard. V1 explicitly uses Windows-safe `spawn`; it does not assume Unix `fork` semantics.

## Diagnostics and future scope

`scripts/benchmark_vector_envs.py` is a manual diagnostic throughput benchmark. Its output is host-dependent evidence only; it has no CI threshold and does not choose a backend automatically.

`SubprocVecEnv` uses Windows-safe `spawn`; executable entrypoints must use a `__main__` guard. Captured scenarios and factories must remain serializable under spawn.

Curriculum, long-running trainer orchestration, callbacks, evaluation, and generalization experiments remain later milestones.
