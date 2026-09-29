# `f1rl.learning.training`

## Purpose

Coordinates repeatable RL training inputs without owning physics, rewards, policy evaluation, or experiment claims.

## Vectorized PPO baseline

The implemented vector layer uses one **same-track** `F1RLEnvFactory` to create fresh, independent environment slots.

`VectorEnvConfig` selects an explicit backend:

- `dummy` — `DummyVecEnv` in one process;
- `subproc` — `SubprocVecEnv` with `spawn`.

Checked-in defaults live in `configs/training/vectorized-ppo.yml` and start at eight `dummy` environments. That value is a deterministic baseline, not an optimum.

`make_vector_env()` returns an SB3 VecEnv. The caller owns its lifecycle and must call `close()`.

## Windows subprocess rule

Any executable entrypoint that constructs the `subproc` backend must use a `__main__` guard. V1 explicitly uses Windows-safe `spawn`; it does not assume Unix `fork` semantics.

## Diagnostics and future scope

`scripts/benchmark_vector_envs.py` is a manual diagnostic throughput benchmark. Its output is host-dependent evidence only; it has no CI threshold and does not choose a backend automatically.

Vectorized PPO currently duplicates the same authored track across slots. Multi-track sampling, curriculum, long-running trainer orchestration, callbacks, evaluation, and generalization experiments remain later milestones.
