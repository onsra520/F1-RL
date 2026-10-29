# `src.training`

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

WP7 research factories additionally share equal `physics_config`, `race_config`
and `reset_config` registrations (and the existing offtrack/body/ramp contract).
When TRAIN reset preparation is enabled, the per-track `reset_track_id` must
match its scenario ID and hash-bound TRAIN registry. Rank and reset sequence
are preserved across environment recreation, independently of track-selection
and Gym seeds. These opt-ins keep policy spaces unchanged. Both runner
entrypoints still reject all research full-body factories before creating
models or artifacts; successful reset preparation does not grant admission.

`sampling_seed` exclusively controls selection. Each vector rank receives an independent stream derived by SHA-256 from the seed and rank. Gym/SB3 `reset(seed=...)` does not restart or alter that stream. `track_id` appears in reset and transition `info` (including terminal transitions), never in policy observations.

## Training orchestration

`TrainingRunConfig.from_yaml("config.yml")` loads the root orchestration input. `TrainingRunner.run_single(factory)` and `TrainingRunner.run_multi(scenarios)` build the existing vector environments, train PPO, save configured intermediate checkpoints, and write one self-contained run under `artifacts/runs/<run-id>/`.

Each run owns mutable `state.json`, a `checkpoints/` directory, and a self-describing final model archive. The state snapshot records requested and actual timesteps, resolved PPO/vector configuration, sampling metadata, latest checkpoint, and final-model path. Root `config.yml` remains immutable input.

State snapshots are written to `state.json.tmp` and atomically replaced into `state.json`, so an interrupted update leaves either the prior complete JSON or the new complete JSON. If final-model cleanup fails after a completed-state write failure, failed state records the cleanup error while leaving `final_model` unset. Checkpoint filenames record the environment interaction count when the SB3 progress callback runs during rollout collection; for example, `step25008.zip` is not a claim that PPO weights were optimized exactly through interaction 25,008.

This layer is framework-only: V1 has no training CLI, resume-from-checkpoint support, evaluation metrics, held-out-track scoring, TensorBoard/W&B, HPO, curriculum, or trainer service. `artifacts/policies/` remains for deliberately selected inference/demo policies and `artifacts/evaluations/` remains for the later evaluation milestone.

## Windows subprocess rule

Any executable entrypoint that constructs the `subproc` backend must use a `__main__` guard. V1 explicitly uses Windows-safe `spawn`; it does not assume Unix `fork` semantics.

## Diagnostics and future scope

`python -m scripts.diagnostics.benchmark_environments` is a manual diagnostic throughput benchmark. Its output is host-dependent evidence only; it has no CI threshold and does not choose a backend automatically.

`SubprocVecEnv` uses Windows-safe `spawn`; executable entrypoints must use a `__main__` guard. Captured scenarios and factories must remain serializable under spawn.

Long-running experiments, evaluation, unseen-track generalization, reward ablation, resume support, and experiment tooling remain later milestones.
