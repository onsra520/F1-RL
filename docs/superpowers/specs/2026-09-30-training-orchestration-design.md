# Training Orchestration Design

**Status:** Approved conversational design; written specification for review.

**Branch:** `models/ppo-baseline`

**Base HEAD at spec creation:** `a820ead54869fd23be8b3790931c86d56030b65b`

## Goal

Add a small, repeatable training-orchestration layer that connects the existing environment factories, vectorized environments, multi-track sampling, and PPO wrapper into one reproducible training run with explicit configuration and generated artifacts.

This milestone completes the framework before any long-running training experiment. It must support both same-track and multi-track PPO runs, checkpointing, run-state snapshots, cleanup, and end-to-end smoke verification without making policy-quality claims.

## Existing architecture

The project already provides:

```text
F1RLEnvFactory / TrainingScenarioSet
        -> make_vector_env() / make_multi_track_vector_env()
        -> Stable-Baselines3 VecEnv
        -> PPOModel
```

`PPOModel` owns the SB3 PPO adapter. `vector_env.py` owns SB3 vector-environment construction. These are intentionally the only production modules that directly import Stable-Baselines3.

The current `VectorTrainingConfig` only combines `PPOConfig` and `VectorEnvConfig`. It does not define run duration, checkpoint policy, artifact ownership, sampling seed, or run lifecycle. Existing lower-level configuration files under `configs/training/` remain valid lower-level inputs/reference files; orchestration adds one root-level `config.yml` as the run configuration source.

## Decision

Add a `TrainingRunner` class rather than a purely functional orchestrator or a job/executor system.

The public training surface gains approximately:

```python
@dataclass(frozen=True, slots=True)
class TrainingRunConfig: ...

@dataclass(frozen=True, slots=True)
class TrainingRunResult: ...

class TrainingRunner:
    def __init__(self, config: TrainingRunConfig) -> None: ...
    def run_single(self, factory: F1RLEnvFactory) -> TrainingRunResult: ...
    def run_multi(self, scenarios: TrainingScenarioSet) -> TrainingRunResult: ...
```

`TrainingRunner` owns run-level lifecycle and artifact creation. It does not own physics, track loading, evaluation, policy semantics, or SB3 implementation details.

No CLI is added in V1. The repository does not yet have a stable authored-track loader/persistence layer that a CLI could use without hard-coding scenarios.

## Run configuration

`F1-RL/config.yml` is the editable run input. Training never mutates this file. `TrainingRunConfig.from_yaml()` loads and validates it.

The resolved configuration contains:

- `PPOConfig`;
- `VectorEnvConfig`;
- `total_timesteps`, default `50_000`;
- `checkpoint_every_steps`, default `25_000`, or `None` to disable intermediate checkpoints;
- `sampling_seed`, default `0` and used only by multi-track sampling;
- `artifacts_root`, default `artifacts/runs`.

The YAML root contains exactly `training`, `ppo`, and `vector_env` mappings. The orchestration fields live under `training`:

```yaml
training:
  total_timesteps: 50000
  checkpoint_every_steps: 25000
  sampling_seed: 123
  artifacts_root: artifacts/runs

ppo:
  # existing PPOConfig fields

vector_env:
  # existing VectorEnvConfig fields
```

Unknown keys are rejected rather than ignored.

Relative artifact paths are resolved from the directory containing `config.yml`, so the checked-in configuration remains portable and does not require machine-specific absolute paths.

`total_timesteps` must be a positive integer. `checkpoint_every_steps` must be `None` or a positive integer. Boolean values are rejected for integer fields. Existing `PPOConfig` and `VectorEnvConfig` validation remains authoritative for their fields.

The training mode is not stored as an input switch. Calling `run_single()` or `run_multi()` selects the mode explicitly. Multi-track `track_id` values are derived from `TrainingScenarioSet`; the user must not duplicate them manually in YAML.

For single-track V1, `F1RLEnvFactory` has no track identifier, so run state does not invent one. Single-track state records no authored track IDs; a future asset-loader milestone may provide stable track identity.

`ppo.seed=None` remains legal because `PPOConfig` already permits it. Artifact naming renders that value explicitly as `seednone` rather than implying determinism.

## Run identity and artifact layout

Run IDs are generated to second precision:

```text
YYYY-MM-DD_HHMMSS_ppo_singletrack_seed42
YYYY-MM-DD_HHMMSS_ppo_multitrack_seed42_sample123
```

Multi-track naming always includes both the PPO seed and scenario-sampling seed because they control different randomness sources. Single-track naming contains only the PPO seed.

Each run owns one directory:

```text
artifacts/runs/
└── 2026-09-30_010500_ppo_multitrack_seed42_sample123/
    ├── state.json
    ├── checkpoints/
    │   └── ppo-multitrack-seed42-sample123-step25008.zip
    └── ppo-multitrack-seed42-sample123-final-model.zip
```

A run directory must never silently overwrite or reuse an existing directory. A generated-ID collision fails before training begins.

`artifacts/policies/` remains reserved for policies deliberately selected/exported for inference or demos. `artifacts/evaluations/` remains reserved for the later evaluation milestone. Training-run checkpoints and final training outputs live under `artifacts/runs/`.

`artifacts/runs/*` is generated output and is ignored by Git. A tracked `artifacts/runs/README.md` may document the directory convention, matching the repository's existing artifact-directory pattern.

## Runtime state snapshot

`state.json` is generated and mutable. It records the resolved run state independently from the editable root `config.yml`, so changing the root configuration later does not erase how an older run was executed.

The snapshot includes at least:

- `run_id`;
- `status`: `running`, `completed`, or `failed`;
- `mode`: `singletrack` or `multitrack`;
- `requested_total_timesteps`;
- `actual_total_timesteps`;
- `checkpoint_every_steps`;
- `latest_checkpoint`;
- `final_model`;
- resolved PPO configuration;
- resolved vector-environment configuration;
- `sampling_seed` for multi-track runs, otherwise `null`;
- derived training track IDs for multi-track runs, otherwise an empty list.

A new run writes `status="running"` before environment/model construction begins. `state.json` is updated after each checkpoint so `actual_total_timesteps` and `latest_checkpoint` reflect persisted progress. Successful completion writes `status="completed"` only after the final model has been saved. Any failure after the run directory exists makes a best-effort update to `status="failed"`, capturing the latest available timestep count when possible, and then re-raises the original exception.

V1 does not store stack traces or error logs in `state.json`; exceptions remain the caller's responsibility. Existing checkpoints are preserved on failure and a failed run is not automatically deleted.

## Timestep semantics and checkpoint policy

Stable-Baselines3 PPO collects complete rollouts. Therefore `total_timesteps` is a requested minimum, not a promise that the final counter equals that exact value.

For example, with `n_steps=256` and `n_envs=8`, one rollout contributes 2,048 timesteps. A request for 50,000 timesteps may finish at 51,200. The run records both values:

```text
requested_total_timesteps = 50000
actual_total_timesteps    = 51200
```

Checkpoint thresholds are generated as:

```text
interval, 2 * interval, 3 * interval, ... < requested_total_timesteps
```

A threshold equal to the requested final target is excluded, so the final point is represented only by the final-model artifact rather than a duplicate checkpoint.

When PPO first crosses a threshold, the checkpoint filename uses the actual `PPOModel.num_timesteps` value at which the save occurs. A target of 25,000 may therefore produce `...-step25008.zip`. If one progress update crosses more than one threshold, all crossed thresholds are considered satisfied but only one archive is written for that actual timestep.

With `checkpoint_every_steps=None`, no intermediate checkpoint directory entries are produced; the final model is still saved on successful completion.

## PPO progress hook and SB3 boundary

`runner.py` must not import Stable-Baselines3. The existing production import boundary remains:

```text
f1rl/learning/algorithms/ppo.py
f1rl/learning/training/vector_env.py
```

`PPOModel.learn()` gains a small project-owned optional progress/checkpoint hook surface. `ppo.py` adapts that hook to the SB3 callback mechanism internally. The training runner only observes project-level timestep progress and invokes `PPOModel.save()` when a configured threshold is crossed.

The hook is not an evaluation callback, logging framework, or general plugin system. It exists only to expose training progress needed for run-state/checkpoint orchestration while keeping SB3 details inside the established adapter boundary.

An exception raised while checkpointing or updating run state aborts the run and follows the same failed-run lifecycle as any other training exception.

## Runner lifecycle

A successful run performs:

```text
load + validate config
-> derive run identity
-> create run directory/checkpoints directory
-> write state.json(status="running")
-> create VecEnv
-> create PPOModel
-> learn with progress/checkpoint hook
-> save final model
-> write state.json(status="completed")
-> close VecEnv
-> return TrainingRunResult
```

If failure occurs after the run directory exists:

```text
exception
-> best-effort state.json(status="failed")
-> close VecEnv if it was created
-> re-raise the original exception
```

`VecEnv.close()` runs from a `finally` path. Cleanup failure must not replace an already-active training exception. There is no automatic retry, resume, rollback, or failed-run deletion in V1. A final model is never written when learning did not complete successfully.

`TrainingRunResult` is returned only for completed runs and contains:

- `run_id`;
- `run_directory`;
- `final_model_path`;
- all checkpoint paths written by the run;
- requested total timesteps;
- actual total timesteps.

## Single-track and multi-track construction

`run_single(factory)` uses the existing `make_vector_env(factory, config.vector_env)`. Its behavior remains same-track across all vector slots.

`run_multi(scenarios)` uses `make_multi_track_vector_env(scenarios, config.vector_env, config.sampling_seed)`. It does not reconstruct, mutate, or weaken the existing `TrainingScenarioSet` contract. Scenario IDs recorded in state are derived directly from declaration order.

Both paths share the same runner lifecycle, PPO creation, progress handling, artifact naming, failure semantics, and final result type. There is no union-style environment constructor and no runtime mutation between single-track and multi-track modes.

## Module layout

Production layout stays intentionally small:

```text
f1rl/learning/
├── algorithms/
│   └── ppo.py
└── training/
    ├── config.py
    ├── env_factory.py
    ├── vector_env.py
    ├── runner.py
    └── __init__.py
```

`config.py` owns `TrainingRunConfig` alongside the existing vector configuration. `runner.py` owns `TrainingRunner`, `TrainingRunResult`, run naming, state-file writes, checkpoint schedule handling, and lifecycle cleanup. Private helpers remain private unless a later requirement needs a public abstraction.

Do not add `job.py`, `executor.py`, `manager.py`, `service.py`, or `artifact_store.py` in V1. The course project does not need a trainer platform.

## Test isolation and test-only doubles

Production code must not contain mocks, fake implementations, test flags, or test-only branches.

Test doubles live only under tests, for example:

```text
tests/learning/training/
├── test_run_config.py
├── test_training_runner.py
└── mocks/
    ├── mock_ppo.py
    ├── mock_vec_env.py
    └── mock_factories.py
```

Any seam used by tests must also be a legitimate production abstraction or injectable callable; there must be no `if testing:` behavior in `f1rl/`.

Unit/runner coverage must prove:

- strict YAML/config validation;
- deterministic run/artifact naming from explicit inputs and injected time;
- correct checkpoint-threshold generation and final-threshold exclusion;
- actual timestep values are used in checkpoint names and state;
- single-track and multi-track builders are selected correctly;
- state transitions from `running` to `completed` or `failed`;
- checkpoints already written before failure remain referenced/preserved where possible;
- failed runs do not create a final model;
- `VecEnv.close()` runs on success and failure;
- the original training exception remains observable when cleanup also fails;
- no run directory is silently overwritten.

At least one integration test must use the real `PPOModel` and a real vector environment with a very small CPU training budget. It verifies the complete orchestration path and generated artifacts, not policy quality. Windows subprocess coverage remains guarded with `__main__` when `SubprocVecEnv` is exercised.

## Scope

In scope:

- root `config.yml` orchestration input;
- immutable `TrainingRunConfig`;
- `TrainingRunner` and `TrainingRunResult`;
- same-track and multi-track orchestration paths;
- generated run IDs and self-describing model/checkpoint filenames;
- `state.json` lifecycle snapshots;
- configurable intermediate checkpointing with a default interval of 25,000 timesteps;
- final-model saving;
- PPO progress-hook plumbing inside `ppo.py` while preserving the SB3 import boundary;
- lifecycle/error cleanup and tests;
- a tiny real-PPO integration smoke run.

Out of scope:

- long-running training experiments or policy-quality claims;
- resume-from-checkpoint support;
- evaluation metrics, held-out-track scoring, or unseen-track generalization experiments;
- reward-shaping ablations;
- TensorBoard, Weights & Biases, HPO, curriculum learning, or adaptive difficulty;
- trainer services, queues, job systems, REST/WebSocket APIs, or background workers;
- track persistence/loaders or procedural generation;
- CLI training entrypoints;
- changes to sandbox physics, observations, rewards, action semantics, race progress, or the fixed-track `F1RLEnv` invariant.

## Verification requirements

Implementation must follow TDD and finish with fresh evidence on the exact final commit. Verification must include:

- focused unit tests for configuration, naming, checkpoint scheduling, state transitions, and failure cleanup;
- a real CPU PPO orchestration smoke test with generated run artifacts;
- regression coverage for existing same-track and multi-track vector behavior;
- full pytest suite;
- `python -m compileall -q f1rl scripts`;
- a production dependency-boundary scan proving SB3 imports remain limited to `ppo.py` and `vector_env.py`;
- a production scan proving no direct PyTorch imports were introduced;
- `git diff --check`;
- final `git status --short --branch` and changed-scope review.

The current environment uses CPU-only PyTorch. This milestone must not claim CUDA training works on the current machine merely because `PPOConfig(device="cuda")` remains architecturally supported.

## Alternatives rejected

**Pure functional orchestration.** A pair of top-level run functions would be smaller initially, but run identity, state persistence, checkpoint scheduling, lifecycle cleanup, and future extension would become scattered helper state. A small `TrainingRunner` keeps these responsibilities together without creating a service framework.

**Job/executor architecture.** Jobs, queues, lifecycle services, and command handlers are unnecessary for a course project that is not yet running remote or concurrent training workloads.

**Put SB3 callbacks directly in `runner.py`.** This would break the existing dependency boundary and couple the training orchestration layer to SB3 internals. The PPO adapter remains the sole owner of algorithm-specific callback plumbing.

**Store generated state in root `config.yml`.** Mutating user input would destroy the distinction between requested configuration and runtime outcome. Root YAML stays immutable; each run owns generated `state.json`.

**Duplicate the final checkpoint.** Saving both a checkpoint at the requested final threshold and a final-model archive creates redundant artifacts. Checkpoint thresholds are therefore strictly less than the requested final target.
