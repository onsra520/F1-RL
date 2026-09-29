# Multi-Track Sampling Design

**Status:** Approved conversational design; written specification for review.

**Branch:** `training/multi-track-sampling`

**Stacked base:** `models/ppo-baseline` at `19df1efae5cfa78162a73d998af798f4082ceec4`

## Goal

Train one PPO policy with vector slots that sample from an explicit finite set of authored training scenarios. The resulting infrastructure must make it possible to train against a distribution of tracks while preserving the existing deterministic simulation, Gym, reward, observation, and PPO contracts.

The milestone establishes training-scenario identity and deterministic, isolated per-rank selection. It does not measure policy quality or unseen-track generalization.

## Existing architecture

The current training layer is:

```text
F1RLEnvFactory
  -> one fresh, fixed-track F1RLEnv per vector slot
  -> DummyVecEnv or SubprocVecEnv(start_method="spawn")
  -> PPOModel
```

`F1RLEnvFactory` already holds every input required to construct a complete environment: `TrackWorld`, `WorldState`, `VehicleParameters`, `SimulationConfig`, `ObservationEncoder`, `RewardComposer`, and `EpisodeConfig`.

`F1RLEnv` constructs its `RaceRoute` and `Simulation` in its constructor. It is therefore fixed to one track for its complete lifetime. Stable-Baselines3 auto-resets a completed vector slot by calling that slot's `reset()` after it receives the terminal transition.

The sandbox and runtime do not import learning. Existing SB3 imports remain owned by `f1rl.learning.algorithms.ppo` and `f1rl.learning.training.vector_env`.

## Decision

Add reset-time scenario sampling in the training layer. Keep `F1RLEnv` fixed-track and compose it inside a private sampled Gym environment.

The public training surface gains:

```python
@dataclass(frozen=True, slots=True)
class TrainingScenario:
    track_id: str
    factory: F1RLEnvFactory


@dataclass(frozen=True, slots=True)
class TrainingScenarioSet:
    scenarios: tuple[TrainingScenario, ...]


def make_multi_track_vector_env(
    scenarios: TrainingScenarioSet,
    config: VectorEnvConfig,
    sampling_seed: int,
) -> VecEnv: ...
```

`TrainingScenario` deliberately reuses `F1RLEnvFactory` rather than repeating its seven construction fields. A scenario consequently includes track-specific initial state and any other construction input that must vary with the track.

`make_vector_env(factory, config)` remains unchanged as the explicit same-track mode. Multi-track construction is a separate function so callers cannot accidentally pass a factory where a pool is required, and the old API does not acquire an unrelated union parameter.

No new YAML configuration is introduced. The existing project has no trainer/orchestration entry point that would consume it. `sampling_seed` is a required explicit construction argument; adding an unused configuration field would only duplicate that value.

## Scenario-set contract

`TrainingScenario` validates that `track_id` is a non-blank string and that `factory` is an `F1RLEnvFactory`.

`TrainingScenarioSet` requires a tuple containing at least one scenario. It rejects duplicate `track_id` values. Declaration order is preserved and is the sampling population order.

Every scenario must expose an action space and observation space compatible with every other scenario. Compatibility means the Gym spaces have the same concrete type, shape, dtype, and bounds; both the action and observation spaces are checked. This is required because one SB3 `VecEnv` and one PPO policy bind to exactly one action space and one observation space.

The current `F1RLEnv` action space is uniform, but it is still checked rather than assumed. Different track geometry and initial world state are permitted. Observation encoders may differ only when their resulting spaces are compatible.

`track_id` is training metadata. It is never appended to, encoded into, or otherwise used to alter the policy observation.

## Reset-time sampling lifecycle

The vector module adds two private top-level, picklable types:

- a ranked callable that captures `TrainingScenarioSet`, `sampling_seed`, and vector rank;
- a Gym wrapper that owns one rank's sampler and its currently active inner `F1RLEnv`.

On every wrapper `reset()`:

1. The wrapper draws one scenario from its own sampler.
2. It calls that scenario's `factory.create()` to create a **new** `F1RLEnv`.
3. It resets the new inner environment and returns its observation plus reset info augmented with `"track_id"`.

An inner environment is never mutated to exchange tracks. A sampled episode always receives a newly constructed fixed-track `F1RLEnv`; the prior inner environment is closed before it is discarded. The wrapper delegates `step()` only to its active inner environment.

On `step()`, the wrapper adds `info["track_id"]` for the scenario that produced that transition. It does this before returning a terminal transition. Thus SB3 receives the terminal transition and its correct track ID before it adds `terminal_observation` and invokes the wrapper's next `reset()`. The later reset selects and reports the next episode's ID separately.

Because each wrapper owns its own sampler, an auto-reset in slot 0 selects only slot 0's next scenario. It cannot advance or otherwise affect slot 1's sequence.

## Sampling determinism

`sampling_seed` is required to be an integer; booleans are rejected. It is the exclusive owner of scenario-selection randomness.

Each rank derives its own initial sampler seed with this stable, versioned standard-library operation:

```python
int.from_bytes(
    hashlib.sha256(
        f"f1rl/multi-track-sampling/v1/{sampling_seed}/{rank}".encode("utf-8")
    ).digest(),
    "big",
)
```

The wrapper initializes one private `random.Random` instance from that derived integer and samples uniformly from the declared scenario tuple. Python `hash()` must not be used at any point in seed derivation because its process-randomized result is not stable.

`reset(seed=...)` follows the Gym contract for the wrapper and its inner environment, but it must not reseed, restart, replace, or otherwise alter the private scenario-selection stream. Gym/SB3 seed calls therefore cannot silently change the sequence selected by `sampling_seed`. Normal SB3 auto-resets pass no selection seed and continue each rank's existing stream.

This design promises reproducible scenario-selection sequences for the same scenario declaration, `sampling_seed`, rank, and Python/runtime version. It does not promise bit-for-bit PPO determinism across multiprocessing, PyTorch, or hardware.

## Backend and ownership rules

The existing `VectorEnvConfig` continues to select `DummyVecEnv` or `SubprocVecEnv` with explicit `spawn` for subprocesses. The ranked callable and all captured scenario values must remain cloudpickle-safe under Windows spawn. The design uses no mutable global registry, process-local lookup, or Unix fork behavior.

Every vector slot owns a distinct wrapper, sampler, active `F1RLEnv`, simulation, and race-progress state. Immutable scenario definitions may be shared by `DummyVecEnv` slots. `make_multi_track_vector_env()` returns a caller-owned `VecEnv`; callers must close it.

## Train/evaluation separation

Only a `TrainingScenarioSet` is accepted by the multi-track training constructor. The type and public API make the intended source explicit and avoid a generic dataset framework. Existing `assets/tracks/training` and `assets/tracks/evaluation` conventions remain in force: held-out evaluation assets are not supplied to a `TrainingScenarioSet`.

Later validation and unseen-test splitting may add separate scenario-set construction and evaluation APIs. This milestone does not add an asset loader, persistence format, evaluation runner, or enforcement beyond the explicit training-only construction boundary.

## Scope

In scope:

- immutable scenario and training-pool representation;
- strict ID and compatible action/observation-space validation;
- deterministic rank-isolated reset-time sampling;
- track identity in Gym info only;
- DummyVecEnv and Windows-spawn SubprocVecEnv construction;
- PPO smoke compatibility using the existing PPO configuration and model binding;
- regression coverage, training documentation, and optional training-package exports.

Out of scope:

- changes to sandbox, runtime, `F1RLEnv`, observation encoding, reward shaping, or PPO hyperparameters;
- procedural generation, track-editor/persistence work, curriculum learning, and adaptive difficulty;
- evaluation/generalization scoring, train/validation/test reports, or policy-quality claims;
- trainer services, callbacks, checkpoint scheduling, TensorBoard/W&B, and scaling claims.

## Verification requirements

Implementation must use test-first tasks and prove the following without asserting policy quality:

- empty pools, duplicate IDs, invalid fields, and incompatible action or observation spaces are rejected;
- a fixed `sampling_seed` produces the same sequence; known differing seeds can produce different sequences;
- every rank uses an isolated sequence, and resetting one rank does not advance another;
- `reset(seed=...)` does not change the scenario-selection sequence owned by `sampling_seed`;
- each reset creates a fresh, fixed-track inner `F1RLEnv` and never swaps an existing one;
- step and terminal-transition info identify the scenario that actually produced the transition;
- track ID remains absent from policy observations;
- same authored inputs retain deterministic transition behavior;
- multi-track DummyVecEnv and guarded Windows-spawn SubprocVecEnv smoke tests pass;
- PPO can complete a small CPU smoke run without algorithm or hyperparameter changes;
- existing same-track vector mode remains supported;
- all existing tests remain green, with full pytest, Python 3.12 compileall, dependency-boundary scan, `git diff --check`, and changed-scope inspection run on the final committed candidate.

## Alternatives rejected

**One fixed scenario per vector rank.** This is simpler but samples only at vector construction. Its distribution depends on slot count and does not provide episode-level training across the pool.

**Mutate `F1RLEnv` to replace its track at reset.** This would couple training policy to the Gym adapter and risks stale `RaceRoute`, simulation, and race-progress state. It conflicts with the fixed-track lifecycle already established by the baseline.

**A generic public sampler hierarchy or dataset framework.** The milestone needs one finite training pool and uniform deterministic selection. A private per-slot sampler is sufficient; broader abstractions belong to a future requirement.
