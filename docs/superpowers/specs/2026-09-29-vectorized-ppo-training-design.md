# Vectorized PPO Training Design

**Status:** Approved conversational design; written specification for review.

**Branch:** `training/vectorized-ppo`

**Stacked base:** `models/ppo-baseline` at `3c29ad340d445ab4458c3e3c0b9b29fd647aa4a1`

## Goal

Add deterministic vectorized F1-RL training support so one Stable-Baselines3 PPO policy can collect rollouts from multiple independent `F1RLEnv` instances.

The milestone proves correct same-track parallelism, explicit `DummyVecEnv` / `SubprocVecEnv` selection, Windows-safe subprocess startup, and PPO rollout/batch validation.

It does **not** claim that more environments improve policy quality or that any particular environment count is fastest.

## Architectural position

```text
F1RLEnv
   -> F1RLEnvFactory
   -> SB3 VecEnv (Dummy or Subproc)
   -> PPOModel
   -> future multi-track sampling / trainer / evaluation
```

The sandbox, deterministic simulation, reward, episode, and Gym contracts remain unchanged.
## Design principles

- One PPO policy is shared across all vectorized environments.
- Every vector slot owns an independent `F1RLEnv` instance and independent `Simulation` state.
- V1 vectorization uses the same authored `TrackWorld` and same immutable constructor inputs for every slot.
- Track sampling/randomization is deferred to a separate milestone.
- Backend selection is explicit, never auto-selected from host hardware.
- No F1-RL layer reimplements SB3 vector-environment semantics.
- The caller owns `VecEnv.close()` lifecycle.

## New production files

```text
f1rl/learning/training/
├── __init__.py
├── config.py
├── env_factory.py
├── vector_env.py
└── README.md

configs/training/
└── vectorized-ppo.yml

scripts/
└── benchmark_vector_envs.py
```

The existing `f1rl/learning/algorithms/ppo.py` is widened only enough to accept vectorized training environments and validate rollout sizing.
## Configuration model

Do not modify `configs/training/ppo-baseline.yml`; its existing strict `PPOConfig.from_yaml()` contract remains authoritative.

Create a separate combined configuration:

```yaml
ppo:
  learning_rate: 0.0003
  n_steps: 256
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  seed: 0
  device: cpu

vector_env:
  n_envs: 8
  backend: dummy
  start_method: spawn
```

The checked-in configuration is deterministic and editable. It is not a benchmark-tuned recommendation.
## `VectorEnvConfig`

```python
@dataclass(frozen=True, slots=True)
class VectorEnvConfig:
    n_envs: int = 8
    backend: str = "dummy"
    start_method: str = "spawn"
```

Validation:

- `n_envs` is an integer `>= 1`; booleans are rejected.
- `backend` is exactly `"dummy"` or `"subproc"`.
- `start_method` is a non-empty string.
- For `backend="subproc"`, V1 accepts only `start_method="spawn"`.
- For `backend="dummy"`, `start_method` is stored for config round-trip but has no runtime effect.
- No automatic clamp to CPU/logical-core count.
- `n_envs=1` is valid for both correctness comparison and baseline measurements.

The default `n_envs=8` is a conservative starting point, not a claim that eight environments are optimal.
## `VectorTrainingConfig`

```python
@dataclass(frozen=True, slots=True)
class VectorTrainingConfig:
    ppo: PPOConfig
    vector_env: VectorEnvConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> VectorTrainingConfig:
        ...
```

YAML rules:

- path must be a non-blank `str` or `Path` resolving to a file;
- document root must be a mapping with exactly `ppo` and `vector_env` mappings;
- unknown top-level keys are rejected;
- unknown keys inside each section are rejected;
- omitted fields inside `ppo` and `vector_env` use their dataclass defaults;
- malformed YAML, missing files, empty files, or wrong mapping shapes raise project-owned `ValueError`;
- PPO field validation delegates to `PPOConfig` rather than duplicating its numeric rules.

This loader does not mutate `PPOConfig.from_yaml()` and therefore preserves the standalone PPO baseline contract.
## Environment factory

```python
@dataclass(frozen=True, slots=True)
class F1RLEnvFactory:
    track: TrackWorld
    initial_state: WorldState
    vehicle_parameters: VehicleParameters
    simulation_config: SimulationConfig
    observation_encoder: ObservationEncoder
    reward_composer: RewardComposer
    episode_config: EpisodeConfig

    def create(self) -> F1RLEnv:
        ...
```

Validation mirrors the concrete `F1RLEnv` constructor types at factory construction time.

Every `create()` call returns a **new** `F1RLEnv`. Immutable track/config/domain values may be shared as constructor inputs, but no `Simulation`, race progress, Gym lifecycle state, or mutable environment instance is shared.

`F1RLEnvFactory` does not own seeding policy, track sampling, curriculum, or model configuration.
## Picklable ranked factory wrapper

`vector_env.py` owns one internal immutable callable:

```python
@dataclass(frozen=True, slots=True)
class _RankedEnvFactory:
    factory: F1RLEnvFactory
    rank: int

    def __call__(self) -> F1RLEnv:
        return self.factory.create()
```

Rules:

- ranks are assigned exactly `0..n_envs-1`;
- rank is identity/diagnostic metadata only in V1 and does not change track or initial state;
- each wrapper is a distinct callable object;
- no anonymous closure is required by the public implementation contract;
- the wrapper and all captured factory state must be cloudpickle/spawn compatible.

Future per-rank seeds or track sampling may use `rank`; this milestone does not invent those behaviors early.
## Vector environment construction

Public API:

```python
def make_vector_env(
    factory: F1RLEnvFactory,
    config: VectorEnvConfig,
) -> VecEnv:
    ...
```

Validation requires concrete `F1RLEnvFactory` and `VectorEnvConfig` instances.

For `backend="dummy"`:

```text
n_envs ranked callables
-> stable_baselines3.common.vec_env.DummyVecEnv
```

For `backend="subproc"`:

```text
n_envs ranked callables
-> SubprocVecEnv(..., start_method="spawn")
```

No custom `VecEnv` subclass is introduced.
## VecEnv semantics

F1-RL delegates vector lifecycle to Stable-Baselines3:

- `VecEnv.reset()` returns a batched observation array;
- `VecEnv.step(actions)` accepts one action per slot;
- `dones` represent SB3's combined termination/truncation vector;
- completed slots are automatically reset by SB3's vector wrapper;
- the final observation before auto-reset is exposed as `info["terminal_observation"]`;
- truncation metadata added by SB3 is preserved;
- sibling slots remain independently stepped/reset according to their own state.

Do not change `F1RLEnv.reset()` or `F1RLEnv.step()` to imitate SB3's vector API.

Returned batched shapes for N environments are:

```text
observations: (N, observation_size)
actions:      (N, 2)
rewards:      (N,)
dones:        (N,)
infos:        sequence length N
```

All observations/actions retain the existing NumPy `float32` element contract.
## Seeding and determinism

This milestone does not add a second vector-specific seed setting.

The existing `PPOConfig.seed` remains the algorithm seed. When SB3 binds a `VecEnv`, its normal seeding behavior may seed vector slots for the next reset.

F1-RL V1 environment dynamics remain deterministic for equal inputs and currently contain no stochastic track or reset sampling.

Required guarantees:

- repeated `F1RLEnvFactory.create()` calls start from equal authored state but own independent runtime state;
- equal batched actions applied to equal vector slots produce equal transitions;
- mutating/stepping one concrete env instance cannot mutate another instance;
- backend choice must not alter F1-RL physics/reward/episode semantics.

This milestone does not promise identical wall-clock ordering or bit-for-bit PPO optimization across multiprocessing schedules.
## Windows subprocess contract

`SubprocVecEnv` V1 uses `start_method="spawn"` explicitly.

Any executable script that constructs a subprocess vector environment must guard construction with:

```python
if __name__ == "__main__":
    main()
```

This is a caller/entrypoint requirement, not something `make_vector_env()` can safely synthesize.

Automated subprocess acceptance must therefore execute a dedicated guarded helper script in a child Python process rather than constructing `SubprocVecEnv` directly inside an unguarded pytest collection context.

The subprocess smoke test must:

- create at least two vector slots;
- reset and step successfully;
- report expected batched shapes;
- call `close()` in `finally`;
- exit with code zero;
- leave no F1-RL-owned worker intentionally running after the helper exits.

No Unix-only `fork` assumption is allowed in production or tests.
## PPO training-environment widening

`PPOModel` currently accepts only `F1RLEnv`. This milestone widens the accepted training environment to:

```python
PPOTrainingEnv = F1RLEnv | VecEnv
```

Conceptual signatures become:

```python
@classmethod
def create(cls, env: PPOTrainingEnv, config: PPOConfig) -> PPOModel:
    ...

@classmethod
def load(
    cls,
    path: str | Path,
    *,
    env: PPOTrainingEnv | None = None,
    device: str = "cpu",
) -> PPOModel:
    ...
```

No generic Gym environment type is accepted; the supported boundary remains the project's concrete `F1RLEnv` or an SB3 `VecEnv` built for training.
## PPO rollout and batch-size validation

The single-environment baseline currently validates `batch_size` against `n_steps` inside `PPOConfig`. That relationship becomes environment-dependent under vectorization.

Change the standalone `PPOConfig` contract to validate only:

- `n_steps >= 2`;
- `batch_size >= 2`;
- all existing numeric/range/device rules remain unchanged.

Do **not** require `batch_size <= n_steps` or `n_steps % batch_size == 0` during dataclass construction.

When a PPO model is bound to an environment, calculate:

```text
num_envs = 1 for F1RLEnv
num_envs = env.num_envs for VecEnv
rollout_size = n_steps * num_envs
```

Binding is valid only when:

```text
batch_size <= rollout_size
rollout_size % batch_size == 0
```

Invalid combinations fail with project-owned `ValueError` before SB3 model construction or continued training begins.
## Validation on create and load

`PPOModel.create()` validates rollout sizing from the supplied `PPOConfig` before constructing SB3 PPO.

`PPOModel.load(..., env=...)` validates after loading the serialized backend because `n_steps` and `batch_size` come from the archive. If the loaded hyperparameters are incompatible with the supplied environment count, raise `ValueError` and do not begin learning.

`PPOModel.load(..., env=None)` remains inference-only and does not perform environment-count validation.

The explicit CUDA no-fallback contract remains unchanged for both create and load.

`learn(total_timesteps)` keeps cumulative semantics with `reset_num_timesteps=False`. With vector environments, SB3 may advance `num_timesteps` in multiples of `n_envs * n_steps`; the project contract remains that a successful call advances by at least the requested amount.

## Prediction remains single-observation

`PPOModel.predict()` keeps its existing V1 contract:

```text
one observation shaped (observation_size,)
-> one action shaped (2,)
```

Vectorized training does not widen the public inference wrapper to batched prediction in this milestone. Direct SB3 internals may handle batches during learning.
## Public package surface

`f1rl.learning.training` is an optional RL-training package and may depend on Stable-Baselines3.

Public exports:

```python
from f1rl.learning.training import (
    F1RLEnvFactory,
    VectorEnvConfig,
    VectorTrainingConfig,
    make_vector_env,
)
```

Do **not** re-export these symbols from `f1rl.learning`; importing the base learning package must not require the optional RL stack.

`_RankedEnvFactory` remains internal.

The existing `f1rl.learning.algorithms` public surface remains `PPOConfig` and `PPOModel`.
## Dependency boundaries

Allowed direct Stable-Baselines3 imports in production after this milestone:

```text
f1rl.learning.algorithms.ppo
f1rl.learning.training.vector_env
```

No direct `torch` import is added anywhere in F1-RL production code.

Still forbidden from importing SB3 or Torch:

- `f1rl.sandbox/**`;
- `f1rl.runtime/**`;
- `f1rl.learning.gym_environment`;
- observation/action/reward/episode modules.

`env_factory.py` should depend only on existing F1-RL domain/runtime/learning types, not SB3.

`config.py` may depend on `PPOConfig` and PyYAML but owns no model/runtime construction.

The base package remains usable without importing `f1rl.learning.training`.
## Diagnostic throughput benchmark

Create `scripts/benchmark_vector_envs.py` as a manual diagnostic, not a CI performance gate.

It must be Windows-spawn safe with a `__main__` guard and accept explicit backend/environment-count choices.

Recommended measurement ladder:

```text
DummyVecEnv:   1, 4, 8, 16
SubprocVecEnv: 4, 8, 16
```

The script reports at least:

```text
backend
n_envs
transitions
elapsed_seconds
transitions_per_second
```

It must close the vector environment in `finally`.

No threshold, winner, automatic backend selection, or committed benchmark result is part of correctness. Host-dependent throughput observations are evidence for later tuning only.
## Correctness acceptance

Focused tests must prove at minimum:

1. `VectorEnvConfig` defaults are exactly `n_envs=8`, `backend="dummy"`, `start_method="spawn"`.
2. Invalid counts/types/backends/start methods fail before SB3 construction.
3. `VectorTrainingConfig.from_yaml()` loads checked-in defaults and rejects malformed/unknown mappings.
4. Existing `PPOConfig.from_yaml("configs/training/ppo-baseline.yml")` behavior remains unchanged.
5. `F1RLEnvFactory.create()` returns distinct environment instances with equal initial authored state.
6. Stepping/resetting one created env does not mutate another created env.
7. `DummyVecEnv` construction creates exactly `n_envs` distinct slots.
8. Batched reset/step shapes match the declared vector contract.
9. Equal slots plus equal batched actions produce equal rewards/dones/observations.
10. Auto-reset exposes `terminal_observation` when a slot ends.
11. `close()` succeeds for both backends.
12. Windows guarded subprocess smoke for `SubprocVecEnv` exits successfully.
13. `PPOConfig` accepts vector-valid combinations where `batch_size > n_steps` but fits total rollout size.
14. Binding a single env still enforces the original effective rollout constraint through `n_steps * 1`.
15. Binding a vector env rejects `batch_size > n_steps * n_envs`.
16. Binding rejects rollout sizes not divisible by `batch_size`.
17. `PPOModel.create()` accepts real `DummyVecEnv` and preserves fixed `MlpPolicy`/device semantics.
18. Real vectorized PPO smoke learning advances `num_timesteps` by at least the requested amount.
19. Save/load with a compatible vector env supports continued cumulative learning.
20. Inference-only load remains unchanged.
21. Public training exports are exactly the approved surface.
22. `f1rl.learning` does not import/re-export optional training symbols.
23. Only approved production modules directly import SB3.
24. No F1-RL production module imports Torch.
25. Existing single-environment PPO tests remain green after contract widening.
26. Existing `F1RLEnv`, sandbox, runtime, reward, observation, action, and episode tests remain green.
27. Python 3.12 compileall succeeds.
28. `git diff --check` succeeds.

Performance measurements are explicitly excluded from pass/fail assertions.
## Test layout

Expected focused tests:

```text
tests/learning/training/
├── test_vector_config.py
├── test_env_factory.py
└── test_vector_env.py

tests/learning/algorithms/
└── test_ppo_model.py                 # widen existing binding tests

tests/integration/
├── test_vectorized_ppo_integration.py
└── helpers/subproc_vector_smoke.py   # guarded executable helper
```

Use real `F1RLEnv` fixtures for integration behavior. Mocking SB3 is acceptable only for narrow pre-backend validation tests where real process/model construction would obscure the contract.

Subprocess acceptance must use the same Python interpreter running pytest so the installed editable package/dependencies are consistent.
## Documentation updates

Update `f1rl/learning/training/README.md` to describe:

- same-track vectorized training;
- `F1RLEnvFactory` ownership;
- explicit `dummy` versus `subproc` backend selection;
- Windows `spawn` / `__main__` requirement;
- vector config YAML;
- caller-owned `close()` lifecycle;
- throughput benchmark as diagnostic only.

Update `f1rl/learning/algorithms/README.md` to state that PPO may train against one `F1RLEnv` or an SB3 `VecEnv`, while public prediction remains single-observation.

Update `f1rl/learning/README.md` to mark vectorized PPO support implemented and keep multi-track sampling/trainer/evaluation as future layers.

Do not claim that 8/16/100 environments are optimal or that vectorization improves final policy quality.
## Explicit non-goals

This milestone does not implement:

- different tracks per vector slot;
- random track selection on reset;
- curriculum schedules;
- a long-running trainer service/class;
- callbacks, TensorBoard, W&B, checkpoint schedules, or training-time evaluation;
- hyperparameter search;
- automatic backend or `n_envs` selection;
- CPU/GPU wheel installation policy;
- a 100-process default;
- policy-quality, convergence, or generalization claims;
- batched public `PPOModel.predict()`;
- changes to sandbox/runtime physics or Gym environment semantics.

The existing RTX 4070 host is not part of the software contract; current local PyTorch may still be CPU-only.
## Failure semantics

Project-owned malformed configuration/factory/model-binding inputs raise `ValueError` before expensive SB3/process work where possible.

Backend/process failures from `DummyVecEnv`, `SubprocVecEnv`, multiprocessing spawn, PPO learning, or serialization propagate with their original diagnostics; F1-RL must not convert them into fake success.

If a `SubprocVecEnv` construction or smoke operation fails after workers exist, test/scripts must attempt `close()`/process cleanup in `finally` where an object was successfully created.

A vector environment returned successfully is caller-owned. F1-RL does not register global shutdown hooks or hidden background workers.

## Verification

Before the branch is considered complete, run fresh under Python 3.12:

```text
focused training/vector-env tests
existing + widened PPO tests
DummyVecEnv real training smoke
Windows SubprocVecEnv guarded helper smoke
full pytest suite
compileall f1rl
source dependency scans
git diff --check
```

Inspect changed production scope and confirm no modifications to sandbox, runtime, Gym environment, action, observation, reward, or episode production modules.
## Post-milestone sequence

After vectorized PPO is verified, the next architectural layer is deliberately separate:

```text
PPO baseline
-> vectorized PPO                 # this milestone
-> multi-track sampling
-> repeatable training orchestration
-> independent evaluation
-> unseen-track generalization
-> reward-shaping ablation
```

Multi-track work may later replace the single shared factory with a controlled factory/sampler set, but this milestone must not pre-build that abstraction.

## Branch integration note

`training/vectorized-ppo` is intentionally stacked on unmerged `models/ppo-baseline`.

Do not merge this branch directly into `main` while the PPO baseline commits are absent there unless the human explicitly chooses to integrate the complete stacked history.

Keeping the branch stacked allows PPO baseline review/integration to remain an independent decision.
## Diagnostic benchmark fixture

Because the repository does not yet have a production track-loader/trainer abstraction, the manual benchmark script may construct one minimal deterministic same-track `F1RLEnvFactory` locally in the script using existing public domain types.

That fixture is diagnostic setup only; it must not become a second production environment implementation or a source of training/evaluation claims.