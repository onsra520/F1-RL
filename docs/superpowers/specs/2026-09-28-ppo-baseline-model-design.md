# PPO Baseline Model Design

## Goal

Add the first trainable RL model integration for F1-RL by wrapping Stable-Baselines3 PPO behind a small project-owned adapter.

The milestone proves that the verified `F1RLEnv` can be consumed by a real PPO implementation, trained for a short rollout, saved, loaded, and used for deterministic inference without moving algorithm concerns into sandbox, runtime, or the Gym adapter.

This is a baseline integration milestone, not a performance milestone. Success means the model pipeline is correct and reproducible enough to build training/evaluation infrastructure on top of it; it does not mean the agent already drives well.

## Architectural position

```text
sandbox / runtime / learning primitives
                |
                v
            F1RLEnv
                |
                v
      PPOModel + PPOConfig
                |
                v
      Stable-Baselines3 PPO
                |
                v
             PyTorch
```
## Scope

### In scope

- Stable-Baselines3 PPO as the first concrete learning algorithm backend.
- `PPOConfig` as an immutable validated baseline hyperparameter contract.
- `PPOModel` as a thin wrapper for create, learn, predict, save, load, and timestep inspection.
- Fixed `MlpPolicy` for the V1 baseline.
- CPU execution as the V1 supported device.
- Short smoke training against a real `F1RLEnv`.
- Stable-Baselines3 environment checker acceptance.
- Save/load round-trip and deterministic inference verification.
- Optional `rl` dependency installation rather than forcing the ML stack on base users.
- Dependency-boundary tests that keep SB3/PyTorch out of lower layers.

### Out of scope

- Vectorized environments or `VecEnv` as public F1-RL inputs.
- Multi-track sampling or reset-time track randomization.
- Curriculum/domain randomization.
- Training orchestration, callbacks, TensorBoard/W&B integration, or checkpoint schedules.
- Evaluation metrics, success-rate targets, lap-time targets, or generalization experiments.
- Hyperparameter search or tuning.
- GPU/CUDA support in the F1-RL PPO baseline contract.
- SAC, TD3, DQN, or a generic algorithm base class.
- Custom policy networks or feature extractors.
- Reward, action, observation, physics, or episode-semantics changes.
## Package and dependency placement

Create:

```text
f1rl/learning/algorithms/
├── __init__.py
└── ppo.py

configs/training/ppo-baseline.yml
```

`ppo.py` is the only production module in this milestone allowed to import `stable_baselines3`. The checked-in YAML file is the human-editable baseline configuration for model hyperparameters and device selection.

Update `pyproject.toml` optional dependencies to include:

```toml
[project.optional-dependencies]
rl = [
    "stable-baselines3>=2.9,<2.10",
    "PyYAML>=6,<7",
]

dev = [
    "pytest>=8,<10",
    "stable-baselines3>=2.9,<2.10",
    "PyYAML>=6,<7",
]
```

Do not add a direct project dependency on `torch`; Stable-Baselines3 owns its PyTorch dependency. F1-RL production code in this milestone must not import `torch` directly. PyYAML is part of the optional `rl` extra because editable PPO configuration is also an RL-only concern.

The base project dependencies remain Gymnasium and NumPy only, so `pip install -e .` continues to support simulation/Gym usage without installing the full RL training stack.
## Public import boundary

The PPO API is public from the algorithm package:

```python
from f1rl.learning.algorithms import PPOConfig, PPOModel
```

and equivalently from:

```python
from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
```

Do **not** re-export `PPOConfig` or `PPOModel` from `f1rl.learning` in V1. Importing `f1rl.learning` must continue to work when the optional `rl` extra is not installed.

Importing `f1rl.learning.algorithms` without Stable-Baselines3 installed may fail with an actionable `ImportError` explaining that the `rl` extra is required. No lower package should import `f1rl.learning.algorithms` implicitly.

## PPO configuration contract

```python
@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int | None = 0
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PPOConfig:
        ...
```
Validation rules:

- All floating-point hyperparameters must be real finite numbers; booleans are rejected as numeric values.
- `learning_rate > 0`.
- `n_steps` is an integer `>= 2`; booleans are rejected.
- `batch_size` is an integer `>= 2`, must be `<= n_steps`, and must divide `n_steps` exactly for the single-environment baseline rollout buffer.
- `n_epochs` is a positive integer; booleans are rejected.
- `gamma` and `gae_lambda` are in `[0, 1]`.
- `clip_range` is in `(0, 1]`.
- `ent_coef` and `vf_coef` are non-negative.
- `max_grad_norm > 0`.
- `seed` is `None` or an integer; booleans are rejected.
- `device` must be one of `"cpu"`, `"cuda"`, or `"auto"`.

`"cpu"` remains the checked-in baseline default because PPO with a small `MlpPolicy` is commonly environment/rollout bound. `"cuda"` explicitly requests GPU-backed model compute, while `"auto"` delegates device selection to Stable-Baselines3/PyTorch. In every case F1-RL simulation, track queries, sensors, rewards, and Gym environment stepping remain CPU-side; using CUDA therefore creates a deliberate hybrid CPU-simulation + accelerator-model pipeline.

If `device="cuda"` cannot be honored by the installed PyTorch runtime, model creation/load must fail with an actionable `RuntimeError` rather than silently falling back to CPU. `device="auto"` may resolve to CPU or CUDA according to the backend.

The configuration remains intentionally narrower than Stable-Baselines3. Vectorized training and tuning infrastructure are still future milestones.

## Editable YAML configuration

The repository includes:

```text
configs/training/ppo-baseline.yml
```

with a single top-level `ppo` mapping whose keys correspond exactly to `PPOConfig` fields. The checked-in values match `PPOConfig()` defaults, including `device: cpu`.

`PPOConfig.from_yaml(path: str | Path) -> PPOConfig` loads this file with `yaml.safe_load` and applies the same dataclass validation as direct construction.

YAML rules:

- the document root must be a mapping containing exactly one `ppo` mapping;
- unknown top-level keys or unknown PPO keys are rejected rather than silently ignored;
- omitted PPO keys use the dataclass defaults;
- explicit YAML `null` is accepted only for `seed`;
- empty files, non-mapping documents, and malformed values fail with `ValueError`;
- loading configuration never mutates the file or environment variables.

The YAML file configures the algorithm only. It does not construct tracks, rewards, environments, artifact paths, or training/evaluation orchestration.

Typical device edits are:

```yaml
ppo:
  device: cpu   # deterministic baseline / often best for small MLP PPO
```

```yaml
ppo:
  device: cuda  # CPU simulation + GPU policy/value network
```

```yaml
ppo:
  device: auto  # let SB3/PyTorch choose available model device
```

## Model API

```python
class PPOModel:
    @classmethod
    def create(cls, env: F1RLEnv, config: PPOConfig) -> PPOModel:
        ...

    def learn(self, total_timesteps: int) -> PPOModel:
        ...

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        ...
```
```python
    def save(self, path: str | Path) -> Path:
        ...

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        env: F1RLEnv | None = None,
        device: str = "cpu",
    ) -> PPOModel:
        ...

    @property
    def num_timesteps(self) -> int:
        ...
```

The wrapper owns no track construction, reward design, episode logic, evaluation policy, or artifact-selection policy. It delegates algorithm behavior to one private Stable-Baselines3 `PPO` instance.

## Model creation

`PPOModel.create()` requires concrete `F1RLEnv` and `PPOConfig` instances.

It constructs Stable-Baselines3 PPO with fixed policy name:

```text
MlpPolicy
```

and forwards every `PPOConfig` field to the matching SB3 constructor argument. Set `verbose=0` in the baseline adapter. Do not add custom policy kwargs, custom rollout buffers, callbacks, or wrappers.
`PPOModel.create()` does not run an environment checker automatically. Compliance belongs in tests and training orchestration, not every model construction call.

## Training semantics

`learn(total_timesteps)` validates that `total_timesteps` is a positive integer and rejects booleans.

Training requires the wrapped SB3 model to have an environment. A model loaded with `env=None` is inference-only until a future explicit rebinding capability exists; calling `learn()` on it raises `RuntimeError` rather than leaking an opaque backend failure.

The wrapper calls SB3 learning with cumulative timestep semantics:

```text
reset_num_timesteps = False
```

Therefore repeated calls continue one training run and `num_timesteps` is monotonic.

Because on-policy PPO collects complete rollout buffers, `num_timesteps` may exceed the requested `total_timesteps`. The contract is that a successful call advances the counter by at least the requested amount; it does not promise exact equality.

`learn()` returns `self` for fluent use and does not save checkpoints, choose artifacts, evaluate performance, or modify F1-RL environment semantics.

## Timestep inspection

`num_timesteps` returns the backend model's current integer timestep count.

It is non-negative and reflects SB3's cumulative counter. The wrapper does not maintain a second independent counter.
## Prediction contract

`predict()` accepts one non-vectorized observation produced for one F1-RL environment state.

Validation before backend inference:

- `observation` must be a NumPy array.
- Shape must exactly match the backend model observation-space shape.
- Dtype must be a real numeric dtype; boolean/object/string arrays are rejected.
- Values must be finite.
- Convert to `np.float32` and require containment in the backend observation space.
- `deterministic` must be a real `bool`.

The wrapper delegates to SB3 `predict()` and returns one action array with:

```text
shape = (2,)
dtype = np.float32
```

The result must be finite and contained by the backend action space. The wrapper does not remap, normalize, or clip actions itself; if the backend violates its declared action-space contract, raise `RuntimeError`.

The existing F1-RL asymmetric V1 action contract remains authoritative:

```text
steering [-1, 1]
throttle [0, 1]
```

This milestone does not add a symmetric-action wrapper merely to satisfy an SB3 recommendation. Action normalization, if later justified experimentally, is a separate milestone.
## Save contract

`save(path)` accepts `str | Path` and writes one Stable-Baselines3 model archive.

Path rules:

- Reject empty/blank string paths and non-string/non-`Path` values.
- Normalize the destination to a `.zip` file. If the supplied path does not already end in `.zip`, append `.zip` rather than replacing an existing suffix.
- The parent directory must already exist and be a directory; the model adapter does not create experiment/artifact directory structures.
- Reject a destination that resolves to an existing directory.
- Existing model files may be overwritten according to SB3 save semantics.
- Return the actual normalized `.zip` `Path` after successful save.

The adapter does not restrict callers to `artifacts/`; tests may use temporary directories. Repository ignore rules already prevent generated files under `artifacts/checkpoints/` and `artifacts/policies/` from being committed accidentally.

## Load contract

`load(path, *, env=None, device="cpu")` accepts the same normalized `.zip` path semantics and requires the archive to exist as a file.

- `env` is `None` or a concrete `F1RLEnv`.
- `device` must be one of `"cpu"`, `"cuda"`, or `"auto"`; explicit unavailable CUDA fails clearly instead of silently falling back.
- Delegate loading to Stable-Baselines3 `PPO.load()`.
- When `env` is supplied, the loaded model may continue cumulative learning.
- When `env=None`, prediction is supported but `learn()` must fail clearly as inference-only.

Load does not reconstruct or expose a second `PPOConfig`; serialized SB3 parameters remain the source of truth for a loaded backend model.
## Save/load inference round-trip

For the same valid observation on CPU, a freshly saved and reloaded model must preserve deterministic inference within floating-point equality appropriate for serialized weights:

```text
action_before ≈ action_after
```

The acceptance test uses a fresh compatible `F1RLEnv` when loading for continued environment interaction.

The loaded deterministic action must remain finite, shape `(2,)`, and valid for `F1RLEnv.action_space`; stepping the fresh environment with that action must succeed.

No test requires stochastic (`deterministic=False`) predictions to match across save/load.

## Stable-Baselines3 environment compatibility

The existing `F1RLEnv` remains unchanged in this milestone.

Acceptance runs:

```python
stable_baselines3.common.env_checker.check_env(
    env,
    warn=False,
    skip_render_check=True,
)
```

The current asymmetric continuous action Box is accepted as a valid environment contract. This branch does not mutate action bounds to remove an advisory symmetry warning.

The model adapter must not register a Gym environment ID, wrap rendering, or alter reset/step semantics.
## Smoke-training acceptance environment

Tests use a deliberately simple deterministic `F1RLEnv` fixture rather than a production training scenario.

The fixture should have:

- one fixed track;
- a valid finish placed far enough away that an 8-step smoke rollout does not accidentally terminate by success;
- a deterministic initial vehicle state;
- a small valid observation sensor suite;
- the existing strict 2D action contract;
- simple existing reward terms only;
- an episode step budget larger than the smoke rollout.

Smoke training uses a deliberately small config such as:

```text
n_steps = 8
batch_size = 4
n_epochs = 1
device = cpu  # checked-in baseline; editable to cuda or auto
```

and a short requested training duration. The test proves the algorithm pipeline executes; it does not assert reward improvement, finish success, lap time, policy quality, or convergence.

The production `PPOConfig` defaults remain the baseline values declared earlier and are not reduced merely to make tests faster.
## Reproducibility semantics

The baseline forwards the configured seed to Stable-Baselines3. Verification uses CPU for portability; deterministic prediction on a saved model must remain stable on the same configured runtime/device.

The environment fixture and F1-RL dynamics are already deterministic for equal inputs. However, this milestone does not promise bit-for-bit equality of independent training runs across PyTorch builds, operating systems, processor implementations, or future library versions.

Required reproducibility guarantees are narrower:

- identical `PPOConfig` values construct the same algorithm configuration;
- `seed` is forwarded unchanged;
- the checked-in/test baseline uses CPU, while `cpu`, `cuda`, and `auto` are supported configuration values;
- CUDA mode leaves simulation/environment execution on CPU and moves only backend model compute to the accelerator;
- deterministic inference for a saved model is preserved by save/load on the same supported runtime/device;
- equal serialized weights and equal observations produce equal/approximately equal deterministic actions.

Do not add global PyTorch deterministic-algorithm flags in V1 because F1-RL does not directly own PyTorch execution policy yet.

## Dependency boundaries

Allowed direct framework dependency:

```text
f1rl.learning.algorithms.ppo -> stable_baselines3
```

Forbidden direct dependencies on `stable_baselines3` or `torch` remain in:

- `f1rl.sandbox/**`;
- `f1rl.runtime/**`;
- `f1rl.learning.gym_environment`;
- observation/action/reward/episode modules.
`f1rl.learning` must not import `f1rl.learning.algorithms` so the base learning package remains usable without the optional RL extra.

`f1rl.learning.algorithms.ppo` must not import sandbox physics/domain modules except through the public `F1RLEnv` type needed by the adapter boundary. It must not construct tracks, rewards, observations, or runtime simulations.

`f1rl.learning.training` and `f1rl.learning.evaluation` remain unchanged except documentation if necessary; they do not become SB3 owners in this milestone.

## Failure semantics

Project-owned validation errors use `ValueError` for malformed config, observations, paths, device values, or argument types consistent with current project style.

Calling `learn()` without a bound environment raises `RuntimeError`.

Backend failures during PPO construction, training, serialization, or loading propagate rather than being converted into fake success. The wrapper does not promise transactional rollback of partially completed backend learning.

If Stable-Baselines3 is missing when the algorithm package is imported, raise an actionable `ImportError` that points users to the `rl` extra, for example:

```text
PPO support requires the optional RL dependencies; install f1-rl with the `rl` extra.
```

Do not catch broad exceptions around training/prediction merely to hide backend diagnostics.

## Artifact behavior

Generated `.zip` model files are runtime artifacts, not source files.

The existing repository ignore rules under `artifacts/checkpoints/` and `artifacts/policies/` remain authoritative. This milestone does not commit a trained model binary.
## Testing and acceptance

Create focused tests, expected under:

```text
tests/learning/algorithms/test_ppo_config.py
tests/learning/algorithms/test_ppo_model.py
tests/integration/test_ppo_baseline_integration.py
```

Acceptance must prove at minimum:

1. `PPOConfig` defaults exactly match the approved baseline values.
2. Each numeric/type/range validation rule rejects malformed values, including booleans where integers/floats are required.
3. `batch_size` must divide `n_steps` for the single-env baseline.
4. `PPOConfig` accepts only `cpu`, `cuda`, or `auto` device values; malformed device strings are rejected.
5. `configs/training/ppo-baseline.yml` matches `PPOConfig()` defaults and `PPOConfig.from_yaml()` rejects malformed/unknown YAML keys while allowing omitted fields to use defaults.
6. `PPOModel.create()` rejects non-`F1RLEnv` / non-`PPOConfig` inputs.
7. Model construction uses SB3 PPO with `MlpPolicy`, the configured seed/device, and forwarded hyperparameters; CPU is used for automated baseline verification.
8. Explicit `device="cuda"` on a runtime where CUDA cannot be honored fails with an actionable `RuntimeError`; tests do not require physical GPU hardware.
9. SB3 `check_env(..., warn=False, skip_render_check=True)` accepts the real smoke `F1RLEnv`.
10. A short real `learn()` call completes and increases `num_timesteps` by at least the requested amount.
11. Repeated `learn()` calls keep a monotonic cumulative timestep count.
12. Invalid `total_timesteps` values fail before backend training.
13. `predict()` rejects wrong type/shape/dtype, non-finite values, and observations outside the declared space.
14. Deterministic prediction returns finite `np.float32` shape `(2,)` inside the model/F1RLEnv action space.
15. A predicted action can be passed to a reset `F1RLEnv.step()` successfully.
16. `save()` normalizes paths to `.zip` and returns the real archive path.
17. `save()` rejects missing parent directories and directory destinations.
18. `load()` rejects missing archives, wrong env types, and unsupported devices.
19. Save/load preserves deterministic prediction for the same observation within tight floating-point tolerance.
20. Loading with a fresh env permits environment interaction and continued cumulative learning.
21. Loading with `env=None` permits prediction but makes `learn()` raise `RuntimeError`.
22. Generated model archives stay in temporary/ignored artifact locations and are not added to Git.
23. `f1rl.learning.algorithms` exports `PPOConfig` and `PPOModel`.
24. `f1rl.learning` does not import or re-export the optional PPO API.
25. Lower layers and `gym_environment.py` do not import `stable_baselines3` or `torch`.
26. Only `f1rl/learning/algorithms/ppo.py` directly imports Stable-Baselines3 in production code for this milestone.
27. No F1-RL production module directly imports PyTorch.
28. Existing `F1RLEnv` action/observation/reward/episode behavior remains unchanged.
29. Existing full test suite remains green after installing the RL development dependencies.
30. Python 3.12 compileall succeeds.
31. `git diff --check` succeeds.

Tests must not assert that reward improves, a finish is reached, or PPO converges during the smoke rollout.

## Documentation

Update `f1rl/learning/algorithms/README.md` to describe the implemented PPO adapter, optional `rl` installation, fixed `MlpPolicy`, YAML configuration, supported `cpu`/`cuda`/`auto` device modes, hybrid CPU-simulation/GPU-model behavior, and ownership boundary.

Update `f1rl/learning/README.md` to mark PPO baseline integration as implemented while keeping vectorization, training orchestration, evaluation, and generalization as later layers.

Do not present `training/` or `evaluation/` as implemented merely because a model can now learn.

## Verification

Before this feature is considered complete, run fresh under Python 3.12:

```text
focused PPO config/YAML/model tests
PPO integration smoke test on CPU
full pytest suite
compileall f1rl
SB3 environment checker acceptance
dependency/source boundary scans
git diff --check
```

Inspect the changed-file scope and confirm there are no production changes to sandbox, runtime, `gym_environment.py`, action, observation, reward, or episode modules.
## Post-milestone sequence

After the PPO baseline is verified, the next architectural milestones are intentionally separate:

```text
PPO baseline model
    -> vectorized environment/training support
    -> multi-track sampling
    -> repeatable training orchestration
    -> independent evaluation metrics
    -> unseen-track generalization experiments
    -> reward-shaping ablation
```

Do not pre-build those layers in this branch.

## Explicit non-goals for scientific claims

A successful baseline smoke run does not establish that PPO is the best algorithm, that the reward is well-shaped, that the agent generalizes, or that the selected hyperparameters are performant.

Those are experimental questions. This branch establishes only the software contract required to ask them reliably later.
