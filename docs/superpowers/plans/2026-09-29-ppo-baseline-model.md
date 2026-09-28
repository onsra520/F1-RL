# PPO Baseline Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin Stable-Baselines3 PPO adapter with editable YAML configuration, hybrid CPU-simulation/GPU-model device selection, smoke training, and save/load inference support.

**Architecture:** `F1RLEnv` remains the authoritative environment while `f1rl.learning.algorithms.ppo` owns the only production SB3 dependency. `PPOConfig` validates direct construction and YAML-loaded settings; `PPOModel` delegates create/learn/predict/save/load to one SB3 PPO instance without taking ownership of tracks, rewards, evaluation, or training orchestration.

**Tech Stack:** Python 3.12, Gymnasium 1.3.x, NumPy 2.x, Stable-Baselines3 2.9.x, PyYAML 6.x, PyTorch transitively through SB3, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-ppo-baseline-model-design.md`

## Global Constraints

- Work only on branch `models/ppo-baseline` in its isolated worktree.
- Preserve existing `F1RLEnv`, sandbox, runtime, action, observation, reward, and episode production semantics unchanged.
- Base dependencies stay Gymnasium + NumPy; SB3 and PyYAML belong to optional `rl` and `dev` extras.
- Do not add a direct project dependency on PyTorch and do not import `torch` from F1-RL production code.
- Only `f1rl/learning/algorithms/ppo.py` may directly import `stable_baselines3` in production code for this milestone.
- `f1rl.learning` must remain importable without the optional RL extra and must not re-export PPO symbols.
- Policy is fixed to `MlpPolicy`; no callbacks, vectorization, custom policy kwargs, curriculum, evaluator, or trainer in this branch.
- Supported device values are exactly `cpu`, `cuda`, and `auto`; checked-in YAML defaults to `cpu`.
- Explicit unavailable `cuda` must fail clearly; `auto` may resolve according to SB3/PyTorch.
- Simulation/world/sensors/reward/env stepping remain CPU-side even when model compute is CUDA-backed.
- Training is cumulative with `reset_num_timesteps=False`.
- Generated `.zip` models are runtime artifacts and must not be committed.

## Review Focus

1. **Explicit CUDA requested but unavailable:** creation/load must raise an actionable `RuntimeError`, never silently use CPU. Task 2 owns this regression test.
2. **Malformed YAML shape or unknown keys:** `PPOConfig.from_yaml()` must reject non-mapping roots, unknown keys, and bad values while allowing omitted fields to use defaults. Task 1 owns these tests.
3. **Inference-only loaded model:** `load(..., env=None)` may predict but `learn()` must fail clearly instead of leaking an SB3 backend error. Task 3 owns this test.
4. **Artifact path ambiguity:** save/load must normalize a missing `.zip` suffix consistently and reject missing parents/directories/missing archives before backend calls. Task 3 owns these tests.
5. **Optional dependency leakage:** importing lower/base learning layers must not require SB3/PyTorch. Task 4 owns source-boundary and import-isolation tests.

---

## File Structure

```text
f1rl/learning/algorithms/
├── __init__.py              # public optional PPO package surface
├── README.md                # implemented adapter/device/config contract
└── ppo.py                   # PPOConfig + PPOModel; sole SB3 production owner

configs/training/
└── ppo-baseline.yml         # editable checked-in defaults

pyproject.toml               # rl/dev optional dependencies
f1rl/learning/README.md      # roadmap update only

tests/learning/algorithms/
├── test_ppo_config.py       # config + YAML validation
└── test_ppo_model.py        # construction/predict/path/unit contracts

tests/integration/
└── test_ppo_baseline_integration.py  # real SB3/F1RLEnv smoke/save-load/boundaries
```

No production files under `f1rl/sandbox`, `f1rl/runtime`, `f1rl/learning/gym_environment.py`, action, observation, rewards, or episode may change.

### Task 1: PPO configuration and editable YAML

**Files:**
- Modify: `pyproject.toml`
- Create: `configs/training/ppo-baseline.yml`
- Create: `f1rl/learning/algorithms/ppo.py`
- Test: `tests/learning/algorithms/test_ppo_config.py`

**Interfaces:**
- Consumes: Python stdlib `dataclasses`, `pathlib.Path`; optional PyYAML from `rl`/`dev` extras.
- Produces: `PPOConfig(...)`; `PPOConfig.from_yaml(path: str | Path) -> PPOConfig`.

- [ ] **Step 1: Write failing dependency/config tests**

Pin exact defaults: `3e-4, 256, 64, 10, 0.99, 0.95, 0.2, 0.0, 0.5, 0.5, seed=0, device="cpu"`. Assert dataclass immutability and validation for booleans, non-finite values, integer/range rules, `batch_size <= n_steps`, and exact divisibility.

- [ ] **Step 2: Write failing YAML tests**

Assert checked-in `configs/training/ppo-baseline.yml` loads to `PPOConfig()`, omitted fields use defaults, `device: cuda` and `device: auto` load successfully, and non-mapping roots / unknown keys / malformed values / missing files / invalid path types fail with project-owned `ValueError`.

- [ ] **Step 3: Run Task 1 tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/algorithms/test_ppo_config.py -q`
Expected: collection/import failure because `PPOConfig` and/or algorithm module do not yet exist.

- [ ] **Step 4: Add optional RL dependencies**

Update `pyproject.toml` so `rl` and `dev` include `stable-baselines3>=2.9,<2.10` and `PyYAML>=6,<7`, while base dependencies remain exactly Gymnasium + NumPy.

- [ ] **Step 5: Implement `PPOConfig` and `from_yaml()`**

Use `@dataclass(frozen=True, slots=True)`. YAML parsing uses `yaml.safe_load`; require a root mapping with optional `ppo:` mapping exactly as specified, reject unknown config keys, merge omitted fields with dataclass defaults, then pass through the same constructor validation.

- [ ] **Step 6: Add checked-in baseline YAML**

Create `configs/training/ppo-baseline.yml` with all approved baseline values and `device: cpu`.

- [ ] **Step 7: Install RL development dependencies in the worktree venv**

Run: `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`
Expected: SB3 2.9.x, PyYAML 6.x, and transitive PyTorch install successfully under Python 3.12.

- [ ] **Step 8: Verify Task 1 GREEN and regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/algorithms/test_ppo_config.py -q
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: focused config tests PASS and the full pre-model suite remains green with RL dependencies installed.

- [ ] **Step 9: Commit Task 1**

```powershell
git add pyproject.toml configs/training/ppo-baseline.yml f1rl/learning/algorithms/ppo.py tests/learning/algorithms/test_ppo_config.py
git commit -m "feat(learning): define ppo baseline configuration"
```

### Task 2: PPO construction, device semantics, and prediction

**Files:**
- Modify: `f1rl/learning/algorithms/ppo.py`
- Create: `tests/learning/algorithms/test_ppo_model.py`
- Create: `tests/integration/test_ppo_baseline_integration.py`

**Interfaces:**
- Consumes: `PPOConfig`; `F1RLEnv`; SB3 `PPO`; NumPy observation/action arrays.
- Produces: `PPOModel.create(env: F1RLEnv, config: PPOConfig) -> PPOModel`; `predict(observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray`; `num_timesteps: int`.

- [ ] **Step 1: Write failing construction tests**

Assert concrete `F1RLEnv`/`PPOConfig` type requirements, fixed `MlpPolicy`, `verbose=0`, configured hyperparameter forwarding, seed forwarding, and device forwarding for `cpu`/`auto`.

- [ ] **Step 2: Pin explicit CUDA failure behavior**

Using backend/device availability isolation rather than requiring physical GPU hardware, assert `device="cuda"` raises actionable `RuntimeError` when CUDA cannot be honored and does not silently construct on CPU.

- [ ] **Step 3: Write failing prediction validation tests**

Reject non-NumPy input, wrong shape, bool/object/string dtype, non-finite values, out-of-space observations, and non-bool `deterministic`. Valid deterministic prediction must return finite `np.float32`, shape `(2,)`, contained by the backend/F1RLEnv action space.

- [ ] **Step 4: Write real environment compatibility test**

Construct the deterministic smoke `F1RLEnv`, run `stable_baselines3.common.env_checker.check_env(env, warn=False, skip_render_check=True)`, create CPU PPO, reset env, predict, and prove the predicted action can be passed to `env.step()`.

- [ ] **Step 5: Run Task 2 tests and verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/algorithms/test_ppo_model.py tests/integration/test_ppo_baseline_integration.py -q
```
Expected: FAIL because `PPOModel` construction/prediction API is not implemented yet.

- [ ] **Step 6: Implement `PPOModel.create()` and `predict()`**

Wrap exactly one private SB3 `PPO` instance. Forward approved config fields, use fixed `MlpPolicy`, `verbose=0`, and preserve SB3 as the sole model backend. Validate observation before inference, coerce valid observations to `np.float32`, and validate backend action output without remapping/clipping in F1-RL code.

For explicit `cuda`, verify the resolved backend device is CUDA and raise actionable `RuntimeError` if construction cannot honor the request; `auto` may resolve to CPU or CUDA.

- [ ] **Step 7: Expose `num_timesteps` directly from backend**

Implement read-only property `num_timesteps -> int`; do not maintain a second counter.

- [ ] **Step 8: Verify Task 2 GREEN and full regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/algorithms/test_ppo_model.py tests/integration/test_ppo_baseline_integration.py -q
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: focused model/env tests PASS and full suite green.

- [ ] **Step 9: Commit Task 2**

```powershell
git add f1rl/learning/algorithms/ppo.py tests/learning/algorithms/test_ppo_model.py tests/integration/test_ppo_baseline_integration.py
git commit -m "feat(learning): add ppo baseline model adapter"
```

### Task 3: Cumulative learning and save/load round-trip

**Files:**
- Modify: `f1rl/learning/algorithms/ppo.py`
- Modify: `tests/learning/algorithms/test_ppo_model.py`
- Modify: `tests/integration/test_ppo_baseline_integration.py`

**Interfaces:**
- Consumes: Task 2 `PPOModel` wrapper and real smoke `F1RLEnv` fixture.
- Produces: `learn(total_timesteps: int) -> PPOModel`; `save(path: str | Path) -> Path`; `load(path: str | Path, *, env: F1RLEnv | None = None, device: str = "cpu") -> PPOModel`.

- [ ] **Step 1: Write failing learning-contract tests**

Reject booleans/non-integers/non-positive `total_timesteps` before backend learning. Assert `learn()` returns `self`, calls cumulative SB3 semantics (`reset_num_timesteps=False`), and repeated calls produce a monotonic backend `num_timesteps` that advances by at least the requested amount per successful call.

- [ ] **Step 2: Write failing path/save/load tests**

Pin path normalization by appending `.zip` when absent, returning the actual archive path, rejecting blank/non-path values, missing parent directories, directory destinations, missing load archives, invalid env values, and invalid device strings.

- [ ] **Step 3: Write failing inference-only load test**

Save a real CPU model, load it with `env=None`, prove prediction works for a compatible observation, and prove `learn()` raises clear `RuntimeError` before backend training.

- [ ] **Step 4: Write failing real smoke-training round-trip test**

Use `n_steps=8`, `batch_size=4`, `n_epochs=1`, CPU device, an episode limit greater than the rollout, and a finish far enough away to avoid accidental success. Train briefly, save, load into a fresh compatible env, compare deterministic pre/post-save actions with tight tolerance, step the fresh env, then continue cumulative learning.

Do **not** assert reward improvement, finish success, lap time, convergence, or policy quality.

- [ ] **Step 5: Run Task 3 tests and verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/algorithms/test_ppo_model.py tests/integration/test_ppo_baseline_integration.py -q
```
Expected: FAIL on missing `learn`/`save`/`load` behavior.

- [ ] **Step 6: Implement cumulative `learn()`**

Validate the request, require a bound backend environment, delegate with `reset_num_timesteps=False`, and return `self`. Do not add checkpointing/evaluation/callback behavior.

- [ ] **Step 7: Implement normalized `save()` and `load()`**

Centralize `.zip` path validation/normalization. `load()` accepts `cpu|cuda|auto`, optional concrete `F1RLEnv`, and applies the same explicit-CUDA-no-fallback rule as model creation.

- [ ] **Step 8: Verify Task 3 GREEN and full regressions**

Run focused tests then `.\.venv\Scripts\python.exe -m pytest -q`.
Expected: all PPO tests and the full suite pass; smoke training completes without performance assertions.

- [ ] **Step 9: Commit Task 3**

```powershell
git add f1rl/learning/algorithms/ppo.py tests/learning/algorithms/test_ppo_model.py tests/integration/test_ppo_baseline_integration.py
git commit -m "feat(learning): train and persist ppo baseline models"
```

### Task 4: Public optional package surface, docs, boundaries, and final acceptance

**Files:**
- Create: `f1rl/learning/algorithms/__init__.py`
- Modify: `f1rl/learning/algorithms/README.md`
- Modify: `f1rl/learning/README.md`
- Modify: `tests/integration/test_ppo_baseline_integration.py`
- Test/inspect: `f1rl/learning/__init__.py`, lower-layer production modules, `.gitignore`

**Interfaces:**
- Consumes: completed `PPOConfig`/`PPOModel` from Tasks 1-3.
- Produces: optional public import `from f1rl.learning.algorithms import PPOConfig, PPOModel`; verified branch ready for integration.

- [ ] **Step 1: Write failing public-import and dependency-boundary tests**

Assert algorithm package exports exactly the approved PPO symbols; `f1rl.learning` does not import/re-export them; only `f1rl/learning/algorithms/ppo.py` imports `stable_baselines3`; no F1-RL production module imports `torch`; sandbox/runtime/Gym/action/observation/reward/episode remain free of SB3/PyTorch imports.

- [ ] **Step 2: Pin generated-artifact safety**

Using a temporary save archive and repository status inspection, prove generated `.zip` files are not staged/tracked and existing `artifacts/checkpoints/*` / `artifacts/policies/*` ignore rules remain authoritative.

- [ ] **Step 3: Run Task 4 boundary tests and verify RED**

Expected: public algorithm package export/docs assertions fail before `__init__.py` and README updates are implemented.

- [ ] **Step 4: Implement optional package exports**

Create `f1rl/learning/algorithms/__init__.py` exporting `PPOConfig` and `PPOModel`. Do not modify `f1rl/learning/__init__.py` to import the optional algorithm package.

- [ ] **Step 5: Update documentation**

Document optional `.[rl]` installation, editable `configs/training/ppo-baseline.yml`, fixed `MlpPolicy`, `cpu/cuda/auto`, hybrid CPU-simulation/GPU-model semantics, cumulative learning, save/load behavior, and the boundary that vectorized training/evaluation/generalization remain future milestones.

- [ ] **Step 6: Run focused final acceptance under Python 3.12**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/algorithms tests/integration/test_ppo_baseline_integration.py -q
```
Expected: all focused PPO/config/integration/boundary tests PASS.

- [ ] **Step 7: Run full verification**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q f1rl
```
Then run source scans for forbidden `stable_baselines3`/`torch` dependencies and `git diff --check`.
Expected: full suite green, compileall PASS, source boundaries PASS, diff check PASS.

- [ ] **Step 8: Inspect changed production scope**

Compare branch against base `4906df7`. Production changes must be limited to `f1rl/learning/algorithms/**`, `configs/training/ppo-baseline.yml`, `pyproject.toml`, and learning documentation. Any sandbox/runtime/Gym/action/observation/reward/episode production diff is a stop condition.

- [ ] **Step 9: Commit Task 4**

```powershell
git add f1rl/learning/algorithms/__init__.py f1rl/learning/algorithms/README.md f1rl/learning/README.md tests/integration/test_ppo_baseline_integration.py
git commit -m "test(learning): verify ppo baseline integration"
```

- [ ] **Step 10: Whole-branch review before integration**

Review `4906df7..HEAD` against the approved spec with special attention to the five Review Focus items. Critical/Important findings require one explicit RED→GREEN fix pass plus a fresh full suite; Minor findings are recorded for the human partner rather than silently expanded into scope.

## Completion Evidence

Before claiming the branch complete, report:

```text
branch / HEAD
commits
SB3 / PyYAML / PyTorch resolved versions
focused PPO tests
real smoke-training result and num_timesteps
save/load deterministic round-trip
full pytest result
Python 3.12 compileall
dependency boundaries
git diff --check
changed production scope
review findings / rulings / deferred minors
push status
```
