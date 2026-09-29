# Vectorized PPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add same-track vectorized PPO training with explicit DummyVecEnv/SubprocVecEnv backends, editable vector YAML configuration, correct PPO rollout sizing, Windows-safe subprocess verification, and a diagnostic throughput benchmark.

**Architecture:** `F1RLEnvFactory` owns repeatable construction of fresh single-track environments; `make_vector_env()` composes those factories into Stable-Baselines3 VecEnv implementations without changing `F1RLEnv`. `PPOModel` widens only its training-environment boundary and validates batch size against `n_steps * num_envs`; multi-track sampling, trainer orchestration, and evaluation remain later layers.

**Tech Stack:** Python 3.12, Gymnasium 1.3.x, NumPy 2.x, Stable-Baselines3 2.9.x, PyYAML 6.x, Windows multiprocessing `spawn`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-vectorized-ppo-training-design.md`

## Global Constraints

- Work only on stacked branch `training/vectorized-ppo`, based on `models/ppo-baseline@3c29ad340d445ab4458c3e3c0b9b29fd647aa4a1`.
- Preserve sandbox/runtime physics and `F1RLEnv` production semantics unchanged.
- Keep `configs/training/ppo-baseline.yml` and standalone `PPOConfig.from_yaml()` backward-compatible and strict.
- Vector V1 uses the same authored track/config inputs in every slot; no multi-track sampling or curriculum.
- Supported vector backends are exactly `dummy` and `subproc`; subprocess V1 uses `spawn` explicitly.
- Default vector config is exactly `n_envs=8`, `backend="dummy"`, `start_method="spawn"`.
- No direct PyTorch import is added to F1-RL production code.- Allowed direct SB3 production imports after this milestone are only `f1rl.learning.algorithms.ppo` and `f1rl.learning.training.vector_env`.
- `f1rl.learning` must not import/re-export optional training symbols.
- `PPOModel.predict()` remains single-observation `(obs_size,) -> (2,)`; no public batched inference API.
- Vector env lifecycle is caller-owned; successful `VecEnv` instances must be explicitly closed by callers/tests/scripts.
- No automatic backend choice, CPU-count clamp, CUDA-wheel policy, callbacks, trainer class, evaluation, or performance winner assertion.
- Performance benchmark output is diagnostic only and never a CI pass/fail threshold.

## Review Focus

1. **Shared mutable environment state:** repeated factory calls and DummyVecEnv slots must own distinct `F1RLEnv`/Simulation lifecycle state; stepping/resetting one slot cannot mutate siblings. Task 1/2 pin this.
2. **Rollout-size validation migration:** removing `batch_size <= n_steps` from `PPOConfig` must not weaken single-env safety; create/load must validate `batch_size` against `n_steps * num_envs` before training. Task 3 pins both single and vector cases.
3. **Windows subprocess recursion/leaks:** SubprocVecEnv smoke must run from a guarded helper using the same interpreter, `spawn`, and `finally: close()` so pytest collection never recursively spawns workers. Task 4 owns this.
4. **Configuration compatibility:** combined vector YAML must be strict while standalone `ppo-baseline.yml` behavior remains unchanged. Task 1 owns regression tests.
5. **Optional dependency leakage/public imports:** base learning/sandbox/runtime/Gym layers must remain SB3/Torch-free and `f1rl.learning` importable without optional training imports. Task 5 owns source-boundary/import tests.

---

## File Structure

```text
f1rl/learning/training/
├── __init__.py              # optional public vector-training surface
├── config.py                # VectorEnvConfig + VectorTrainingConfig YAML loader
├── env_factory.py           # F1RLEnvFactory; no SB3 dependency
├── vector_env.py            # ranked callables + Dummy/Subproc construction
└── README.md                # implemented vector-training contract

configs/training/
└── vectorized-ppo.yml       # combined PPO + vector defaults

scripts/
└── benchmark_vector_envs.py # guarded diagnostic throughput runner
```
Additional modified files:

```text
f1rl/learning/algorithms/ppo.py
f1rl/learning/algorithms/README.md
f1rl/learning/README.md
tests/learning/training/*
tests/learning/algorithms/test_ppo_model.py
tests/integration/test_vectorized_ppo_integration.py
tests/integration/helpers/subproc_vector_smoke.py
```

No production file under `f1rl/sandbox`, `f1rl/runtime`, `f1rl/learning/gym_environment.py`, action, observation, rewards, or episode may change.

### Task 1: Combined vector configuration and fresh environment factory

**Files:**
- Create: `f1rl/learning/training/config.py`
- Create: `f1rl/learning/training/env_factory.py`
- Create: `configs/training/vectorized-ppo.yml`
- Test: `tests/learning/training/test_vector_config.py`
- Test: `tests/learning/training/test_env_factory.py`

**Interfaces:**
- Consumes: `PPOConfig`; existing `F1RLEnv` constructor/domain types.
- Produces: `VectorEnvConfig`; `VectorTrainingConfig.from_yaml(path: str | Path)`; `F1RLEnvFactory.create() -> F1RLEnv`.

- [ ] **Step 0: Prepare isolated Python 3.12 worktree environment**

If `.venv` is absent, create it with the project Python 3.12.10 interpreter, then run `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`. Verify Python 3.12.x, SB3 2.9.x, PyYAML 6.x, and transitive PyTorch resolve before RED tests. Do not change dependency versions in this milestone.

- [ ] **Step 1: Write failing `VectorEnvConfig` tests**

Assert exact defaults `8/dummy/spawn`; frozen dataclass behavior; reject bool/non-int/zero/negative `n_envs`; reject backend values outside `dummy|subproc`; reject blank/non-string start methods; require `spawn` for `subproc`; allow stored non-empty start method for `dummy` without runtime meaning.

- [ ] **Step 2: Write failing combined YAML tests**

Assert `configs/training/vectorized-ppo.yml` loads to `VectorTrainingConfig(PPOConfig(), VectorEnvConfig())`; omitted fields inside each mapping use defaults; root must contain exactly `ppo` and `vector_env`; unknown keys, wrong shapes, malformed/empty YAML, missing files, blank/invalid path values raise `ValueError`.- [ ] **Step 3: Pin standalone PPO YAML backward compatibility**

Assert `PPOConfig.from_yaml("configs/training/ppo-baseline.yml") == PPOConfig()` and that the standalone loader still rejects extra `vector_env`/unknown root mappings exactly as before.

- [ ] **Step 4: Write failing factory validation/freshness tests**

Construct a deterministic smoke domain fixture. Assert `F1RLEnvFactory` rejects wrong constructor-field types at factory creation, every `create()` returns a different `F1RLEnv`, equal initial observations/info are produced after reset, and stepping/resetting one created env leaves the other env's tick/state unchanged.

- [ ] **Step 5: Run Task 1 tests and verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_vector_config.py tests/learning/training/test_env_factory.py -q
```
If the branch has no local `.venv`, use the verified PPO worktree Python 3.12 interpreter with this worktree on `PYTHONPATH` until Task execution creates/installs its own environment.
Expected: import/collection failures because training config/factory production modules do not exist.

- [ ] **Step 6: Implement `VectorEnvConfig` and `VectorTrainingConfig.from_yaml()`**

Use frozen slotted dataclasses. `VectorTrainingConfig` owns combined YAML parsing, validates exact root/section mappings, builds `PPOConfig(**ppo_values)` and `VectorEnvConfig(**vector_values)`, and converts loader/constructor failures into project-owned `ValueError` without modifying `PPOConfig.from_yaml()`.

- [ ] **Step 7: Add checked-in `vectorized-ppo.yml`**

Copy all approved PPO defaults plus vector defaults from the spec. Do not modify `ppo-baseline.yml`.

- [ ] **Step 8: Implement `F1RLEnvFactory`**

Use the exact seven fields/signature from the spec, mirror concrete constructor type validation, and make `create()` instantiate a fresh `F1RLEnv` on every call. Do not import SB3 or add seeding/track-selection behavior.

- [ ] **Step 9: Verify Task 1 GREEN and full regression**

Run the two focused files, then the full project pytest command under Python 3.12. Expected: focused tests PASS and the pre-vector suite remains green.

- [ ] **Step 10: Commit Task 1**

```powershell
git add configs/training/vectorized-ppo.yml f1rl/learning/training/config.py f1rl/learning/training/env_factory.py tests/learning/training/test_vector_config.py tests/learning/training/test_env_factory.py
git commit -m "feat(learning): define vector training configuration"
```
### Task 2: DummyVecEnv construction and vector lifecycle semantics

**Files:**
- Create: `f1rl/learning/training/vector_env.py`
- Create: `tests/learning/training/test_vector_env.py`
- Create: `tests/integration/test_vectorized_ppo_integration.py`

**Interfaces:**
- Consumes: Task 1 `F1RLEnvFactory`; `VectorEnvConfig`; SB3 `VecEnv`, `DummyVecEnv`, `SubprocVecEnv` types.
- Produces: `make_vector_env(factory: F1RLEnvFactory, config: VectorEnvConfig) -> VecEnv`; internal `_RankedEnvFactory(factory, rank)`.

- [ ] **Step 1: Write failing constructor/input tests**

Assert `make_vector_env()` requires concrete `F1RLEnvFactory` and `VectorEnvConfig`; for `dummy`, exactly `n_envs` ranked callable objects are built with ranks `0..n_envs-1`, each call returns a fresh environment, and returned type is `DummyVecEnv`.

- [ ] **Step 2: Write failing batched shape/determinism tests**

Using a real deterministic factory with `n_envs=4`, assert reset observations shape `(4, obs_size)`/`float32`; step accepts action shape `(4, 2)`; rewards/dones shapes are `(4,)`; infos length is 4; equal initial slots plus equal batched actions yield equal observations/rewards/dones.

- [ ] **Step 3: Pin independent slot progression and auto-reset**

Use one shared same-track factory and deliberately divergent batched actions/trajectory so one slot reaches an episode boundary before a sibling. Do not inject different tracks or authored initial states per slot. Assert sibling state remains valid; completed slot auto-resets through SB3; its info includes `terminal_observation`; truncation metadata from SB3 is preserved when applicable.

- [ ] **Step 4: Pin close behavior**

Create a real DummyVecEnv, reset/step it, then assert one explicit caller-owned `close()` completes without error. Do not add idempotence promises, hidden ownership, or shutdown hooks beyond SB3's own contract.

- [ ] **Step 5: Run Task 2 tests and verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_vector_env.py tests/integration/test_vectorized_ppo_integration.py -q
```
Expected: FAIL because `make_vector_env` / ranked wrapper are not implemented.

- [ ] **Step 6: Implement `_RankedEnvFactory` and `make_vector_env()`**

Use an immutable top-level callable dataclass (no required anonymous closure). For `dummy`, pass the ranked callables to `DummyVecEnv`; for `subproc`, construct `SubprocVecEnv(..., start_method=config.start_method)`. Do not create a custom VecEnv subclass and do not alter `F1RLEnv`.

- [ ] **Step 7: Verify Task 2 GREEN and regressions**

Run Task 2 focused tests, existing Gym environment tests, then full pytest. Expected: vector semantics PASS; all pre-existing single-env semantics remain green.

- [ ] **Step 8: Commit Task 2**

```powershell
git add f1rl/learning/training/vector_env.py tests/learning/training/test_vector_env.py tests/integration/test_vectorized_ppo_integration.py
git commit -m "feat(learning): add vector environment construction"
```
### Task 3: PPO VecEnv binding and rollout-size validation

**Files:**
- Modify: `f1rl/learning/algorithms/ppo.py`
- Modify: `tests/learning/algorithms/test_ppo_config.py`
- Modify: `tests/learning/algorithms/test_ppo_model.py`
- Modify: `tests/integration/test_vectorized_ppo_integration.py`

**Interfaces:**
- Consumes: Task 2 `VecEnv`; existing `PPOConfig`/`PPOModel`.
- Produces: training boundary `PPOTrainingEnv = F1RLEnv | VecEnv`; environment-aware rollout validation in `create()` and `load()`.

- [ ] **Step 1: Write failing `PPOConfig` contract-migration tests**

Assert standalone construction accepts vector-valid configurations where `batch_size > n_steps` and no longer enforces `batch_size <= n_steps` or exact divisibility. Preserve every other numeric/type/range/seed/device validation test unchanged.

- [ ] **Step 2: Write failing single-env binding tests**

For `F1RLEnv`, assert `num_envs=1` effective rollout size. Reject `batch_size > n_steps` and `n_steps % batch_size != 0` in `PPOModel.create()` before SB3 construction; valid historical baseline combinations continue to construct.

- [ ] **Step 3: Write failing VecEnv binding tests**

With real/fake VecEnv boundaries as appropriate, assert `rollout_size = n_steps * env.num_envs`; accept `n_steps=8, n_envs=4, batch_size=16/32` when divisible; reject batch size above rollout or non-divisor values before backend model construction.

- [ ] **Step 4: Pin load-time environment compatibility**

Save a real PPO archive, then assert `PPOModel.load(..., env=compatible_vec_env)` succeeds and incompatible vector counts raise project-owned `ValueError` after archive load but before any continued learning. `load(..., env=None)` remains inference-only and performs no env-count check.

- [ ] **Step 5: Pin public prediction contract regression**

Existing single-observation prediction tests must still require shape `(observation_size,)` and return `(2,)`; a batched observation remains rejected by `PPOModel.predict()` despite vectorized training support.

- [ ] **Step 6: Write real vectorized PPO smoke training test**

Use `DummyVecEnv`, small `n_steps`/`batch_size`/`n_epochs`, CPU model device, and at least 2 slots. Assert `PPOModel.create()` accepts it; `learn(total_timesteps)` returns self; `num_timesteps` advances by at least requested amount; save/load into a fresh compatible vector env permits continued cumulative learning. Close every VecEnv in `finally`.

- [ ] **Step 7: Run Task 3 tests and verify RED**

Run focused PPO config/model + vector integration tests. Expected: failures from old standalone batch validation and old `F1RLEnv`-only model binding.

- [ ] **Step 8: Move rollout validation from `PPOConfig` to model binding**

Remove only the two environment-dependent constraints from `PPOConfig.__post_init__`. Add one internal helper in `ppo.py` that derives `num_envs=1` for `F1RLEnv` or `env.num_envs` for concrete SB3 `VecEnv`, computes rollout size, and raises `ValueError` for oversized/non-divisible batch sizes.

- [ ] **Step 9: Widen `PPOModel.create()` / `load()` environment types**

Accept only concrete `F1RLEnv | VecEnv`; run rollout validation before `PPO(...)` construction in create. On load with env, load backend, inspect archived `n_steps`/`batch_size`, validate against supplied env count, preserve CUDA no-fallback semantics, and keep env=None inference-only behavior.- [ ] **Step 10: Verify Task 3 GREEN and full regressions**

Run PPO config/model tests, vector integration tests, existing single-env PPO integration tests, then full pytest. Expected: both single/vector binding semantics PASS; existing save/load/predict tests remain green.

- [ ] **Step 11: Commit Task 3**

```powershell
git add f1rl/learning/algorithms/ppo.py tests/learning/algorithms/test_ppo_config.py tests/learning/algorithms/test_ppo_model.py tests/integration/test_vectorized_ppo_integration.py
git commit -m "feat(learning): train ppo with vector environments"
```

### Task 4: Windows SubprocVecEnv smoke and diagnostic throughput benchmark

**Files:**
- Create: `tests/integration/helpers/subproc_vector_smoke.py`
- Modify: `tests/integration/test_vectorized_ppo_integration.py`
- Create: `scripts/benchmark_vector_envs.py`

**Interfaces:**
- Consumes: Task 2 `make_vector_env`; Task 1 configs/factory; Python executable/multiprocessing spawn.
- Produces: guarded subprocess acceptance helper; manual benchmark entrypoint.

- [ ] **Step 1: Write failing guarded subprocess integration test**

From pytest, launch `tests/integration/helpers/subproc_vector_smoke.py` with `sys.executable` in a child process. Require exit code 0 and parse/assert reported backend=`subproc`, `n_envs>=2`, reset shape `(N, obs_size)`, step reward/done vector shapes, and successful close marker. Include a finite timeout so a recursive spawn/hang fails loudly.

- [ ] **Step 2: Write guarded helper script**

The helper defines `main()` and calls it only under `if __name__ == "__main__":`. It builds a deterministic factory, constructs `VectorEnvConfig(n_envs=2, backend="subproc", start_method="spawn")`, resets/steps, prints machine-readable diagnostics, and closes the VecEnv in `finally` if construction succeeded.

- [ ] **Step 3: Run subprocess acceptance and verify GREEN/cleanup**

Run only the guarded subprocess test first. Expected: helper exits 0 without recursive process creation; repeated invocation succeeds. Inspect no intentionally retained F1-RL child process/session remains after helper exit; do not kill unrelated host processes.

- [ ] **Step 4: Write benchmark CLI contract tests or import-level unit tests**

Pin parser/input validation for explicit backend and positive env counts, measurement result fields `backend/n_envs/transitions/elapsed_seconds/transitions_per_second`, and no performance threshold/winner selection. Keep timing assertions to positivity/finite values only.

- [ ] **Step 5: Implement `scripts/benchmark_vector_envs.py`**

Use a `main()` + `__main__` guard. Accept explicit backend and one/more environment counts; recommended manual ladder remains Dummy `1,4,8,16`, Subproc `4,8,16`. Step batched sampled/neutral actions for an explicit finite transition budget, time with a monotonic clock, print diagnostics, and close VecEnv in `finally`.- [ ] **Step 6: Smoke benchmark both backends without treating throughput as correctness**

Run a tiny explicit diagnostic such as Dummy `1,4` and Subproc `2` with a small transition budget. Expected: both commands exit 0 and report positive finite elapsed/throughput values; do not compare or assert a winner.

- [ ] **Step 7: Run Task 4 integration regressions**

Run guarded subprocess tests twice, vector integration suite, then full pytest. Expected: no hang/orphan symptom and all suites green.

- [ ] **Step 8: Commit Task 4**

```powershell
git add tests/integration/helpers/subproc_vector_smoke.py tests/integration/test_vectorized_ppo_integration.py scripts/benchmark_vector_envs.py
git commit -m "test(learning): verify vector subprocess training"
```

### Task 5: Optional public training surface, documentation, boundaries, and final acceptance

**Files:**
- Create: `f1rl/learning/training/__init__.py`
- Modify: `f1rl/learning/training/README.md`
- Modify: `f1rl/learning/algorithms/README.md`
- Modify: `f1rl/learning/README.md`
- Modify/Test: `tests/integration/test_vectorized_ppo_integration.py`
- Inspect: `f1rl/learning/__init__.py`, lower-layer source tree, Git status/worktree topology.

**Interfaces:**
- Consumes: Tasks 1-4 completed APIs.
- Produces: public optional import `from f1rl.learning.training import F1RLEnvFactory, VectorEnvConfig, VectorTrainingConfig, make_vector_env`; verified stacked branch ready for an integration decision.

- [ ] **Step 1: Write failing public-export tests**

Assert `f1rl.learning.training.__all__`/direct imports expose exactly the four approved public symbols; `_RankedEnvFactory` stays private; `f1rl.learning` has none of these attributes and does not import the training package.

- [ ] **Step 2: Write dependency-boundary/source-scan tests**

Assert direct `stable_baselines3` imports in production are limited exactly to `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`; no F1-RL production module imports `torch`; `env_factory.py`, sandbox/runtime/Gym/action/observation/reward/episode are SB3/Torch-free.

- [ ] **Step 3: Pin documentation contract**

Assert training docs mention same-track vectorization, explicit dummy/subproc choice, Windows spawn + `__main__`, `vectorized-ppo.yml`, caller-owned `close()`, and diagnostic-only benchmark. Algorithm docs must mention single-env/VecEnv training with single-observation prediction. Learning docs must mark vectorized PPO implemented while multi-track/trainer/evaluation remain future work.

- [ ] **Step 4: Run Task 5 tests and verify RED**

Expected: public exports/docs tests fail before `training/__init__.py` and README updates; already-implemented source-boundary checks should pass.

- [ ] **Step 5: Implement optional public exports and docs**

Create `training/__init__.py` importing only the approved four symbols. Do not modify `f1rl.learning.__init__` to import training. Update the three READMEs without making throughput/policy-quality/optimal-env-count claims.
- [ ] **Step 6: Run focused final acceptance under Python 3.12**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/training tests/learning/algorithms/test_ppo_config.py tests/learning/algorithms/test_ppo_model.py tests/integration/test_vectorized_ppo_integration.py -q
```
Expected: all vector/config/factory/PPO/subprocess/boundary tests PASS.

- [ ] **Step 7: Run fresh full verification**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q f1rl scripts
```
Then run source scans for forbidden SB3/Torch imports and `git diff --check`.
Expected: full suite green, compileall PASS, dependency boundaries PASS, diff check PASS.

- [ ] **Step 8: Inspect changed production scope**

Compare against stacked base `3c29ad340d445ab4458c3e3c0b9b29fd647aa4a1`. Production changes must be limited to `f1rl/learning/training/**`, `f1rl/learning/algorithms/ppo.py`, the approved learning/algorithm/training READMEs, `configs/training/vectorized-ppo.yml`, and `scripts/benchmark_vector_envs.py`. Any sandbox/runtime/Gym/action/observation/reward/episode production diff is a stop condition.

- [ ] **Step 9: Re-run Windows subprocess helper on committed candidate**

After all code/doc changes are present in the candidate working tree, invoke the guarded helper twice with the exact Python 3.12 interpreter used for tests. Expected: both exit 0 and close cleanly; this is fresh evidence against spawn recursion/leak regressions.

- [ ] **Step 10: Commit Task 5**

```powershell
git add f1rl/learning/training/__init__.py f1rl/learning/training/README.md f1rl/learning/algorithms/README.md f1rl/learning/README.md tests/integration/test_vectorized_ppo_integration.py
git commit -m "docs(learning): expose vectorized ppo training"
```

- [ ] **Step 11: Whole-branch review before integration**

Review `3c29ad3..HEAD` against the approved vector spec, with the five Review Focus items checked deliberately. After the review/fix pass, rerun the full suite, compileall, dependency scans, diff-check, and guarded subprocess helper twice on the exact committed HEAD. Critical/Important findings require one explicit RED→GREEN fix pass plus a fresh full suite/subprocess smoke. Minor findings are recorded rather than silently expanding scope.

## Completion Evidence

Before claiming the branch complete, report:

```text
branch / exact HEAD / stacked base
commit list
Python / SB3 / PyYAML / PyTorch resolved versions
VectorTrainingConfig checked-in defaults
DummyVecEnv slots + batched shapes
independent/auto-reset terminal_observation evidence
PPO single-env + vector rollout-validation evidence
real vectorized PPO smoke num_timesteps
save/load + continued vector learning
Windows SubprocVecEnv helper runs / close result
benchmark smoke outputs (diagnostic only)
focused pytest result
full pytest result
Python 3.12 compileall
dependency boundaries
git diff --check
changed production scope
whole-branch review findings/rulings
push/merge status
```
