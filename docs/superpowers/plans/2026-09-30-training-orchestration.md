# Training Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small reproducible training-run framework that loads root `config.yml`, runs same-track or multi-track PPO, writes checkpoints and `state.json`, and saves a self-describing final model.

**Architecture:** `TrainingRunConfig` resolves user input; `TrainingRunner` owns run identity, artifacts, lifecycle, and delegates environment/model construction to existing APIs. SB3 callback plumbing stays inside `ppo.py`; `runner.py` only receives project-level timestep progress.

**Tech Stack:** Python 3.12, dataclasses, pathlib/json/yaml, Gymnasium, Stable-Baselines3 2.9.0, pytest, Windows PowerShell.

**Spec:** `docs/superpowers/specs/2026-09-30-training-orchestration-design.md`

## Global Constraints

- Implement from `models/ppo-baseline` in an isolated Git worktree; do not push or merge without explicit user authorization.
- Root `config.yml` is immutable input and contains exactly `training`, `ppo`, and `vector_env` mappings; unknown keys are rejected.
- Defaults: `total_timesteps=50_000`, `checkpoint_every_steps=25_000`, `sampling_seed=0`, `artifacts_root=artifacts/runs`.
- Multi-track track IDs are derived from `TrainingScenarioSet`; single-track state records no invented track ID.
- `total_timesteps` is a requested minimum; persisted artifact/state counters use actual `PPOModel.num_timesteps`.
- Checkpoint thresholds are `interval, 2*interval, ... < requested_total_timesteps`; no duplicate checkpoint at the final target.
- Production SB3 imports remain limited to `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`; add no production `torch` imports.
- Production code contains no mocks, fakes, test switches, or test-only branches; test doubles live under `tests/learning/training/mocks/`.
- V1 adds no CLI, resume support, evaluation, HPO, TensorBoard/W&B, trainer service, or track loader.
- Current verification environment is CPU-only; do not claim CUDA training works on this machine.

## Review Focus

1. `artifacts_root` resolves to an existing file or unusable destination: fail before model training and never overwrite unrelated filesystem content — covered in Task 3 runner-artifact tests.
2. Two runs resolve to the same second-level run ID: reject the collision rather than reuse/overwrite the first run — covered in Task 3 naming/artifact tests.
3. A progress/checkpoint hook raises while PPO is learning: propagate the original hook error and let runner state become `failed` — covered in Tasks 2 and 4.
4. `VecEnv.close()` fails: after an otherwise completed run, propagate the cleanup error while keeping completed artifacts/state; if training is already failing, preserve that training exception as primary — covered in Tasks 4 and 5.
5. `ppo.seed=None`: run IDs/model names use `seednone`, state remains JSON-serializable, and no deterministic-seed claim is implied — covered in Tasks 1 and 3.

---

## File Structure

**Create:**
- `config.yml` — checked-in orchestration input with resolved V1 defaults.
- `f1rl/learning/training/runner.py` — run naming, state persistence, checkpoint scheduling, lifecycle, `TrainingRunner`, and `TrainingRunResult`.
- `artifacts/runs/README.md` — tracked documentation placeholder while generated run children stay ignored.
- `tests/learning/training/test_run_config.py` — orchestration config contract.
- `tests/learning/training/test_training_runner.py` — runner/naming/state/lifecycle behavior.
- `tests/learning/training/mocks/__init__.py` — test-double package.
- `tests/learning/training/mocks/mock_ppo.py` — deterministic PPO-model doubles used only by runner tests.
- `tests/learning/training/mocks/mock_vec_env.py` — VecEnv/lifecycle doubles used only by runner tests.
- `tests/learning/training/mocks/mock_factories.py` — environment/scenario test helpers used only by runner tests.
- `tests/learning/training/mocks/mock_io.py` — failing state-writer helpers used only by runner tests.
- `tests/integration/test_training_orchestration_integration.py` — tiny real-PPO end-to-end run artifacts.

**Modify:**
- `f1rl/learning/training/config.py` — add `TrainingRunConfig` YAML parsing/validation without changing existing vector contracts.
- `f1rl/learning/algorithms/ppo.py` — add optional project-owned progress hook adapted internally to SB3 callbacks.
- `f1rl/learning/training/__init__.py` — export orchestration public types.
- `.gitignore` — ignore `artifacts/runs/*` while retaining `artifacts/runs/README.md`.
- `artifacts/README.md` — document `runs/` ownership.
- `f1rl/learning/training/README.md` and `f1rl/learning/README.md` — mark orchestration implemented and state non-goals.
- `tests/learning/algorithms/test_ppo_model.py` — prove progress-hook semantics without putting test doubles in production.

### Task 1: Define orchestration configuration and checked-in root YAML

**Files:**
- Create: `config.yml`
- Create: `tests/learning/training/test_run_config.py`
- Modify: `f1rl/learning/training/config.py`

**Interfaces:**
- Consumes: existing `PPOConfig`, `VectorEnvConfig`.
- Produces: `TrainingRunConfig(ppo, vector_env, total_timesteps=50_000, checkpoint_every_steps=25_000, sampling_seed=0, artifacts_root=Path("artifacts/runs"))` and `TrainingRunConfig.from_yaml(path: str | Path) -> TrainingRunConfig`.

- [ ] **Step 1: Write failing config-default and validation tests**

Add tests asserting exact defaults, frozen dataclass behavior, rejection of bool/non-positive timestep values, invalid `sampling_seed`, and `ppo.seed=None` acceptance.

- [ ] **Step 2: Write failing YAML-contract tests**

Assert root mappings are exactly `training`, `ppo`, `vector_env`; unknown root/nested keys, malformed documents, invalid paths, blank `artifacts_root`, and non-mapping sections raise `ValueError`. Assert omitted training fields use defaults, relative `artifacts_root` resolves against the YAML file's parent, and absolute paths remain absolute.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_run_config.py -q`
Expected: FAIL because `TrainingRunConfig` and root `config.yml` do not exist.

- [ ] **Step 4: Implement `TrainingRunConfig` minimally**

Keep existing `VectorTrainingConfig` unchanged. Reuse dataclass-field introspection for strict PPO/vector keys; validate the `training` mapping separately and resolve `artifacts_root` during `from_yaml()`.

- [ ] **Step 5: Add checked-in `config.yml` with exact V1 defaults and run GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_run_config.py tests\learning\training\test_vector_config.py tests\learning\algorithms\test_ppo_config.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit: `feat(learning): define training run configuration`

### Task 2: Add PPO progress-hook plumbing without widening the SB3 boundary

**Files:**
- Modify: `f1rl/learning/algorithms/ppo.py`
- Modify: `tests/learning/algorithms/test_ppo_model.py`

**Interfaces:**
- Consumes: existing `PPOModel.learn(total_timesteps)` behavior.
- Produces: `PPOModel.learn(total_timesteps: int, *, progress_hook: Callable[[int], None] | None = None) -> PPOModel`; `progress_hook` receives actual cumulative `num_timesteps` and SB3 remains private to `ppo.py`.

- [ ] **Step 1: Write failing tests for hook delivery and backward compatibility**

Assert `learn(5)` still calls backend with `reset_num_timesteps=False`; when a hook is supplied, observed values are cumulative actual timesteps; `None` preserves existing behavior.

- [ ] **Step 2: Write failing real-PPO hook tests**

Use `PPOModel.create(_env(), PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device="cpu"))` with a tiny learn budget. Assert progress values are monotonic actual cumulative timesteps, the last observed value equals `model.num_timesteps`, and a sentinel exception raised by the hook escapes unchanged. Do not add a new inline mock backend for these tests.

- [ ] **Step 3: Run focused PPO tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\algorithms\test_ppo_model.py -q`
Expected: FAIL on the new `progress_hook` argument/behavior.

- [ ] **Step 4: Implement a private SB3 callback adapter in `ppo.py`**

Keep the callback class/private adapter in `ppo.py`; invoke the project hook with `int(model.num_timesteps)` from SB3 callback progress. Do not add callback imports anywhere else in production.

- [ ] **Step 5: Run PPO unit + integration regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\algorithms\test_ppo_model.py tests\integration\test_ppo_baseline_integration.py tests\integration\test_vectorized_ppo_integration.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit: `feat(learning): expose ppo training progress hook`

### Task 3: Add deterministic run/artifact/state primitives

**Files:**
- Create: `f1rl/learning/training/runner.py`
- Create: `tests/learning/training/test_training_runner.py`
- Create: `artifacts/runs/README.md`
- Modify: `.gitignore`
- Modify: `artifacts/README.md`

**Interfaces:**
- Consumes: `TrainingRunConfig`.
- Produces: `TrainingRunResult(run_id: str, run_directory: Path, final_model_path: Path, checkpoint_paths: tuple[Path, ...], requested_total_timesteps: int, actual_total_timesteps: int)`; private `_checkpoint_thresholds(total_timesteps, interval)`, `_build_run_id(started_at, mode, ppo_seed, sampling_seed)`, `_artifact_stem(mode, ppo_seed, sampling_seed)`, and JSON state-writing helpers used by later runner tasks.
- `state.json` keys are exactly the spec snapshot fields: run/status/mode, requested+actual timesteps, checkpoint interval, latest checkpoint, final model, resolved PPO/vector config, sampling seed/null, and training track IDs.

- [ ] **Step 1: Write failing naming/schedule tests**

Assert second-precision IDs, `seednone`, single-track vs multi-track seed/sample names, exact `< requested_total_timesteps` thresholds, `None -> ()`, and one actual archive name per progress update even if it satisfies multiple thresholds.

- [ ] **Step 2: Write failing artifact/state tests**

Assert run-directory collision is rejected; `artifacts_root` resolving to a file is rejected without mutation; initial `state.json` is valid JSON with `running`, zero actual steps, no final model, resolved config snapshot, and correct empty/derived track metadata inputs.

- [ ] **Step 3: Run focused runner tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_training_runner.py -q`
Expected: FAIL because runner primitives do not exist.

- [ ] **Step 4: Implement runner primitives and artifact directory contract**

Create the immutable result type and private pure helpers. Create run/checkpoint directories only after validating the artifact root and collision. Serialize resolved config/state with relative artifact references. Add `.gitignore` rules for `artifacts/runs/*` plus the tracked README exception.

- [ ] **Step 5: Run GREEN and filesystem regression checks**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_training_runner.py -q`
Expected: PASS.

Run: `git check-ignore -v artifacts\runs\example\model.zip`
Expected: matched by the new generated-run ignore rule.

- [ ] **Step 6: Commit**

Commit: `feat(learning): define training run artifacts`

### Task 4: Implement same-track TrainingRunner lifecycle with isolated test doubles

**Files:**
- Create: `tests/learning/training/mocks/__init__.py`
- Create: `tests/learning/training/mocks/mock_ppo.py`
- Create: `tests/learning/training/mocks/mock_vec_env.py`
- Create: `tests/learning/training/mocks/mock_factories.py`
- Create: `tests/learning/training/mocks/mock_io.py`
- Modify: `f1rl/learning/training/runner.py`
- Modify: `tests/learning/training/test_training_runner.py`

**Interfaces:**
- Produces: `TrainingRunner(config: TrainingRunConfig)` and `run_single(factory: F1RLEnvFactory) -> TrainingRunResult`.
- Uses: `make_vector_env(factory, config.vector_env)`, `PPOModel.create(env, config.ppo)`, and `PPOModel.learn(..., progress_hook=...)`.

- [ ] **Step 1: Add test-only doubles outside production**

`MockPPOModel` must support deterministic `create`, `learn`, `num_timesteps`, and `save`; `MockVecEnv` must record/optionally fail `close()`. `mock_factories.py` supplies valid F1-RL factory/scenario fixtures, and `mock_io.py` supplies failing state-write helpers, without adding any `if testing` path to production.

- [ ] **Step 2: Write failing same-track lifecycle tests**

Assert `run_single()` selects `make_vector_env`, writes `running`, saves intermediate checkpoints when progress crosses configured thresholds, saves the final self-describing model, writes `completed`, returns `TrainingRunResult`, and closes the VecEnv.

- [ ] **Step 3: Write failing `ppo.seed=None` test**

Assert run ID and filenames contain `seednone`, `state.json` remains JSON-serializable, and single-track state has `sampling_seed: null` plus an empty training-track list.

- [ ] **Step 4: Run focused tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_training_runner.py -q`
Expected: FAIL on missing `TrainingRunner.run_single()` lifecycle.

- [ ] **Step 5: Implement the minimal same-track lifecycle**

Use one private shared `_run(...)` path owned by `TrainingRunner`; bind the same-track env builder, create PPO, install the project progress hook, persist checkpoint state after each saved checkpoint, save final model only after successful learning, and close the VecEnv in `finally`.

- [ ] **Step 6: Run GREEN and same-track regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_training_runner.py tests\learning\training\test_vector_env.py tests\integration\test_vectorized_ppo_integration.py::test_real_vectorized_ppo_training_save_load_and_continue -q`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit: `feat(learning): orchestrate same-track training runs`

### Task 5: Add multi-track orchestration and failure/cleanup semantics

**Files:**
- Modify: `f1rl/learning/training/runner.py`
- Modify: `tests/learning/training/test_training_runner.py`
- Modify: `tests/learning/training/mocks/mock_ppo.py`
- Modify: `tests/learning/training/mocks/mock_vec_env.py`

**Interfaces:**
- Produces: `run_multi(scenarios: TrainingScenarioSet) -> TrainingRunResult`.
- Uses: `make_multi_track_vector_env(scenarios, config.vector_env, config.sampling_seed)` and shared `_run(...)` lifecycle.

- [ ] **Step 1: Write failing multi-track metadata/naming tests**

Assert scenario IDs are derived in declaration order; run ID and model/checkpoint names contain both `seed<ppo>` and `sample<sampling_seed>`; no track ID is accepted from YAML or invented by runner code.

- [ ] **Step 2: Write failing checkpoint coalescing/state tests**

Drive progress updates that cross one threshold and then multiple thresholds. Assert one archive per actual timestep update, checkpoint filename uses actual `num_timesteps`, `latest_checkpoint` is updated only after a successful save, and a final-target threshold is never scheduled.

- [ ] **Step 3: Write failing training/checkpoint failure tests**

Assert learning failure, checkpoint-save failure, and state-update failure produce `status="failed"` best-effort, preserve already written checkpoints, never create final model, close VecEnv, and re-raise the triggering exception.

- [ ] **Step 4: Write failing cleanup-failure tests**

For an otherwise completed run, make `VecEnv.close()` raise and assert the cleanup exception propagates while completed state/final artifact remain truthful. Separately configure training to raise sentinel A and close to raise sentinel B; assert caller observes sentinel A.

- [ ] **Step 5: Run focused tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_training_runner.py -q`
Expected: FAIL on multi-track and failure semantics.

- [ ] **Step 6: Implement multi-track binding and robust failure handling**

Reuse the same `_run(...)` path; supply mode/track metadata and multi-track env construction only at the boundary. On failure, capture latest available `model.num_timesteps` when possible, make a best-effort failed-state write, preserve the triggering exception, and always attempt VecEnv cleanup.

- [ ] **Step 7: Run GREEN plus multi-track regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training\test_training_runner.py tests\learning\training\test_scenarios.py tests\learning\training\test_vector_env.py tests\integration\test_vectorized_ppo_integration.py::test_real_multi_track_ppo_training_smoke -q`
Expected: PASS.

- [ ] **Step 8: Commit**

Commit: `feat(learning): orchestrate multi-track training runs`

### Task 6: Prove the real framework end-to-end, expose API, document, and verify

**Files:**
- Create: `tests/integration/test_training_orchestration_integration.py`
- Modify: `f1rl/learning/training/__init__.py`
- Modify: `f1rl/learning/training/README.md`
- Modify: `f1rl/learning/README.md`

**Interfaces:**
- Public exports: `TrainingRunConfig`, `TrainingRunResult`, `TrainingRunner` alongside existing training exports.

- [ ] **Step 1: Write a failing real-PPO orchestration smoke test**

Create a temporary YAML with CPU PPO (`n_steps=4`, `batch_size=4`, `n_epochs=1`), `n_envs=2`, `backend=dummy`, requested `total_timesteps=8`, checkpoint interval `4`, fixed PPO seed and sampling seed, and a temp artifacts root. Load it through `TrainingRunConfig.from_yaml()` and run real multi-track orchestration with the existing real scenario fixture.

Assert the run completes, actual timesteps are `>= 8`, `state.json` is `completed`, track IDs match the scenario declaration, one checkpoint/final model exist with self-describing names, and returned paths equal persisted state references.

- [ ] **Step 2: Run the smoke test and confirm RED before final wiring**

Run: `.\.venv\Scripts\python.exe -m pytest tests\integration\test_training_orchestration_integration.py -q`
Expected: FAIL until public exports/final integration wiring are complete.

- [ ] **Step 3: Export the public API and update documentation**

Export only `TrainingRunConfig`, `TrainingRunResult`, and `TrainingRunner`; keep private runner helpers private. Document root YAML -> runner -> run artifacts, framework-only status, no CLI/resume/evaluation, and retained `artifacts/policies` / `artifacts/evaluations` roles.

- [ ] **Step 4: Run focused orchestration + existing vector integration tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\learning\training tests\learning\algorithms\test_ppo_model.py tests\integration\test_training_orchestration_integration.py tests\integration\test_vectorized_ppo_integration.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the completed implementation**

Commit: `feat(learning): complete training orchestration`

- [ ] **Step 6: Run fresh full verification on the exact committed HEAD**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all tests PASS.

Run: `.\.venv\Scripts\python.exe -m compileall -q f1rl scripts`
Expected: exit code 0.

- [ ] **Step 7: Re-run Windows subprocess guards explicitly**

Run: `.\.venv\Scripts\python.exe -m pytest tests\integration\test_vectorized_ppo_integration.py::test_guarded_subproc_vector_smoke_script tests\integration\test_vectorized_ppo_integration.py::test_guarded_multi_track_subproc_smoke_script -q`
Expected: 2 passed.

- [ ] **Step 8: Verify dependency boundaries**

Run: `Get-ChildItem f1rl -Recurse -Filter *.py | Select-String -Pattern 'stable_baselines3'`
Expected: production matches only in `f1rl\learning\algorithms\ppo.py` and `f1rl\learning\training\vector_env.py`.

Run: `Get-ChildItem f1rl -Recurse -Filter *.py | Select-String -Pattern '^\s*(from|import)\s+torch\b'`
Expected: no matches.

- [ ] **Step 9: Verify whitespace, scope, and clean worktree**

Capture the implementation base with `git merge-base models/ppo-baseline HEAD`, then run `git diff --check <base>..HEAD`, inspect `git diff --stat <base>..HEAD` and `git diff --name-only <base>..HEAD`, and finish with `git status --short --branch`.

Expected: diff check exit 0; changed files stay within this plan's scope; worktree is clean.

- [ ] **Step 10: Whole-branch review before integration choice**

Use a fresh reviewer on the complete implementation branch. Fix any Critical/Important defects with targeted TDD, rerun affected tests plus the full verification gate, then invoke `finishing-a-development-branch`. Do not push or merge until the user explicitly chooses integration.
