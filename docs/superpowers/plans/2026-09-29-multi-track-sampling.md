# Multi-Track Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic reset-time sampling of authored training tracks to vectorized PPO while preserving one shared non-track environment contract.

**Architecture:** `TrainingScenario` reuses the existing complete `F1RLEnvFactory`; `TrainingScenarioSet` validates one semantic training contract and canonical Gym spaces. A private per-rank Gym wrapper in `vector_env.py` owns its sampler and creates a new fixed-track inner `F1RLEnv` on every reset. Existing `make_vector_env()` continues to provide the unchanged same-track path.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3 2.9.0, PyYAML 6.0.3, PyTorch 2.14.0+cpu, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-multi-track-sampling-design.md`

## Global Constraints

- Stack on `models/ppo-baseline` at `19df1efae5cfa78162a73d998af798f4082ceec4`; do not push or merge.
- Keep sandbox -> runtime -> learning -> training/algorithms dependency direction; add no PyTorch import and keep SB3 imports confined to the existing PPO/vector layer.
- Do not modify `F1RLEnv`, sandbox, runtime, observations, rewards, episode semantics, PPO algorithm behavior, or hyperparameters.
- `make_vector_env(factory, config)` remains behaviorally compatible as the same-track API.
- A scenario set may vary only `TrackWorld`, track-dependent `initial_state`, and `track_id`; all other factory inputs must be equal.
- Validate compatible canonical action and observation Gym spaces before the first reset; wrapper construction must not sample an episode.
- `sampling_seed` exclusively controls scenario selection. Derive each rank with the approved SHA-256 formula; never use Python `hash()` and do not let `reset(seed=...)` alter the stream.
- Every reset builds a new fixed-track inner environment. Close replaced/current/partially-created environments according to the spec.
- Keep `track_id` out of observations; add it to reset and transition info, including terminal transitions before SB3 auto-reset.
- Use explicit `spawn` for `SubprocVecEnv`; all captured environment callables must remain top-level and cloudpickle-safe on Windows.
- Add no new dependencies or YAML fields. Callers own returned `VecEnv.close()`.

## Review Focus

- A semantically different observation encoder with the same Box shape must be rejected as a shared-contract mismatch. Pin in Task 1 with two factories that differ only in encoder configuration.
- A wrapper must expose canonical spaces and leave its first selected ID unchanged until `reset()`. Pin in Task 2 by comparing identical wrappers before and after construction, then after their first reset.
- A `reset(seed=...)` call must forward Gym seeding without changing the next scenario-selection sequence. Pin in Task 2 by comparing equal-length seeded and unseeded reset sequences from equal sampler seeds.
- A failed inner construction/reset must close the partial environment and leave the wrapper without a stale active environment. Pin in Task 2 with a factory whose created environment raises during reset.
- A terminal `step()` info must retain the completed scenario ID when `DummyVecEnv` adds `terminal_observation` and auto-resets. Pin in Task 3 with a one-step terminal track and a distinct next scenario.

---

## File Structure

```text
f1rl/learning/training/
├── env_factory.py             # Existing F1RLEnvFactory plus scenario/value-set validation
├── vector_env.py              # Existing same-track builder plus private sampled wrapper and public multi-track builder
├── __init__.py                # Optional training-package exports
└── README.md                  # Training construction and seed/lifecycle contract

f1rl/learning/README.md        # Mark multi-track sampling implemented and evaluation pending

tests/learning/training/
├── test_env_factory.py        # Existing factory regressions
├── test_scenarios.py          # Scenario-set semantic contract and canonical-space validation
└── test_vector_env.py         # Wrapper lifecycle, deterministic sampling, and DummyVecEnv behavior

tests/integration/
├── helpers/subproc_multi_track_smoke.py  # Guarded Windows-spawn helper
└── test_vectorized_ppo_integration.py    # Spawn, PPO, exports/docs, and dependency-boundary coverage
```

### Task 1: Scenario values, shared contract, and canonical spaces

**Files:**
- Modify: `f1rl/learning/training/env_factory.py`
- Create: `tests/learning/training/test_scenarios.py`
- Modify: `tests/learning/training/test_env_factory.py`

**Interfaces:**
- Consumes: existing `F1RLEnvFactory.create() -> F1RLEnv`.
- Produces: `TrainingScenario(track_id: str, factory: F1RLEnvFactory)` and `TrainingScenarioSet(scenarios: tuple[TrainingScenario, ...])`.
- Produces: read-only `TrainingScenarioSet.action_space` and `.observation_space` canonical Gym spaces for vector-wrapper construction.

- [ ] **Step 1: Write failing value-validation tests**

Create scenario fixtures from the existing deterministic training factory. Add tests that reject blank/non-string IDs, non-factory values, non-tuple/empty scenario collections, and duplicate IDs; assert a frozen scenario/set preserves declaration order.

- [ ] **Step 2: Write failing shared-contract and space tests**

Add tests that accept two scenarios differing only in `TrackWorld`, track-dependent `WorldState`, and ID. Parameterize separate failures where factories differ only in `vehicle_parameters`, `simulation_config`, observation encoder sensor configuration/normalization, reward composer, or episode config. Assert canonical action/observation spaces equal those of the compatible factory and use test doubles returned by `factory.create()` to prove mismatched Box type, shape, dtype, low, or high bounds are rejected and all temporary environments are closed.

- [ ] **Step 3: Run the focused tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_scenarios.py tests/learning/training/test_env_factory.py -q`

Expected: FAIL during collection because `TrainingScenario` and `TrainingScenarioSet` do not exist.

- [ ] **Step 4: Implement `TrainingScenario` and `TrainingScenarioSet` in `f1rl/learning/training/env_factory.py`**

Use frozen, slotted dataclasses. Validate exact shared factory fields before testing canonical Gym-space compatibility. Create temporary environments only to obtain/compare spaces; close each on both success and failure. Store canonical spaces on the set without starting an episode or introducing a generic scenario/dataset hierarchy.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_scenarios.py tests/learning/training/test_env_factory.py -q`

Expected: PASS. Existing factory freshness/independent-runtime tests remain green.

- [ ] **Step 6: Run the affected regression suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_env_factory.py tests/learning/training/test_vector_env.py tests/learning/test_gym_environment.py -q`

Expected: PASS; no existing same-track behavior changes.

- [ ] **Step 7: Commit Task 1**

```powershell
git add f1rl/learning/training/env_factory.py tests/learning/training/test_scenarios.py tests/learning/training/test_env_factory.py
git commit -m "feat(learning): define training scenarios"
```

### Task 2: Sampled wrapper lifecycle, deterministic rank streams, and DummyVecEnv construction

**Files:**
- Modify: `f1rl/learning/training/vector_env.py`
- Modify: `tests/learning/training/test_vector_env.py`

**Interfaces:**
- Consumes: `TrainingScenarioSet.action_space`, `.observation_space`, `F1RLEnvFactory.create()`, and `VectorEnvConfig`.
- Produces: private `_SampledF1RLEnv(scenarios: TrainingScenarioSet, sampling_seed: int, rank: int)` and private top-level `_RankedScenarioEnvFactory` callable.
- Produces: `make_multi_track_vector_env(scenarios: TrainingScenarioSet, config: VectorEnvConfig, sampling_seed: int) -> VecEnv`.

- [ ] **Step 1: Write failing wrapper-construction, sampler, and builder tests**

Import the private wrapper only within this test module. Assert constructor input validation rejects wrong scenario/config/seed/rank values; assert it immediately exposes the scenario-set canonical spaces, has no active inner environment, and has made no scenario draw. Use identical pools/seeds/ranks to assert equal first selections, a fresh wrapper of the same rank to prove rank-local sequence reproducibility, and a known different seed to assert a different deterministic sequence. Add public-builder tests that reject invalid arguments and construct a `DummyVecEnv` with the requested number of distinct wrappers.

- [ ] **Step 2: Write failing reset, info, and seed-isolation tests**

Assert each wrapper reset constructs a distinct inner `F1RLEnv`, returns the selected `track_id` only in info, and never changes observation shape/content to carry identity. Compare equal wrappers reset the same number of times, with explicit `reset(seed=...)` used only on one: their scenario-ID sequences must remain equal. Through `DummyVecEnv`, assert batched shapes and that a completed slot retains the scenario ID that produced its terminal transition while auto-resetting without advancing a sibling wrapper's sequence.

- [ ] **Step 3: Write failing lifecycle-failure tests**

Use a controlled factory/test environment to assert replacing an active episode closes the old inner environment; `close()` closes the active environment; and an inner reset failure closes the partial new environment, clears the active reference, raises the original exception, and leaves no old environment available for a later `step()`.

- [ ] **Step 4: Run focused tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_vector_env.py -q`

Expected: FAIL because the sampled wrapper, rank seed derivation, and multi-track builder do not exist.

- [ ] **Step 5: Implement private sampler/wrapper and `make_multi_track_vector_env()` in `f1rl/learning/training/vector_env.py`**

Keep `_RankedEnvFactory` and `make_vector_env()` unchanged. Derive each rank seed exactly as `int.from_bytes(hashlib.sha256(f"f1rl/multi-track-sampling/v1/{sampling_seed}/{rank}".encode("utf-8")).digest(), "big")` and initialize a private `random.Random`; do not use `hash()`. Assign canonical spaces during wrapper construction without drawing. On reset, close/clear any old inner environment, draw exactly one scenario, create/reset one new inner environment, and clean up/reset active state if creation or reset raises. Forward Gym reset seeds only to Gym/inner reset behavior, never the selection RNG. Overlay the current scenario ID on reset and step info before returning terminal transitions.

- [ ] **Step 6: Run focused tests to verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_vector_env.py -q`

Expected: PASS, including no hidden episode-zero sampling and all cleanup paths.

- [ ] **Step 7: Run same-track regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_vector_env.py tests/integration/test_vectorized_ppo_integration.py::test_real_dummy_vector_env_reset_step_smoke -q`

Expected: PASS. Existing `make_vector_env()` retains the original same-track batched observations and auto-reset behavior.

- [ ] **Step 8: Commit Task 2**

```powershell
git add f1rl/learning/training/vector_env.py tests/learning/training/test_vector_env.py
git commit -m "feat(learning): sample training scenarios per reset"
```

### Task 3: Windows-spawn and PPO integration

**Files:**
- Create: `tests/integration/helpers/subproc_multi_track_smoke.py`
- Modify: `tests/integration/test_vectorized_ppo_integration.py`

**Interfaces:**
- Consumes: `make_multi_track_vector_env(scenarios, config, sampling_seed) -> VecEnv` and existing `PPOModel.create()` / `.learn()`.
- Produces: guarded Windows-spawn acceptance evidence and a deterministic CPU PPO smoke test over a multi-track `DummyVecEnv`.

- [ ] **Step 1: Write a failing guarded spawn integration test**

Make the integration test launch `tests/integration/helpers/subproc_multi_track_smoke.py` with `sys.executable`, a finite timeout, and JSON assertions for `backend="subproc"`, two environments, reset/step shapes, transition track IDs, and clean close.

- [ ] **Step 2: Run the guarded spawn test to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_vectorized_ppo_integration.py -k multi_track_subproc -q`

Expected: FAIL because the guarded helper does not exist.

- [ ] **Step 3: Implement the guarded spawn helper**

Add `tests/integration/helpers/subproc_multi_track_smoke.py` with a `main()` invoked only under `if __name__ == "__main__":`. Use `VectorEnvConfig(backend="subproc", start_method="spawn")`, a top-level scenario fixture, and machine-readable diagnostics; no mutable registry or closure may cross the spawn boundary.

- [ ] **Step 4: Add PPO multi-track smoke coverage**

Use a two-slot multi-track `DummyVecEnv`, existing CPU `PPOConfig(n_steps=4, batch_size=4, n_epochs=1)`, and `PPOModel`. Assert construction and a short `learn()` call advance `num_timesteps`; make no reward-improvement or track-completion assertion. Close the VecEnv in `finally`. The existing PPO binding is expected to satisfy this compatibility coverage without a production-code change.


- [ ] **Step 5: Run focused integration tests to verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_vector_env.py tests/integration/test_vectorized_ppo_integration.py -q`

Expected: PASS for DummyVecEnv sampling, terminal info, guarded spawn, and PPO smoke training.

- [ ] **Step 6: Re-run the spawn helper twice**

Run: `.\.venv\Scripts\python.exe tests/integration/helpers/subproc_multi_track_smoke.py` twice.

Expected: both commands exit 0 and report clean closure; do not claim throughput or PPO determinism.

- [ ] **Step 7: Commit Task 3**

```powershell
git add tests/integration/helpers/subproc_multi_track_smoke.py tests/integration/test_vectorized_ppo_integration.py
git commit -m "test(learning): verify multi-track vector training"
```

### Task 4: Public surface, documentation, and source boundaries

**Files:**
- Modify: `f1rl/learning/training/__init__.py`
- Modify: `f1rl/learning/training/README.md`
- Modify: `f1rl/learning/README.md`
- Modify: `tests/integration/test_vectorized_ppo_integration.py`

**Interfaces:**
- Consumes: `TrainingScenario`, `TrainingScenarioSet`, and `make_multi_track_vector_env()` from Tasks 1-2.
- Produces: optional public imports while keeping `f1rl.learning` free of training/SB3 imports, plus documentation that distinguishes implemented sampling from future evaluation.

- [ ] **Step 1: Write failing public-surface, documentation, and boundary tests**

Update the exact `training.__all__` assertion to include `TrainingScenario`, `TrainingScenarioSet`, and `make_multi_track_vector_env`, while asserting they are absent from `f1rl.learning`. Extend documentation assertions to require reset-time sampling, explicit `sampling_seed`, scenario IDs in info only, shared non-track contracts, canonical spaces, and Windows spawn. Retain the direct-import scan: SB3 remains limited to `ppo.py` and `vector_env.py`, no production module imports `torch`, and sandbox/runtime do not import learning.

- [ ] **Step 2: Run focused tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_vectorized_ppo_integration.py -q`

Expected: FAIL because exports and docs still describe multi-track sampling as future work.

- [ ] **Step 3: Export and document the approved surface**

Add only the three approved multi-track names to `f1rl.learning.training.__all__`. Do not alter `f1rl.learning.__init__`. Update the two READMEs with the construction, stream, lifecycle, and scope rules from the spec; mark evaluation/generalization as future work and make no performance or policy-quality claim.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_vectorized_ppo_integration.py -q`

Expected: PASS; optional dependency and lower-layer import boundaries remain intact.

- [ ] **Step 5: Commit Task 4**

```powershell
git add f1rl/learning/training/__init__.py f1rl/learning/training/README.md f1rl/learning/README.md tests/integration/test_vectorized_ppo_integration.py
git commit -m "docs(learning): expose multi-track sampling"
```

### Task 5: Whole-branch review and final verification

**Files:**
- Inspect: all files changed since `19df1efae5cfa78162a73d998af798f4082ceec4`

**Interfaces:**
- Consumes: completed Tasks 1-4 and the approved specification.
- Produces: fresh evidence on the exact committed branch head; no new feature scope.

- [ ] **Step 1: Review changed production scope and specification coverage**

Inspect `git diff --name-only 19df1ef..HEAD` and production diffs. Confirm production changes are limited to `f1rl/learning/training/**` and the two approved learning READMEs; reject any sandbox/runtime/Gym/PPO/config modification. Cross-check every approved invariant against its focused test.

- [ ] **Step 2: Run fresh focused acceptance tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/learning/training tests/integration/test_vectorized_ppo_integration.py -q`

Expected: PASS, including contract validation, seed isolation, lifecycle failure cleanup, DummyVecEnv, spawned subprocess, PPO smoke, exports, docs, and boundaries.

- [ ] **Step 3: Run the full verification suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q f1rl scripts
git diff --check 19df1efae5cfa78162a73d998af798f4082ceec4..HEAD
```

Expected: full suite PASS, compileall exits 0, and diff check reports no whitespace errors.

- [ ] **Step 4: Re-run exact Windows acceptance and inspect repository state**

Run the guarded multi-track spawn helper twice with the verified Python 3.12 interpreter. Then record branch name, exact HEAD, base, `git status --short`, and push/merge state. Expected: both helpers close cleanly, worktree is clean, and no push or merge occurred.

- [ ] **Step 5: Invoke finishing review workflow before handoff**

Use `superpowers:finishing-a-development-branch` only after all prior checks pass. Report its result and leave integration to explicit human authorization.
