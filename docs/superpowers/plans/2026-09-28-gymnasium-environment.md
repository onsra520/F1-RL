# Gymnasium Environment V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a standards-compliant fixed-track `F1RLEnv` that composes the existing deterministic simulation, observation, action, reward, race-progress, and episode-status primitives.

**Architecture:** Add one thin Gymnasium adapter at `f1rl/learning/gym_environment.py`. It owns framework-facing spaces, lifecycle, NumPy conversion, and composition order only; all racing semantics remain in existing primitives. Gymnasium/NumPy dependencies stop at the learning adapter boundary.

**Tech Stack:** Python 3.12+, Gymnasium `>=1.3,<2`, NumPy `>=2,<3`, pytest, existing F1-RL sandbox/runtime/learning primitives.

**Spec:** `docs/superpowers/specs/2026-09-28-gymnasium-environment-design.md`

## Global Constraints

- One `F1RLEnv` instance is permanently bound to one `TrackWorld`.
- `reset()` never randomizes or replaces the track in V1.
- `step()` returns Gymnasium's 5-tuple: observation, reward, terminated, truncated, info.
- Reward uses the authoritative pre-step route target; next observation uses the post-progress target.
- Invalid actions fail before simulation mutation and are never clamped/remapped.
- Any failure after simulation advancement makes the environment reset-required before re-raising.
- No Stable-Baselines3, PyTorch, PPO, vectorized env, rendering, or Gym global registration in this branch.
- Sandbox and runtime production code must not import Gymnasium or NumPy.
- Existing sandbox/runtime/learning semantics remain authoritative; the adapter must not duplicate them.

## Review Focus

- A boolean/object-dtype/non-numeric action must not sneak through NumPy coercion; pin strict pre-step validation in Task 2.
- An observation encoder result with wrong length, non-finite values, or values outside the declared Box must fail deterministically; pin reset and post-step behavior in Tasks 1â€“2.
- Repeated `reset()` calls after active, terminal, truncated, or faulted states must fully recover simulation/progress lifecycle; pin in Tasks 1â€“2.
- A custom reward term that raises after `Simulation.step()` must force reset without advancing stored race progress; pin in Task 2.
- Gymnasium checker behavior must not cause hidden track mutation, rendering assumptions, or registration side effects; pin in Task 3.

## File Structure

- Modify `pyproject.toml` â€” declare Gymnasium and NumPy runtime dependencies.
- Create `f1rl/learning/gym_environment.py` â€” Gym-facing spaces, lifecycle, reset/step composition, info diagnostics.
- Modify `f1rl/learning/__init__.py` â€” re-export `F1RLEnv` while preserving current exports.
- Modify `f1rl/learning/README.md` â€” mark Gym composition complete and PPO/model as next layer.
- Create `tests/learning/test_gym_environment.py` â€” focused unit/contract coverage.
- Create `tests/integration/test_gym_environment_integration.py` â€” real stepping, route/reward timing, finish/timeout, checker acceptance.

---

### Task 1: Dependencies, spaces, constructor, and deterministic reset

**Files:**
- Modify: `pyproject.toml`
- Create: `f1rl/learning/gym_environment.py`
- Create: `tests/learning/test_gym_environment.py`

**Interfaces:**
- Consumes: `TrackWorld`, `WorldState`, `VehicleParameters`, `SimulationConfig`, `ObservationEncoder`, `RewardComposer`, `EpisodeConfig`.
- Produces: `F1RLEnv(gym.Env[np.ndarray, np.ndarray])` with constructor exactly matching the spec, plus Gym `action_space`, `observation_space`, `reset()`.

- [ ] **Step 1: Add Gymnasium/NumPy dependencies and install the project for this worktree**

Set runtime dependencies exactly to `gymnasium>=1.3,<2` and `numpy>=2,<3`, preserving the existing dev pytest extra. Install with `py -3.12 -m pip install -e ".[dev]"` so checker/tests run against declared requirements.

- [ ] **Step 2: Write failing constructor/space tests**

Pin `F1RLEnv.metadata == {"render_modes": []}`, `render_mode is None`, action Box low/high `[-1, 0]` / `[1, 1]`, shape `(2,)`, `np.float32`, and observation Box shape `(encoder.size,)` with exact per-feature bounds and `np.float32`.

- [ ] **Step 3: Write failing constructor-validation tests**

For each constructor argument, replace the valid concrete instance with a wrong object and assert `ValueError`. Also assert a track with zero or multiple finish lines fails through `RaceRoute.from_track` rather than being accepted by Gym-specific fallback logic.

- [ ] **Step 4: Write failing reset tests**

Assert `reset()` returns `(np.ndarray, info)`, observation is `float32` and inside `observation_space`, initial info has tick/time/progress/score zero-step diagnostics, repeated reset restores the exact same observation/state, different accepted seeds do not alter deterministic V1 state, empty `{}` options are accepted, and non-empty options raise `ValueError`.

- [ ] **Step 5: Pin malformed observation output on reset**

Use a test encoder/configuration stub or controlled replacement that returns the wrong length, non-finite data, and out-of-space data. Each reset must raise and leave the environment in reset-required state.

- [ ] **Step 6: Run RED**

Run: `python -m pytest tests/learning/test_gym_environment.py -q`
Expected: collection/import failure because `F1RLEnv` does not exist.

- [ ] **Step 7: Implement constructor, spaces, observation conversion helper, and reset**

Implement the spec constructor and `reset(*, seed=None, options=None)`. Derive `RaceRoute` from the track, build one `Simulation`, construct Box spaces from existing action/encoder contracts, bridge `Vec3 | None` to `NavigationTarget | None`, and centralize observation conversion/containment validation in one private helper.

- [ ] **Step 8: Run GREEN + existing regressions**

Run:
- `python -m pytest tests/learning/test_gym_environment.py -q`
- `python -m pytest tests/learning -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 9: Commit Task 1**

Commit: `feat(learning): add gymnasium environment reset contract`

---

### Task 2: Step composition, lifecycle, reward timing, and fault recovery

**Files:**
- Modify: `f1rl/learning/gym_environment.py`
- Modify: `tests/learning/test_gym_environment.py`
- Create: `tests/integration/test_gym_environment_integration.py`

**Interfaces:**
- Consumes: Task 1 `F1RLEnv`; existing `PolicyAction`, `map_policy_action`, `RewardTransition`, `RewardComposer`, `advance_race_progress`, `episode_status`.
- Produces: `F1RLEnv.step(action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]` and complete episode lifecycle semantics.

- [ ] **Step 1: Write failing lifecycle/action validation tests**

Assert step-before-reset raises `RuntimeError`; wrong action type, wrong shape, bool/object/non-numeric arrays, NaN/Inf, and out-of-range values raise before tick/progress mutation; a valid `action_space.sample()` action is accepted.

- [ ] **Step 2: Write failing one-step composition tests**

Assert one legal action advances exactly one simulation tick, reward is exactly `float(RewardComposer.evaluate(...).total)`, returned observation is `float32`/contained, and info exposes only the primitive diagnostics from the spec. Verify `reward_components` preserves contribution declaration order and float values.

- [ ] **Step 3: Write failing pre-target/post-target integration test**

Use a real track transition that enters the expected checkpoint. Assert the reward contribution is computed against the checkpoint target active before the step, while the returned observation encodes the next route target after progress advancement.

- [ ] **Step 4: Write failing termination/truncation tests**

Cover valid finish => `(True, False)`, unfinished max-step exhaustion => `(False, True)`, finish exactly on `max_steps` => `(True, False)`, and a second `step()` after either done state => `RuntimeError` until reset.

- [ ] **Step 5: Write failing score/reward separation test**

Cross a score pickup and prove `info["game_score"]` reflects sandbox scoring while Gym reward remains exactly the reward composer total with no score conversion.

- [ ] **Step 6: Write failing post-advance fault tests**

First force `Simulation.step()` itself to raise before returning and assert simulation/progress remain unchanged and the episode stays active. Then use a valid custom `RewardTerm` whose `evaluate()` raises after successful simulation advancement: assert tick advanced once, stored race progress did not advance, subsequent step raises reset-required, and `reset()` fully recovers. Repeat reset-required behavior for malformed/out-of-space post-step observation encoding.

- [ ] **Step 7: Run RED**

Run: `python -m pytest tests/learning/test_gym_environment.py tests/integration/test_gym_environment_integration.py -q`
Expected: failures because `step()`/lifecycle behavior is not implemented.

- [ ] **Step 8: Implement strict action conversion and `step()` composition**

Follow the spec order exactly: validate action; capture pre-step target/vehicle; map to control; call simulation step; only after that call returns successfully does the post-advance fault guard begin; evaluate reward with pre-step target; advance progress; compute status; encode using post-progress target; construct info; mark reset-required on done or any exception after successful simulation advancement. A `Simulation.step()` exception must leave the active lifecycle/progress unchanged.

- [ ] **Step 9: Run GREEN + regressions**

Run:
- `python -m pytest tests/learning/test_gym_environment.py tests/integration/test_gym_environment_integration.py -q`
- `python -m pytest tests/learning -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 10: Commit Task 2**

Commit: `feat(learning): compose deterministic gymnasium steps`

---

### Task 3: Gymnasium compliance, public API, docs, and dependency boundaries

**Files:**
- Modify: `f1rl/learning/__init__.py`
- Modify: `f1rl/learning/README.md`
- Modify: `tests/learning/test_gym_environment.py`
- Modify: `tests/integration/test_gym_environment_integration.py`

**Interfaces:**
- Consumes: complete Task 1â€“2 `F1RLEnv`.
- Produces: public `f1rl.learning.F1RLEnv` and a Gymnasium-checker-clean environment boundary.

- [ ] **Step 1: Write failing public-export regression test**

Assert `from f1rl.learning import F1RLEnv` is the same class as `f1rl.learning.gym_environment.F1RLEnv`, while all existing action/observation/reward/episode exports remain available.

- [ ] **Step 2: Write failing Gymnasium checker test**

Construct a valid environment and run `gymnasium.utils.env_checker.check_env(env, skip_render_check=True)`. Assert it completes without warnings/errors that indicate API non-compliance; do not register a global env ID.

- [ ] **Step 3: Write failing dependency-boundary tests**

Scan production sandbox/runtime sources and assert they contain no Gymnasium/NumPy imports. Scan `gym_environment.py` and assert it contains no Stable-Baselines3/PyTorch import, no `gymnasium.register`, and no rendering dependency or non-empty render modes.

- [ ] **Step 4: Write checker-side-effect tests**

Record the bound track identity/content before checker/reset calls and prove checker execution does not replace or mutate the authored track. Repeated checker-style resets must remain deterministic.

- [ ] **Step 5: Run RED**

Run: `python -m pytest tests/learning/test_gym_environment.py tests/integration/test_gym_environment_integration.py -q`
Expected: public export/docs or compliance assertions fail until Task 3 changes land.

- [ ] **Step 6: Re-export `F1RLEnv` and update learning docs**

Add `F1RLEnv` to `f1rl.learning` without removing any current public symbol. Update `f1rl/learning/README.md` to mark Gymnasium composition implemented and identify PPO/model integration as next; do not claim SB3/vectorization/rendering support.

- [ ] **Step 7: Run Gym compliance + regression suites**

Run:
- `python -m pytest tests/learning/test_gym_environment.py tests/integration/test_gym_environment_integration.py -q`
- `python -m pytest -q`

Expected: zero failures, including Gymnasium checker acceptance.

- [ ] **Step 8: Commit Task 3**

Commit: `test(learning): verify gymnasium environment contract`

---

### Task 4: Whole-branch acceptance and merge readiness

**Files:**
- Modify only if a discovered acceptance gap requires a tested fix: files already owned by Tasks 1â€“3.
- No new architecture or dependencies.

**Interfaces:**
- Consumes: completed `F1RLEnv` public surface.
- Produces: fresh evidence that the feature branch is safe to integrate into core, then `main` via the finishing workflow.

- [ ] **Step 1: Add paired-environment determinism acceptance test**

Build two independent `F1RLEnv` instances from equal immutable inputs. After equal resets and an identical finite action sequence, assert equal observations, rewards, terminated/truncated values, progress diagnostics, score diagnostics, and reset recovery.

- [ ] **Step 2: Run focused suite fresh under Python 3.12**

Run: `py -3.12 -m pytest tests/learning/test_gym_environment.py tests/integration/test_gym_environment_integration.py -q`
Expected: zero failures.

- [ ] **Step 3: Run full suite fresh under Python 3.12**

Run: `py -3.12 -m pytest -q`
Expected: zero failures.

- [ ] **Step 4: Run compilation and source-boundary gates**

Run:
- `py -3.12 -m compileall -q f1rl`
- source scans proving sandbox/runtime do not import Gymnasium/NumPy
- source scan proving `gym_environment.py` does not import SB3/PyTorch or register/render
- `git diff --check`

Expected: all commands exit zero and all scans report no violations.

- [ ] **Step 5: Audit changed-file scope**

Compare against the feature base `537569412c348c2ea2b88f66f0c1e5ed46ba97a3`. Expected production changes are limited to `pyproject.toml`, `f1rl/learning/gym_environment.py`, `f1rl/learning/__init__.py`, and `f1rl/learning/README.md`; no sandbox/runtime production file is modified.

- [ ] **Step 6: Whole-branch review against spec**

Check every acceptance item in `docs/superpowers/specs/2026-09-28-gymnasium-environment-design.md`. If any Critical/Important gap is found, add a failing regression test first, make one focused fix, and rerun Steps 2â€“5 before proceeding.

- [ ] **Step 7: Commit final acceptance coverage if Task 4 added tests/fixes**

Commit: `test(learning): verify gymnasium integration readiness`

## Expected Branch Result

The feature branch adds a fixed-track Gymnasium adapter that exposes correct NumPy spaces, deterministic reset/step behavior, authoritative reward/progress timing, strict lifecycle/fault recovery, primitive diagnostics, and Gymnasium checker compliance without moving framework dependencies into lower layers.

After verification, finish the branch through the existing finishing workflow. The already agreed integration sequence is:

```text
feat/gymnasium-environment
  -> merge locally into feat/sanbox-physics-core
  -> verify merged core
  -> merge verified core into main
  -> verify main
  -> create models/ppo-baseline from verified main
```

Do not create the PPO branch early and do not add SB3/PyTorch to the Gym feature branch.
