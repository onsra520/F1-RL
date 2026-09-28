# Race Progress / Episode Semantics V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic single-lap route progress and episode status semantics that turn sandbox checkpoint/finish events into authoritative navigation targets and Gym-compatible terminated/truncated signals.

**Architecture:** Add a pure `f1rl.learning.episode` package. `route.py` owns immutable authored route definition and target selection, `progress.py` owns immutable ordered checkpoint/finish transitions, and `status.py` owns max-step terminated/truncated semantics. Production episode code depends only on sandbox domain/event types plus stdlib; runtime is allowed only in integration tests.

**Tech Stack:** Python 3.12, frozen/slotted dataclasses, pytest, existing sandbox/runtime primitives.

**Spec:** `docs/superpowers/specs/2026-09-28-race-progress-episode-semantics-design.md`

## Global Constraints

- V1 is single-lap only.
- Production `f1rl.learning.episode` may depend only on Python stdlib and sandbox domain/event types.
- Do not import runtime, API, drivers, observation, action, rewards, Gymnasium, NumPy, PyTorch, SB3/PPO, FastAPI/WebSocket, or renderer/web code from production episode modules.
- Sandbox must not import `f1rl.learning`.
- Route progress is immutable, deterministic, and history-free.
- Checkpoint progression is strictly ordered and may advance at most one checkpoint per accepted transition.
- Early finish, stale/re-entered checkpoints, future checkpoints, wrong IDs/orders, obstacle hits, and score pickups must not advance route progress.
- Final expected checkpoint plus configured finish may complete in one transition.
- `RaceProgressResult.target_position` must always equal `route.target_for(result.state)`.
- Valid finish has precedence over timeout at the same step.
- This milestone does not modify reward semantics or add checkpoint/finish bonuses.

## Review Focus

- Route/state incompatibility must fail fast instead of indexing past the route or treating an impossible finished state as valid — pinned in Task 2.
- Multiple checkpoint events in one transition must never skip more than one route stage, even when events overlap or are duplicated — pinned in Task 2.
- Final checkpoint plus finish must work in the same transition without relying on raw event tuple ordering — pinned in Task 2.
- A valid finish on `step_count == max_steps` must return terminated-only, while unfinished timeout returns truncated-only — pinned in Task 3.
- Public exports and production dependency boundaries must not regress existing observation/action/reward exports or create sandbox reverse dependencies — pinned in Task 4.

## File Structure

- Create `f1rl/learning/episode/route.py` — route value objects, canonicalization, `from_track`, and authoritative target selection.
- Create `f1rl/learning/episode/progress.py` — progress state/result and pure event-driven advancement.
- Create `f1rl/learning/episode/status.py` — episode config/status and terminated/truncated precedence.
- Create `f1rl/learning/episode/__init__.py` — package public exports.
- Modify `f1rl/learning/__init__.py` — re-export episode APIs while preserving existing observation/action/reward exports.
- Modify `f1rl/learning/README.md` — document episode authority and the next Gymnasium boundary.
- Create `tests/learning/episode/test_route.py`.
- Create `tests/learning/episode/test_progress.py`.
- Create `tests/learning/episode/test_status.py`.
- Create `tests/integration/test_episode_integration.py`.

---

### Task 1: Immutable route definition and authoritative target selection

**Files:**
- Create: `f1rl/learning/episode/route.py`
- Create: `f1rl/learning/episode/__init__.py`
- Create: `tests/learning/episode/test_route.py`

**Interfaces:**
- Consumes: sandbox `TrackWorld`, `Vec3`, authored `Checkpoint` and `FinishLine` data.
- Produces: `RaceCheckpoint`, `RaceFinish`, `RaceRoute`, `RaceRoute.from_track(track)`, and `RaceRoute.target_for(state)`.

- [ ] **Step 1: Write failing route value-object validation tests**

Cover non-empty IDs, non-negative integer checkpoint order with booleans rejected, `Vec3` target requirement, and frozen immutability for `RaceCheckpoint` and `RaceFinish`.

- [ ] **Step 2: Write failing `RaceRoute` validation/canonicalization tests**

Assert `checkpoints` must be a tuple and `finish` a `RaceFinish`; checkpoints are canonicalized by numeric `order`; duplicate orders, non-contiguous orders, duplicate checkpoint IDs, and finish/checkpoint ID collisions fail. Prove lexical IDs do not control route order.

- [ ] **Step 3: Write failing `from_track` tests**

Construct tracks with checkpoint IDs intentionally out of lexical/numeric alignment. Assert route order follows `Checkpoint.order`, each target equals `volume.center`, zero checkpoints are valid, and exactly one finish line is required.

- [ ] **Step 4: Write failing target-selection tests**

Use compatible progress states to assert target sequence is checkpoint 0, later expected checkpoint, finish after all checkpoints, and `None` when finished. Zero-checkpoint route must target finish immediately.

- [ ] **Step 5: Run RED**

Run: `python -m pytest tests/learning/episode/test_route.py -q`

Expected: import/collection failure because `f1rl.learning.episode` does not exist.

- [ ] **Step 6: Implement route contracts**

Implement exact spec signatures in `route.py`. Use one private route/state compatibility helper that later progress code can reuse without importing observation/reward/runtime. Canonicalize checkpoints by numeric `order`; do not mutate `TrackWorld`.

- [ ] **Step 7: Export Task 1 route APIs**

Create `f1rl/learning/episode/__init__.py` exporting `RaceCheckpoint`, `RaceFinish`, and `RaceRoute` only for now.

- [ ] **Step 8: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/episode/test_route.py -q`
- `python -m pytest tests/learning -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 9: Commit**

Commit message: `feat(learning): define single-lap race routes`

---

### Task 2: Ordered race-progress state machine

**Files:**
- Create: `f1rl/learning/episode/progress.py`
- Modify: `f1rl/learning/episode/__init__.py`
- Create: `tests/learning/episode/test_progress.py`

**Interfaces:**
- Consumes: Task 1 `RaceRoute`; sandbox `WorldEvent`, `CheckpointEntered`, and `FinishLineEntered`.
- Produces: `RaceProgressState`, `RaceProgressResult`, and `advance_race_progress(route, state, events)`.

- [ ] **Step 1: Write failing state/result validation tests**

Assert initial `RaceProgressState(0, False)` is valid; reject boolean/non-integer/negative indices and non-bool `finished`. Assert result fields are immutable and enforce the documented types.

- [ ] **Step 2: Write failing route/state compatibility tests**

Assert `next_checkpoint_index > len(route.checkpoints)` fails and `finished=True` requires index exactly equal to checkpoint count. Prove unfinished state at checkpoint count is valid and targets finish.

- [ ] **Step 3: Write failing expected-checkpoint advancement tests**

For a multi-checkpoint route, matching both expected ID and order advances exactly one index and reports `completed_checkpoint_id`; post-transition target must equal `route.target_for(result.state)`.

- [ ] **Step 4: Write failing invalid-event/no-op tests**

Prove stale/re-entered checkpoint, future checkpoint, matching-ID/wrong-order, matching-order/wrong-ID, obstacle hit, pickup, early finish, and wrong finish ID leave state unchanged with no completion diagnostics.

- [ ] **Step 5: Write failing max-one-checkpoint tests**

Supply tuples containing expected checkpoint plus duplicated/future checkpoint events. Assert a call advances at most one route stage regardless of tuple order or duplication.

- [ ] **Step 6: Write failing finish tests**

Assert finish works only after all checkpoints are effectively complete. Cover zero-checkpoint direct finish, early finish ignore, finished-state subsequent no-op, and final expected checkpoint plus configured finish completing in the same transition.

- [ ] **Step 7: Write failing event-order-independence test**

For final checkpoint plus finish, test both `(CheckpointEntered, FinishLineEntered)` and reversed order and assert identical `RaceProgressResult`.

- [ ] **Step 8: Run RED**

Run: `python -m pytest tests/learning/episode/test_progress.py -q`

Expected: import failure because progress APIs do not exist.

- [ ] **Step 9: Implement pure progress transition**

Implement `RaceProgressState`, `RaceProgressResult`, and `advance_race_progress(...)`. Evaluate the currently expected checkpoint against the whole event tuple, accept at most one, then evaluate finish eligibility against the whole event tuple. Ignore obstacle/pickup events. Never mutate route/state/events.

- [ ] **Step 10: Export progress APIs**

Update package exports with `RaceProgressState`, `RaceProgressResult`, and `advance_race_progress`.

- [ ] **Step 11: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/episode/test_progress.py -q`
- `python -m pytest tests/learning/episode -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 12: Commit**

Commit message: `feat(learning): track ordered race progress`

---

### Task 3: Deterministic terminated/truncated semantics

**Files:**
- Create: `f1rl/learning/episode/status.py`
- Modify: `f1rl/learning/episode/__init__.py`
- Create: `tests/learning/episode/test_status.py`

**Interfaces:**
- Consumes: Task 2 `RaceProgressState`.
- Produces: `EpisodeConfig`, `EpisodeStatus`, and `episode_status(progress, step_count, config)`.

- [ ] **Step 1: Write failing `EpisodeConfig` validation tests**

Accept positive integer `max_steps`; reject booleans, zero, negatives, floats, strings, and other objects. Assert immutability.

- [ ] **Step 2: Write failing step-count/input validation tests**

Reject non-`RaceProgressState`, non-`EpisodeConfig`, boolean/non-integer/negative `step_count`.

- [ ] **Step 3: Write failing status truth-table tests**

Assert unfinished before limit => `(False, False)`; unfinished at/after limit => `(False, True)`; finished at any step including exactly `max_steps` => `(True, False)`.

- [ ] **Step 4: Write failing non-mutation/determinism test**

Repeated equal calls return equal `EpisodeStatus` and leave progress/config unchanged.

- [ ] **Step 5: Run RED**

Run: `python -m pytest tests/learning/episode/test_status.py -q`

Expected: import failure because status APIs do not exist.

- [ ] **Step 6: Implement status semantics**

Implement exact precedence from the spec: finished always terminated-only; otherwise `step_count >= max_steps` is truncated-only; otherwise neither. Do not mutate progress or own reset behavior.

- [ ] **Step 7: Export status APIs**

Update package exports with `EpisodeConfig`, `EpisodeStatus`, and `episode_status`.

- [ ] **Step 8: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/episode/test_status.py -q`
- `python -m pytest tests/learning/episode -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 9: Commit**

Commit message: `feat(learning): define episode completion status`

---

### Task 4: Real-event integration, public exports, docs, and dependency acceptance

**Files:**
- Create: `tests/integration/test_episode_integration.py`
- Modify: `f1rl/learning/__init__.py`
- Modify: `f1rl/learning/README.md`

**Interfaces:**
- Consumes: complete Tasks 1–3 APIs plus existing sandbox/runtime primitives in tests only.
- Produces: verified public episode surface ready for Gymnasium composition.

- [ ] **Step 1: Add real route-construction integration test**

Build a `TrackWorld` whose lexical checkpoint IDs differ from numeric order. Assert `RaceRoute.from_track` yields numeric route order and exact authored `volume.center` targets.

- [ ] **Step 2: Add real ordered-world-event progression test**

Use existing world stepping or deterministic event-producing fixtures to obtain checkpoint entry events. Prove ordered progression updates target correctly and stale/future re-entry events cannot advance progress.

- [ ] **Step 3: Add final-checkpoint-plus-finish integration test**

Use overlapping final checkpoint/finish authored volumes so one accepted world transition emits both events. Assert progress completes one checkpoint, finishes immediately, and returns `target_position=None`.

- [ ] **Step 4: Add zero-checkpoint and episode-status integration tests**

Prove zero-checkpoint route finishes directly from real finish entry. Assert finished-at-`max_steps` => terminated-only and unfinished timeout => truncated-only.

- [ ] **Step 5: Add deterministic/non-mutating acceptance test**

Create independent equal routes/states/events/configs and assert complete result equality with source track/world/progress values unchanged.

- [ ] **Step 6: Add public-export acceptance test**

Import all nine episode symbols from both `f1rl.learning.episode` and `f1rl.learning`; assert identity agreement and prove existing observation/action/reward exports remain available.

- [ ] **Step 7: Add source/dependency boundary tests**

Scan production `f1rl/learning/episode/*.py` imports and reject runtime/API/drivers/observation/action/rewards/Gymnasium/NumPy/PyTorch/SB3/FastAPI/WebSocket/render dependencies. Scan sandbox production modules and prove none import `f1rl.learning`.

- [ ] **Step 8: Update learning documentation**

Document the authoritative route target, ordered single-lap progress, valid finish/timeout semantics, reward target timing invariant, and that Gymnasium is now the next composition layer. Do not claim multi-lap support or checkpoint reward bonuses.

- [ ] **Step 9: Re-export complete episode API from `f1rl.learning`**

Preserve all existing action/observation/reward exports while adding `RaceCheckpoint`, `RaceFinish`, `RaceRoute`, `RaceProgressState`, `RaceProgressResult`, `advance_race_progress`, `EpisodeConfig`, `EpisodeStatus`, and `episode_status`.

- [ ] **Step 10: Run final verification**

Run:
- `python -m pytest -q`
- `python -m compileall -q f1rl`
- episode dependency/source scans from Step 7
- sandbox reverse-dependency scan
- `git diff --check`

Expected: zero test failures, successful compilation, no forbidden dependencies, and clean whitespace checks.

- [ ] **Step 11: Commit**

Commit message: `test(learning): verify race episode semantics`

## Expected Branch Result

The feature branch contains only the approved spec/plan plus `f1rl.learning.episode`, its tests, and learning-facing exports/docs. No sandbox or runtime production file changes are required. The resulting layer supplies authoritative pre/post navigation targets, validated single-lap checkpoint/finish progress, and deterministic terminated/truncated semantics while preserving the existing fixed-target reward contract.
