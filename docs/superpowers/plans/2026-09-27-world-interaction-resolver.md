# World Interaction Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic fixed-step world orchestration that turns TrackWorld membership into grounded boost/slow effects, blocking obstacle behavior, checkpoint/finish entry events, score-pickup consumption, and game-score updates.

**Architecture:** Add generic `MotionModifiers` at the physics boundary; keep interaction edge state/events and scoring in focused sandbox modules; keep TrackWorld query-only; make `sandbox/world.py` the sole authoritative orchestrator for query → modifiers → physics → candidate → obstacle rollback → events/pickups/score. Existing physics callers remain compatible when modifiers are omitted.

**Tech Stack:** Python 3.11+, standard-library `dataclasses`/`math`, existing sandbox physics and TrackWorld APIs, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-world-interaction-resolver-design.md`

## Global Constraints

- World stepping is deterministic and immutable.
- Boost/slow effects apply only when the current query has a supporting `SurfaceContact`.
- Multiple boost bonuses sum; multiple slow multipliers multiply.
- Airborne motion ignores grounded-track modifiers.
- Checkpoint and finish events are entry-triggered, not emitted every frame.
- Obstacle V1 uses candidate reference-point blocking, rollback, and zero linear/angular velocity.
- Rejected candidate positions must not produce checkpoint/finish/pickup effects.
- `ScorePickupCollected` drives game/demo score only; no RL reward dependency is introduced.
- Returned `TrackQuery` always describes the accepted final vehicle position.
- Event category order is `ObstacleHit* -> CheckpointEntered* -> FinishLineEntered* -> ScorePickupCollected*` with lexical ID ordering inside each category.
- No interaction/rule/world module may import learning, API, web, React, Three.js, or a third-party physics engine.

## Review Focus

- Entering boost/slow during a step must not retroactively affect that same movement; the effect begins on the next step.
- A collision candidate overlapping both obstacle and pickup/checkpoint must rollback and emit no pickup/checkpoint event from the rejected position.
- Leaving and then re-entering checkpoint/finish must emit exactly one new entry event while remaining inside emits none.
- Multiple simultaneous obstacles/events must preserve exact category and lexical-ID ordering.
- Optional modifiers omitted by legacy physics callers must produce byte-for-byte equal domain results to pre-feature behavior.

---
### Task 1: Physics-facing motion modifiers

**Files:**
- Create: `f1rl/sandbox/dynamics/modifiers.py`
- Modify: `f1rl/sandbox/dynamics/ground_dynamics.py`
- Modify: `f1rl/sandbox/dynamics/air_dynamics.py`
- Modify: `f1rl/sandbox/dynamics/vehicle_dynamics.py`
- Create: `tests/sandbox/test_motion_modifiers.py`

**Interfaces:**
- Consumes: existing `VehicleState`, `VehicleControl`, `VehicleParameters`, `SurfaceContact`.
- Produces: immutable `MotionModifiers(longitudinal_acceleration_bonus=0.0, rolling_drag_multiplier=1.0)`; `compute_ground_motion(..., modifiers: MotionModifiers | None = None)`; `compute_air_motion(..., modifiers: MotionModifiers | None = None)`; and `compute_motion(..., modifiers: MotionModifiers | None = None)`. Grounded dynamics honors modifiers while airborne dynamics ignores them.

- [ ] Write failing tests for modifier validation, boost acceleration, slow-zone drag multiplication, combined modifiers, airborne no-op behavior, and omitted-modifier backward compatibility.
- [ ] Run `python -m pytest tests/sandbox/test_motion_modifiers.py -q`; verify failure because modifier APIs do not exist.
- [ ] Implement `MotionModifiers`; thread optional modifiers through `compute_motion()` and grounded/air dynamics without changing existing default behavior.
- [ ] Run the focused test and `python -m pytest -q`; require zero failures.
- [ ] Commit as `feat(world): add generic motion modifiers`.

### Task 2: Interaction state and entry events

**Files:**
- Create: `f1rl/sandbox/interactions/__init__.py`
- Create: `f1rl/sandbox/interactions/state.py`
- Create: `f1rl/sandbox/interactions/events.py`
- Create: `f1rl/sandbox/interactions/resolver.py`
- Create: `tests/sandbox/test_interactions.py`
**Interfaces:**
- Consumes: `TrackQuery` checkpoint/finish memberships.
- Produces: immutable `InteractionState(active_checkpoint_ids, active_finish_line_ids)`, `CheckpointEntered`, `FinishLineEntered`, `ObstacleHit`, and `resolve_entry_events(query, previous_state) -> tuple[InteractionState, tuple[CheckpointEntered, ...], tuple[FinishLineEntered, ...]]`.

- [ ] Write failing tests for state/event validation, first entry, staying inside with no repeat event, leave clearing active membership, re-entry emitting again, simultaneous IDs sorted lexically, and checkpoint `order` preservation.
- [ ] Run `python -m pytest tests/sandbox/test_interactions.py -q`; verify failure because interaction APIs do not exist.
- [ ] Implement immutable state/events and edge detection from current `TrackQuery` membership only; do not enforce checkpoint ordering or race completion.
- [ ] Run the focused test and full suite; require zero failures.
- [ ] Commit as `feat(world): add interaction entry events`.

### Task 3: Game score aggregation

**Files:**
- Create: `f1rl/sandbox/rules/__init__.py`
- Create: `f1rl/sandbox/rules/scoring.py`
- Create: `tests/sandbox/test_scoring.py`

**Interfaces:**
- Consumes: existing `ScorePickupCollected` events.
- Produces: immutable `ScoreState(points: int = 0)` and `apply_score_events(score, events) -> ScoreState`.

- [ ] Write failing tests for zero default, non-negative integer validation with bool rejection, one/multiple pickup additions, empty-event no-op equality, and input immutability.
- [ ] Run `python -m pytest tests/sandbox/test_scoring.py -q`; verify failure because scoring APIs do not exist.
- [ ] Implement score aggregation from pickup events only; do not import or calculate RL rewards.
- [ ] Run the focused test and full suite; require zero failures.
- [ ] Commit as `feat(world): add game score state`.
### Task 4: Continuous effects and obstacle blocking resolver

**Files:**
- Modify: `f1rl/sandbox/interactions/resolver.py`
- Create: `tests/sandbox/test_world_resolver.py`

**Interfaces:**
- Consumes: `TrackQuery`, Task 1 `MotionModifiers`, `VehicleState`.
- Produces: `resolve_motion_modifiers(query: TrackQuery) -> MotionModifiers` and `resolve_obstacle_block(previous: VehicleState, candidate: VehicleState, candidate_query: TrackQuery) -> tuple[VehicleState, tuple[ObstacleHit, ...]]`.

- [ ] Write failing tests proving no-contact queries return default modifiers, overlapping boosts sum, overlapping slows multiply, boost/slow order is irrelevant, and grounded-zone values are finite/deterministic.
- [ ] Write failing obstacle tests proving no obstacle accepts the candidate unchanged; one/multiple obstacle hits rollback position/orientation, preserve previous `grounded`, zero linear/angular velocity, and emit lexical-ID ordered `ObstacleHit` events.
- [ ] Run `python -m pytest tests/sandbox/test_world_resolver.py -q`; verify failure because resolver functions do not exist.
- [ ] Implement modifier aggregation from current query membership and the V1 reference-point blocking rule only; do not re-query TrackWorld inside the resolver.
- [ ] Run focused and full tests; require zero failures.
- [ ] Commit as `feat(world): resolve track effects and obstacle blocking`.

### Task 5: Authoritative world step

**Files:**
- Create: `f1rl/sandbox/world.py`
- Create: `tests/sandbox/test_world_step.py`

**Interfaces:**
- Consumes: `TrackWorld`, `VehicleControl`, `VehicleParameters`, Tasks 1-4, existing physics `compute_motion()` / `integrate()`, and pickup collection state.
- Produces: immutable `WorldState(vehicle, track, interactions, score)`, `WorldEvent = ObstacleHit | CheckpointEntered | FinishLineEntered | ScorePickupCollected`, `WorldStepResult(state, query, modifiers, events)`, and `step_world(track, state, control, parameters, dt, max_surface_distance) -> WorldStepResult`.

- [ ] Write failing tests for the exact 10-step orchestration order: current query drives modifiers, physics creates candidate, obstacle rejection re-queries rollback position, entry events use accepted query, pickup collection updates `TrackState`, and score updates from pickup events.
- [ ] Add tests proving a zone entered during movement affects only the next step, a rejected candidate cannot collect a pickup or enter checkpoint/finish, and returned query matches accepted position.
- [ ] Add tests proving event order is obstacle hits → checkpoint entries → finish entries → pickup collections, with lexical ID ordering inside each category.
- [ ] Run `python -m pytest tests/sandbox/test_world_step.py -q`; verify failure because world-step APIs do not exist.
- [ ] Implement `step_world()` exactly to the spec. After pickup collection, re-query the accepted vehicle position using the new `TrackState` so `WorldStepResult.query` is consistent with both accepted position and returned state.
- [ ] Run focused and full tests; require zero failures.
- [ ] Commit as `feat(world): orchestrate deterministic world steps`.

### Task 6: Interaction acceptance pass

**Files:**
- Create: `tests/integration/test_world_interactions.py`
- Modify: `f1rl/sandbox/rules/README.md`
- Create: `f1rl/sandbox/interactions/README.md`
- Modify: `f1rl/sandbox/README.md`

**Interfaces:**
- Consumes: full `step_world()` API over existing TrackWorld and physics.
- Produces: a verified sandbox world-step baseline ready for runtime/session integration.

- [ ] Add a boost test proving a vehicle already inside a grounded boost zone accelerates more than the same baseline step without boost, while entering a boost zone during movement delays the effect until the next tick.
- [ ] Add a slow-zone test proving rolling drag increases while inside the zone without mutating `VehicleParameters`; add an airborne case proving boost/slow have no motion effect without supporting contact.
- [ ] Add checkpoint/finish leave/re-entry tests proving edge-trigger semantics across multiple `step_world()` calls.
- [ ] Add a black-card flow proving one pickup event, one score increment, pickup removal from returned query/state, and no second increment on later steps.
- [ ] Add an obstacle scenario where the rejected candidate also overlaps checkpoint, finish, and pickup; prove rollback, zero velocity, stable `ObstacleHit` ordering, and absence of rejected-position trigger/pickup events.
- [ ] Run `python -m pytest -q`; require zero failures.
- [ ] Run repeated identical multi-step simulations and assert exact equality of all `WorldStepResult` values.
- [ ] Run `python -m compileall -q f1rl`, scan modified sandbox modules for forbidden learning/API/web/renderer imports, run `git diff --check`, and require a clean worktree after commit.
- [ ] Commit as `test(world): verify world interaction flow`.
