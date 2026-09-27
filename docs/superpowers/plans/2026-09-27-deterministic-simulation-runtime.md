# Deterministic Simulation Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a passive deterministic runtime that advances the existing sandbox world exactly one fixed timestep per `Simulation.step(control)`, exposes immutable snapshots, and can reset exactly to its initial world state.

**Architecture:** Keep all physics and interaction behavior in `sandbox.step_world(...)`. `f1rl.runtime.simulation` owns only fixed-step configuration, tick/time metadata, the current runtime pointer, initial-state reset semantics, and immutable snapshots. It never owns a wall-clock loop, Driver, API, thread, or async scheduler.

**Tech Stack:** Python >=3.11, standard-library `dataclasses`/`math`, existing `f1rl.sandbox` world APIs, pytest >=8,<10.

**Spec:** `docs/superpowers/specs/2026-09-27-deterministic-simulation-runtime-design.md`

## Global Constraints

- `Simulation.step(control)` advances exactly one sandbox timestep and calls `step_world(...)` exactly once.
- `fixed_dt` is finite and strictly positive; `max_surface_distance` is finite and non-negative.
- Runtime construction starts at tick `0` and elapsed time `0.0`.
- Elapsed time is recomputed as `tick * fixed_dt`, never accumulated by repeated addition.
- `reset()` restores the exact constructor-supplied `WorldState`, including vehicle, pickup state, interaction membership, and score.
- A failed `step()` propagates the original exception and leaves runtime state/snapshot unchanged.
- Runtime snapshots are immutable value objects and preserve sandbox world/query/events without reinterpretation.
- Runtime must not import `f1rl.learning`, `f1rl.api`, drivers, FastAPI/WebSocket, browser/rendering code, or wall-clock/thread/async scheduling primitives.
- No public `start()`, `stop()`, pause/resume, hidden catch-up, or implicit substepping belongs in this milestone.
- Existing sandbox behavior and tests must remain unchanged.

## Review Focus

- Python `bool` is an `int` subclass: tick/config numeric validation must reject booleans where integer/float metadata is expected.
- Reset must restore the exact initial immutable world reference rather than reconstructing a merely similar world with stale pickup/interaction state.
- `step_world()` exceptions must not replace either `_state` or `_snapshot`, even when the failure happens after previous successful ticks.
- Tick-derived time must remain multiplication-based at values such as `fixed_dt=0.1`, where repeated addition would visibly drift after ten steps.
- Interleaving calls from different simulations must not introduce shared mutable state or cadence dependence.

---

### Task 1: Runtime value objects and validation

**Files:**
- Create: `f1rl/runtime/__init__.py`
- Create: `f1rl/runtime/simulation.py`
- Create: `tests/runtime/test_simulation_values.py`

**Interfaces:**
- Consumes: existing `WorldState`, `TrackQuery`, and `WorldEvent` from `f1rl.sandbox.world`.
- Produces: immutable `SimulationConfig(fixed_dt: float, max_surface_distance: float)`, `SimulationState(tick: int, elapsed_time: float, world: WorldState)`, and `SimulationSnapshot(tick: int, elapsed_time: float, world: WorldState, query: TrackQuery, events: tuple[WorldEvent, ...])`.

- [ ] **Step 1: Write failing validation/value tests**

Add tests proving:
- `SimulationConfig(0.1, 0.5)` is accepted and immutable.
- `fixed_dt` rejects `0.0`, negatives, `nan`, `inf`, and booleans.
- `max_surface_distance` accepts `0.0` and rejects negatives, `nan`, `inf`, and booleans.
- `SimulationState` accepts tick `0`/time `0.0`, rejects negative/non-integer/bool ticks, and rejects negative/non-finite/bool elapsed time.
- `SimulationSnapshot` is frozen and preserves its supplied world/query/events exactly.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/runtime/test_simulation_values.py -q`

Expected: collection/import failure because runtime value APIs do not exist.

- [ ] **Step 3: Implement the immutable value types**

Implement the three dataclasses in `f1rl/runtime/simulation.py` with only the validation required by the spec. Re-export them from `f1rl/runtime/__init__.py`.

- [ ] **Step 4: Run GREEN + regression suite**

Run:
- `python -m pytest tests/runtime/test_simulation_values.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 5: Commit**

Commit message: `feat(runtime): add deterministic simulation value types`

---

### Task 2: Simulation construction, snapshot, and exact reset

**Files:**
- Modify: `f1rl/runtime/simulation.py`
- Modify: `f1rl/runtime/__init__.py`
- Create: `tests/runtime/test_simulation_lifecycle.py`

**Interfaces:**
- Consumes: Task 1 value objects, `TrackWorld.query_point(position, max_surface_distance, state)`, `WorldState`, and `VehicleParameters`.
- Produces: `Simulation(track: TrackWorld, initial_state: WorldState, parameters: VehicleParameters, config: SimulationConfig)`, read-only `snapshot: SimulationSnapshot`, and `reset() -> SimulationSnapshot`.

- [ ] **Step 1: Write failing construction/reset tests**

Add tests proving:
- constructor snapshot has tick `0`, elapsed `0.0`, the exact `initial_state` object, a query for the initial vehicle position/track state, and `events == ()`;
- `simulation.snapshot` returns the current immutable snapshot and has no setter;
- calling `reset()` at tick 0 returns the same exact initial world reference with a freshly correct query and zero events;
- constructor/reset do not mutate `TrackWorld`, `WorldState`, or `VehicleParameters`.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/runtime/test_simulation_lifecycle.py -q`

Expected: failure because `Simulation` does not exist.

- [ ] **Step 3: Implement constructor/snapshot/reset**

Store immutable references to track, parameters, config, and the exact constructor `initial_state`. Keep current runtime metadata private. Build initial/reset snapshots through `TrackWorld.query_point(initial_state.vehicle.position, config.max_surface_distance, initial_state.track)` and never recreate domain entities.

- [ ] **Step 4: Run GREEN + regression suite**

Run:
- `python -m pytest tests/runtime/test_simulation_lifecycle.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 5: Commit**

Commit message: `feat(runtime): add simulation lifecycle and reset`

---

### Task 3: One-step execution, tick-derived time, and transactionality

**Files:**
- Modify: `f1rl/runtime/simulation.py`
- Create: `tests/runtime/test_simulation_step.py`

**Interfaces:**
- Consumes: Task 2 `Simulation`, existing `step_world(track, state, control, parameters, dt, max_surface_distance) -> WorldStepResult`, and `VehicleControl`.
- Produces: `Simulation.step(control: VehicleControl) -> SimulationSnapshot`.

- [ ] **Step 1: Write failing one-step tests**

Add real-domain tests proving:
- one call to `step()` advances tick from `0` to `1` and elapsed time to exactly `fixed_dt`;
- two calls produce ticks `1`, `2` and times `fixed_dt`, `2 * fixed_dt`;
- snapshot `world`, `query`, and `events` equal the `WorldStepResult` data produced by the sandbox behavior for that step;
- entering/collecting world interactions are carried through unchanged rather than recomputed by runtime.

- [ ] **Step 2: Add focused call-count/time-drift tests**

Using `monkeypatch` only at the `f1rl.runtime.simulation.step_world` boundary, prove:
- one `Simulation.step()` invokes `step_world` exactly once with the configured `fixed_dt` and `max_surface_distance`;
- ten successful ticks at `fixed_dt=0.1` report `elapsed_time == 10 * 0.1` and not a separately accumulated counter;
- there is no hidden second/catch-up step.

- [ ] **Step 3: Add transactional failure test**

Monkeypatch runtime's `step_world` to raise a sentinel exception after the simulation has already completed at least one successful tick. Assert the same exception propagates and both current snapshot equality and object identity remain unchanged after failure.

- [ ] **Step 4: Run RED**

Run: `python -m pytest tests/runtime/test_simulation_step.py -q`

Expected: failure because `Simulation.step()` does not exist.

- [ ] **Step 5: Implement `Simulation.step()`**

Read the current state first, call `step_world(...)` once, and do not assign new state/snapshot until it returns successfully. On success compute `next_tick = current.tick + 1` and `elapsed_time = next_tick * config.fixed_dt`, then atomically replace the private state/snapshot pointers and return the new snapshot.

- [ ] **Step 6: Run GREEN + regression suite**

Run:
- `python -m pytest tests/runtime/test_simulation_step.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 7: Commit**

Commit message: `feat(runtime): advance one deterministic simulation tick`

---

### Task 4: Runtime acceptance, deterministic replay, and boundary verification

**Files:**
- Create: `tests/integration/test_simulation_runtime.py`
- Modify: `f1rl/runtime/README.md`

**Interfaces:**
- Consumes: complete `Simulation` API from Tasks 1-3 plus existing sandbox track/world primitives.
- Produces: verified deterministic runtime baseline ready for Driver, Gym, replay, evaluation, or real-time session layers.

- [ ] **Step 1: Add exact-reset acceptance flow**

Build a world containing at least one score pickup and checkpoint/finish interaction. Step until world state changes, then call `reset()` and prove vehicle state, `TrackState`, `InteractionState`, score, query, tick/time, and zero-event snapshot all return to the exact initial values/reference required by the spec.

- [ ] **Step 2: Add deterministic replay test**

Create two independent `Simulation` instances from identical immutable inputs, apply the same ordered `VehicleControl` sequence, and assert the complete ordered `SimulationSnapshot` sequences are exactly equal.

- [ ] **Step 3: Add cadence/interleaving independence test**

Apply the same controls to two simulations in different caller interleavings (for example A-A-B-B versus paired A-B calls) and prove each simulation's final snapshot depends only on its own ordered controls, not call cadence or another runtime instance.

- [ ] **Step 4: Add public-surface/boundary checks**

Prove the runtime exposes `step`, `reset`, and `snapshot` but no public `start`/`stop` behavior. Scan runtime Python imports/source for forbidden `f1rl.learning`, `f1rl.api`, driver, FastAPI/WebSocket, `time.sleep`, threading, and asyncio scheduling dependencies.

- [ ] **Step 5: Update runtime README**

Document that runtime is a passive fixed-step primitive: external callers decide when to invoke `step(control)`, while future Driver/Gym/API/session layers sit above it.

- [ ] **Step 6: Run final verification**

Run:
- `python -m pytest -q`
- `python -m compileall -q f1rl`
- dependency/source boundary scan from Step 4
- `git diff --check`

Expected: zero test failures, successful compilation, no forbidden runtime dependencies, and clean whitespace checks.

- [ ] **Step 7: Commit**

Commit message: `test(runtime): verify deterministic simulation lifecycle`
