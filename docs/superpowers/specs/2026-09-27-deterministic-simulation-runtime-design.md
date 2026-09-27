# Deterministic Simulation Runtime Design

## Goal

Build a small deterministic runtime primitive that advances the existing sandbox world exactly one fixed timestep whenever the caller invokes `step(control)`.

The runtime is intentionally passive: it never owns a wall-clock loop, sleeps, spawns threads, reads keyboard input, calls a policy, or starts itself. Human gameplay, PPO training, replay, evaluation, FastAPI, and future session code can all drive the same simulation by deciding when and how often to call `step()`.

## Architectural choice

V1 uses a **manual fixed-step simulation**.

```text
external caller
    |
    | VehicleControl
    v
Simulation.step(...)
    |
    v
sandbox.step_world(...)
    |
    v
SimulationSnapshot
```

This keeps time progression explicit and deterministic. A real-time loop can later call `step()` at approximately 60 Hz, while training can call it as fast as the CPU permits.

Rejected alternatives for this milestone:

- Runtime-owned `while running` loop: mixes deterministic simulation with wall-clock scheduling too early.
- Runtime-owned Driver: couples the simulation primitive to human/policy control sources before the Driver layer exists.
- Async/threaded stepping: unnecessary for the domain primitive and harder to test deterministically.

## Scope

### In scope

- Fixed timestep configuration.
- Runtime tick and elapsed simulation time.
- One-step advancement through existing `sandbox.step_world(...)`.
- Immutable snapshots after each accepted step.
- Exact reset to the constructor's initial world state.
- Deterministic repeated runs from the same initial state and control sequence.
- Transactional state updates: a failed step leaves the runtime unchanged.

### Out of scope

- `start()`, `stop()`, pause/resume, sleep, timers, threads, asyncio, or wall-clock synchronization.
- Driver ownership or driver switching.
- Human keyboard/gamepad input.
- PPO/Gymnasium wrappers.
- FastAPI/WebSocket delivery.
- Session/lap lifecycle, race completion, respawn, or results.
- Snapshot serialization formats.

## Package layout

V1 stays deliberately small:

```text
f1rl/runtime/__init__.py
f1rl/runtime/simulation.py
tests/runtime/test_simulation.py
```

`simulation.py` owns runtime-only configuration, state, snapshots, and the passive `Simulation` orchestrator. It does not duplicate sandbox physics or interaction rules.

## Configuration

```python
SimulationConfig(
    fixed_dt: float,
    max_surface_distance: float,
)
```

Validation:

- `fixed_dt` must be finite and strictly positive.
- `max_surface_distance` must be finite and non-negative.

Configuration is immutable for one `Simulation` instance. Changing timestep mid-run is outside V1 because it complicates deterministic replay and makes time-series comparisons ambiguous.

## Runtime state

```python
SimulationState(
    tick: int,
    elapsed_time: float,
    world: WorldState,
)
```

Rules:

- `tick` is a non-negative integer and rejects booleans.
- `elapsed_time` is finite and non-negative.
- Runtime construction starts at `tick == 0` and `elapsed_time == 0.0`.
- After every successful step, `tick` increments by exactly one.
- `elapsed_time` is recomputed as `tick * fixed_dt`, not accumulated with `elapsed_time += fixed_dt`.

The recomputation rule avoids cumulative floating-point drift caused by repeated addition and keeps replay results dependent only on tick and configuration.

## Snapshot contract

```python
SimulationSnapshot(
    tick: int,
    elapsed_time: float,
    world: WorldState,
    query: TrackQuery,
    events: tuple[WorldEvent, ...],
)
```

Snapshots are immutable value objects.

The runtime exposes its latest snapshot through:

```python
simulation.snapshot -> SimulationSnapshot
```

The initial snapshot and every reset snapshot contain:

- `tick == 0`
- `elapsed_time == 0.0`
- the exact initial `WorldState`
- a fresh `TrackWorld.query_point(...)` for that initial position/state
- `events == ()`

A step snapshot contains the accepted `WorldState`, final `TrackQuery`, and events returned by `step_world()` for that step.

The runtime does not reinterpret, reorder, or synthesize sandbox events.

## Simulation object

```python
Simulation(
    track: TrackWorld,
    initial_state: WorldState,
    parameters: VehicleParameters,
    config: SimulationConfig,
)
```

The runtime retains immutable references to track, vehicle parameters, config, and the exact initial world state. It owns only the current runtime state/snapshot pointer.

Public operations:

```python
simulation.step(control: VehicleControl) -> SimulationSnapshot
simulation.reset() -> SimulationSnapshot
simulation.snapshot -> SimulationSnapshot
```

No public `start()` or `stop()` exists in this milestone.

## Step semantics

`step(control)` performs exactly one simulation tick:

```text
current runtime state
      |
      v
step_world(
  track,
  current.world,
  control,
  parameters,
  fixed_dt,
  max_surface_distance,
)
      |
      v
WorldStepResult
      |
      v
tick + 1
elapsed = tick * fixed_dt
      |
      v
new SimulationState + SimulationSnapshot
```

Authoritative order:

1. Read the current `SimulationState` without mutating it.
2. Call existing `sandbox.step_world(...)` exactly once using `config.fixed_dt` and `config.max_surface_distance`.
3. If `step_world()` raises, propagate the exception and leave runtime state/snapshot unchanged.
4. On success, compute `next_tick = current.tick + 1`.
5. Compute `elapsed_time = next_tick * config.fixed_dt`.
6. Replace the current runtime state/snapshot with values derived from the successful `WorldStepResult`.
7. Return that immutable snapshot.

The runtime never performs additional physics substeps implicitly.

## Reset semantics

`reset()` is exact, not incremental.

It restores:

```text
tick = 0
elapsed_time = 0.0
world = exact initial WorldState
```

This means reset restores all state already contained in `WorldState`, including:

- vehicle pose/velocity,
- track collectible state,
- checkpoint/finish interaction membership,
- game score.

After restoring the initial world, reset queries the track at the initial vehicle position using the initial `TrackState` and returns a zero-event snapshot.

No entity definition or `TrackWorld` object is recreated or mutated during reset.

## Determinism

For identical:

- `TrackWorld`,
- `WorldState`,
- `VehicleParameters`,
- `SimulationConfig`,
- ordered sequence of `VehicleControl` values,

repeated simulations must produce exactly equal ordered `SimulationSnapshot` sequences.

Runtime code must not read wall-clock time, random values, environment variables, networking state, or rendering state.

## Transactional failure behavior

A runtime call must not partially advance state.

If validation or sandbox stepping fails during `step(control)`:

```text
simulation.snapshot before call == simulation.snapshot after failure
```

Examples include invalid downstream state or a sandbox exception. The exception is propagated to the caller; V1 does not wrap it in a runtime-specific exception type.

## Dependency boundary

```text
sandbox
  ^
  |
runtime.simulation
  ^
  |
future Driver / Gym / API / Session
```

`f1rl.runtime` may depend on sandbox domain types and world stepping.

It must not import:

- `f1rl.learning`
- `f1rl.api`
- browser/rendering code
- React or Three.js
- WebSocket/FastAPI
- wall-clock scheduling libraries for simulation progression

Future Driver and API layers may depend on runtime; runtime must not depend on them.

## Verification criteria

The subsystem is ready for Driver/Gym/session integration when tests prove:

1. Construction produces a tick-0, time-0, zero-event snapshot matching the initial world and its query.
2. One `step()` advances exactly one sandbox tick and returns the sandbox world/query/events unchanged except for runtime metadata.
3. Two steps produce ticks `1`, `2` and times `fixed_dt`, `2 * fixed_dt`.
4. Time is derived from tick multiplication rather than cumulative addition.
5. `reset()` after arbitrary world changes restores the exact initial world, pickups, interaction state, and score.
6. Reset returns a query matching the restored initial position/state and no events.
7. Two simulations with the same control sequence produce exactly equal snapshots.
8. Different caller cadence does not affect simulation results when the same controls are applied in the same order.
9. `step()` calls `step_world()` exactly once per invocation; there is no hidden catch-up or substepping.
10. A failed step leaves runtime state and latest snapshot unchanged.
11. Invalid config/state metadata is rejected deterministically.
12. Runtime contains no wall-clock/start-stop/thread/async behavior.
13. Existing sandbox tests remain green and dependency boundaries stay intact.

## Next milestone boundary

After this primitive is stable, the next layer can introduce control ownership without changing sandbox physics:

```text
Driver / Gym adapter
       |
       v
Simulation.step(control)
       |
       v
SimulationSnapshot
```

A later real-time/session layer can implement `Start`, `Pause`, or fixed-rate scheduling externally by deciding when to call `Simulation.step()`.