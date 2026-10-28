# `src.core`

## Purpose

Application-level deterministic stepping above the sandbox world.

## Current baseline

`simulation.py` provides a passive fixed-step `Simulation` primitive:

- callers invoke `step(control)` to advance exactly one configured timestep;
- `tick` and simulation time advance deterministically;
- immutable snapshots expose the accepted world, query, and sandbox events;
- `reset()` restores the exact constructor-supplied initial `WorldState`;
- failed steps leave the latest runtime state/snapshot unchanged.

The simulation does not own a wall-clock loop, `start`/`stop`, sleeping, threads, asyncio, a Driver, Gymnasium, FastAPI, WebSocket, or rendering behavior.

## Opt-in WP7 research physics

`physics.PhysicsConfig` explicitly registers owner-sized body geometry and
independent suspension, with separate ballistic/automatic-release or grounded
yaw modes. Pass it as `Simulation(..., physics_config=...)`; nested contact,
rotation and sweep policies reach `step_world` without replacing the default
baseline. Existing body/ramp arguments cannot be mixed with this registration.
Snapshots expose the actual solver audits. Unsupported impacts/recovery and
uncertain contact remain errors before a snapshot is published.

`reset(initial_state=...)` is allowed only with explicit research physics;
it validates the supplied state and keeps the constructor baseline unchanged.
Ordinary `reset()` still restores the original world. Grounded chassis heave
and world-frame pitch/roll rates must agree with persistent suspension rates;
explicit takeoff has its own launch domain. These are software contracts, not
physical calibration or training admission.

Future Driver, Gym, replay, evaluation, API, and session layers sit above this primitive and decide when to call `Simulation.step(control)`.

## Planned next layer

- `session.py` — lifecycle, control ownership, race/demo timing policy, and results without duplicating sandbox physics.

Runtime may depend on sandbox domain types and stepping. Sandbox must not depend on runtime.
