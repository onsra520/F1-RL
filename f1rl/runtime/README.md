# `f1rl.runtime`

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

Future Driver, Gym, replay, evaluation, API, and session layers sit above this primitive and decide when to call `Simulation.step(control)`.

## Planned next layer

- `session.py` — lifecycle, control ownership, race/demo timing policy, and results without duplicating sandbox physics.

Runtime may depend on sandbox domain types and stepping. Sandbox must not depend on runtime.
