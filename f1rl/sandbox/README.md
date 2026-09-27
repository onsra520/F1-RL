# `f1rl.sandbox`

## Purpose

The source of truth for the simulated racing world. This layer owns world state and deterministic state transitions, independent of RL and presentation concerns.

## Planned files

- `world.py` — aggregate simulation state and world stepping.
- `track.py` — track geometry and navigable boundaries.
- `vehicle.py` — vehicle state, dimensions, and physical limits.

## Subdirectories

- `dynamics/` — vehicle motion equations and numerical integration.
- `perception/` — local sensing of the world.
- `entities/` — obstacles, zones, checkpoints, and finish-line objects.
- `rules/` — collision, timing, and race scoring rules.

## Boundary

No PPO, Gymnasium, FastAPI, React, or Three.js dependencies belong here.
