# `f1rl.sandbox`

## Purpose

The source of truth for the simulated racing world. This layer owns deterministic 3D world state and state transitions, independent of RL and presentation concerns.

## Planned files

- `world.py` — aggregate simulation state, surface queries, and world stepping.
- `track.py` — 3D track geometry, drivable surfaces, ramps, and navigable boundaries.
- `surface.py` — presentation-neutral surface/contact values shared by track and physics.
- `vehicle.py` — 3D vehicle state, controls, dimensions, and physical limits.

## Subdirectories

- `geometry/` — renderer-independent `Vec3` and quaternion math values.
- `dynamics/` — grounded/airborne vehicle motion and deterministic integration.
- `perception/` — local sensing of the world.
- `entities/` — obstacles, zones, checkpoints, and finish-line objects.
- `rules/` — collision, timing, and race scoring rules.

## Boundary

No PPO, Gymnasium, FastAPI, React, Three.js, or third-party physics-engine dependencies belong here.
