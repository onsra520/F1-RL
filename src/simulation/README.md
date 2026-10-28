# `src.simulation`

## Purpose

The source of truth for the simulated racing world. This layer owns deterministic 3D world state and state transitions, independent of RL and presentation concerns.

## Core files

- `world.py` — immutable aggregate state and authoritative fixed-step orchestration.
- `surface.py` — presentation-neutral surface/contact values shared by track and physics.
- `vehicle.py` — 3D vehicle state, controls, dimensions, and physical limits.

## Subdirectories

- `geometry/` — renderer-independent `Vec3` and quaternion math values.
- `dynamics/` — grounded/airborne motion, deterministic integration, and generic motion modifiers.
- `track/` — finite road surfaces, spatial volumes, aggregate world queries, and collectible state.
- `entities/` — solid obstacles, trigger zones, checkpoints, finish lines, and collectibles.
- `interactions/` — zone effects, obstacle rollback, and edge-triggered world events.
- `rules/` — game/demo scoring plus future collision/timing rules.
- `perception/` — local sensing of the world.

## Boundary

No PPO, Gymnasium, FastAPI, React, Three.js, or third-party physics-engine dependencies belong here.
