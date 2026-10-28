# `src.simulation.geometry`

## Purpose

Small renderer-independent 3D math values used by the sandbox physics core.

## Planned files

- `vector.py` — immutable `Vec3` arithmetic, dot/cross products, magnitude, normalization, and projection/rejection.
- `rotation.py` — immutable normalized quaternion orientation, composition, axis-angle construction, and vector rotation.

## Boundary

Keep this package minimal and deterministic. Do not import NumPy, Three.js, a game engine, RL code, or application-layer concerns.
