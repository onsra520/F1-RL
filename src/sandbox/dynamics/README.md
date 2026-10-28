# `f1rl.sandbox.dynamics`

## Purpose

Deterministic simplified 3D vehicle physics for both supported-surface driving and airborne motion.

## Planned files

- `ground_dynamics.py` — surface-tangent acceleration, braking/drag, steering, slope response, and grounded orientation.
- `air_dynamics.py` — gravity, simple air drag, and airborne angular-motion behavior.
- `vehicle_dynamics.py` — selects ground or air dynamics from the supplied surface contact and exposes the high-level physics API.
- `integrator.py` — advances 3D position, velocity, orientation, and ground/air transitions through simulation time.

## Boundary

This package consumes vehicle state, controls, geometry values, and optional surface contact only. It does not query track geometry, choose RL actions, calculate rewards, or render the vehicle.
