# `f1rl.sandbox.dynamics`

## Purpose

Vehicle motion and state-transition mathematics.

## Planned files

- `vehicle_dynamics.py` — converts control input and vehicle state into motion derivatives such as acceleration and yaw rate.
- `integrator.py` — advances vehicle state through simulation time using the selected numerical integration method.

## Boundary

This package consumes physical state and control values only. It does not choose actions or calculate rewards.
