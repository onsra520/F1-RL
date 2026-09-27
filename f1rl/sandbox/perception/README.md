# `f1rl.sandbox.perception`

## Purpose

Physical sensing of the sandbox from the vehicle's local frame.

## Planned files

- `rangefinder.py` — distance queries from vehicle-mounted directional range finders.
- `sensor_suite.py` — coordinates the complete set of sensors attached to a vehicle and returns raw readings.

## Boundary

Perception returns raw measurements. Normalization and neural-network observation encoding belong in `f1rl.learning`.
