# `f1rl.sandbox.perception`

## Purpose

Deterministic raw physical sensing from the vehicle's local frame.

## Current baseline

- `rangefinder.py` defines immutable rangefinder specs/readings and obstacle-only sensing.
- `sensor_suite.py` evaluates multiple rangefinders in exact declared configuration order.
- Rays originate at `vehicle.position` and local directions rotate with vehicle orientation.
- Local axes follow `+x = right`, `+y = up`, `+z = forward`.
- Rangefinders use analytic ray-vs-oriented-box intersection and report nearest obstacle distance.
- Equal-distance obstacle hits resolve by lexical obstacle ID.
- A miss reports the configured maximum distance with `hit_id = None`.

## Boundary

Perception returns raw physical measurements only. It does not detect boost, slow, checkpoint,
finish, or pickup trigger volumes, and it does not normalize values or calculate rewards.
Normalization and neural-network observation encoding belong in `f1rl.learning.observation`.
Runtime scheduling, policy inference, API delivery, and rendering remain outside the sandbox.
