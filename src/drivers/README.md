# `f1rl.drivers`

## Purpose

Interchangeable control sources for the same vehicle simulation.

## Planned files

- `base.py` — common driver interface that produces normalized vehicle controls.
- `policy_driver.py` — obtains controls from a trained RL policy.
- `human_driver.py` — obtains controls from keyboard/gamepad input supplied by a client/runtime adapter.

This separation allows Human vs AI modes without duplicating vehicle physics.
