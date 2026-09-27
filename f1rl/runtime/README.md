# `f1rl.runtime`

## Purpose

Application-level orchestration for a live F1-RL run.

## Planned files

- `session.py` — lifecycle of a race/demo session, selected track, driver, timer, and result.
- `simulation.py` — fixed-step runtime loop coordinating sandbox stepping, driver actions, and emitted snapshots.

Runtime may depend on sandbox, drivers, and learning inference, but not on the browser implementation.
