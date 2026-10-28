# `tests`

## Purpose

Automated verification of the F1-RL core and adapters.

## Subdirectories

- `sandbox/` — deterministic simulation, dynamics, sensing, entities, and rules.
- `learning/` — observation/action, reward, Gymnasium, algorithm adapters, and metrics.
- `runtime/` — session and simulation-loop behavior.
- `integration/` — cross-layer tests such as API/runtime and policy/sandbox flows.

Tests should mirror ownership boundaries in `src/`.
