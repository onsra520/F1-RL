# `f1rl`

## Purpose

The Python domain core of F1-RL. Everything required to simulate, train, evaluate, and run an autonomous racing agent belongs behind this package boundary.

## Subdirectories

- `sandbox/` — deterministic racing world and vehicle simulation.
- `learning/` — Gymnasium adapter, observations, actions, rewards, algorithms, training, and evaluation.
- `drivers/` — human and learned-policy control adapters.
- `runtime/` — orchestration of live simulation sessions.
- `api/` — FastAPI adapter for external clients.

## Dependency rule

Lower-level domain code must not import the web client. `sandbox/` must not know about FastAPI or a specific RL algorithm.
