# `f1rl`

## Purpose

The Python domain core of F1-RL. Everything required to simulate, train, evaluate, and run an autonomous racing agent belongs behind this package boundary.

## Subdirectories

- `sandbox/` — deterministic racing world and vehicle simulation.
- `track_assets/` — portable JSON track persistence, semantic fingerprints, manifest integrity, and deterministic initial-state derivation.
- `learning/` — Gymnasium adapter, observations, actions, rewards, algorithms, training, and evaluation.
- `drivers/` — human and learned-policy control adapters.
- `runtime/` — orchestration of live simulation sessions.
- `api/` — FastAPI adapter for external clients.

## Dependency rule

`sandbox/` is the lowest deterministic domain layer and must not import `track_assets/`, learning, runtime, API, or web concerns. `track_assets/` may depend on sandbox domain types, but must not depend on learning/runtime frameworks, Gymnasium, Stable-Baselines3, PyTorch, FastAPI, or frontend/editor code.

Lower-level domain code must not import the web client.
