# `f1rl.learning`

## Purpose

Reinforcement-learning layer built on top of the sandbox.

## Planned files

- `gym_environment.py` — Gymnasium adapter around the F1-RL sandbox.
- `observation.py` — converts raw domain state and sensor readings into model observations.
- `action.py` — maps model actions to normalized vehicle controls.

## Subdirectories

- `rewards/` — composable reward terms.
- `algorithms/` — algorithm adapters such as PPO.
- `training/` — training orchestration.
- `evaluation/` — metrics and deterministic evaluation.

The sandbox remains reusable without this package.
