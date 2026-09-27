# `f1rl.learning`

## Purpose

Reinforcement-learning layer built on top of the sandbox.

## Current baseline

`observation.py` provides deterministic local-frame observation encoding:

- rangefinder distances are normalized in exact `SensorSuite` declaration order;
- vehicle linear velocity is projected into local right/up/forward axes and scaled/clamped;
- an explicit `NavigationTarget | None` supplies vehicle-relative target direction and distance;
- `target_available` distinguishes no target from standing exactly on a target;
- grounded is emitted as an exact `0.0`/`1.0` flag;
- observation size is always `sensor_count + 9` and matches stable `feature_names` metadata.

The encoder returns pure-Python `tuple[float, ...]` values. It does not choose the next checkpoint, track race progress, calculate rewards, map actions, step the simulation, or depend on NumPy/Gymnasium/PyTorch.

## Next layers

- `action.py` — deterministic mapping from policy actions into `VehicleControl`.
- `gym_environment.py` — future Gymnasium adapter around observation, action, simulation, and reward contracts.
- `rewards/` — composable RL reward terms, kept separate from game score.
- `algorithms/` / `training/` / `evaluation/` — PPO integration, training orchestration, and generalization metrics.

Route-progress/next-checkpoint selection remains a separate authority that supplies the current `NavigationTarget`; changing that policy must not require changing observation encoding. Sandbox remains reusable without importing the learning package.
