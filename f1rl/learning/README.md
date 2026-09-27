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

`action.py` provides the strict V1 policy-action contract:

- action order is exactly `("steering", "throttle")`;
- bounds are exactly `(-1.0, 0.0)` to `(1.0, 1.0)`;
- invalid, non-finite, boolean, and out-of-range values fail fast;
- mapping copies steering/throttle exactly with no clamp or remap;
- every mapped `VehicleControl` has `brake == 0.0`.

Both layers return pure-Python values and remain framework-agnostic. They do not choose route progress, calculate rewards, own simulation stepping, or depend on NumPy/Gymnasium/PyTorch/PPO.

## Next layers

- `gym_environment.py` — future Gymnasium adapter around observation, action, simulation, and reward contracts.
- `rewards/` — composable RL reward terms, kept separate from game score.
- `algorithms/` / `training/` / `evaluation/` — PPO integration, training orchestration, and generalization metrics.

Route-progress/next-checkpoint selection remains a separate authority that supplies the current `NavigationTarget`. Policy-controlled braking is also a separate future action-contract change; V1 remains two-dimensional with brake fixed off. Sandbox remains reusable without importing the learning package.
