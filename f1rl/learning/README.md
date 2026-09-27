# `f1rl.learning`

## Purpose

Reinforcement-learning layer built on top of the sandbox.

## Current baseline

`observation.py` provides deterministic local-frame observation encoding with declared-order rangefinders, local velocity, explicit vehicle-relative navigation target features, target availability, and grounded state. Observation size is always `sensor_count + 9`.

`action.py` provides the strict V1 two-dimensional policy-action contract: `("steering", "throttle")`, bounds `(-1.0, 0.0)` to `(1.0, 1.0)`, fail-fast validation, exact mapping with no remap/clamp, and `brake == 0.0`.

`rewards/` provides deterministic transition rewards:

- `TargetProgressReward` measures signed 3D distance progress toward one fixed pre-step target;
- `CollisionPenalty` applies at most once per transition;
- `TimePenalty` scales by elapsed seconds;
- `RewardComposer` preserves term order and returns an inspectable per-term breakdown.

These layers remain pure-Python and framework-agnostic. Game score is not RL reward, and checkpoint/finish entry events have no implicit reward because validated route-progress and terminal episode semantics do not exist yet.

## Next layers

- explicit route-progress / episode semantics: authoritative next checkpoint, ordered completion, lap state, valid finish, termination/truncation;
- `gym_environment.py`: compose observation, action, simulation, route progress, reward, reset, and episode semantics;
- `algorithms/` / `training/` / `evaluation/`: PPO integration, training orchestration, reward-shaping ablation, and generalization metrics.

Sandbox remains reusable without importing the learning package. Policy-controlled braking remains a separate future action-contract change.
