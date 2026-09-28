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

`episode/` defines the authoritative route target and deterministic single-lap progress. A route accepts checkpoints only in numeric authored order, and finishes only after all checkpoints are complete. Its result carries the next target; a finished route has no target.

`EpisodeConfig(max_steps=...)` truncates unfinished episodes at the limit. A valid finish at that same step wins: it is terminated, never truncated.

Reward timing remains fixed-target: calculate the current `RewardTransition` with the pre-step route target, then advance route progress and use its returned target for the next observation. There are no checkpoint or finish bonuses.

These layers remain pure-Python and framework-agnostic. Game score is not RL reward. Gymnasium is the next composition layer.

## Next layers

- `gym_environment.py`: compose observation, action, simulation, route progress, reward, reset, and episode semantics;
- `algorithms/` / `training/` / `evaluation/`: PPO integration, training orchestration, reward-shaping ablation, and generalization metrics.

Sandbox remains reusable without importing the learning package. Policy-controlled braking remains a separate future action-contract change.
