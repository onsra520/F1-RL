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

The action, observation, reward, and episode primitives remain framework-agnostic. `gym_environment.py` is the thin Gymnasium adapter above them: one env is bound to one track, exposes NumPy `float32` spaces, composes deterministic reset/step behavior, preserves pre-step reward target timing, and keeps game score separate from RL reward.

`algorithms/ppo.py` now provides the implemented PPO baseline through Stable-Baselines3. `PPOConfig` can load editable defaults from `configs/training/ppo-baseline.yml`; `PPOModel` supports deterministic prediction, cumulative learning, and `.zip` save/load. Model compute may use `cpu`, `cuda`, or `auto`, while the F1-RL simulation remains CPU-side.

The PPO symbols intentionally are not re-exported from `f1rl.learning`, so the base simulation/Gym package remains usable without the optional RL dependencies.

Vectorized PPO training is **implemented** above the baseline adapter, and multi-track sampling is implemented in the training layer. `make_vector_env()` keeps the same-track `DummyVecEnv` or Windows-safe spawned `SubprocVecEnv` path. `make_multi_track_vector_env()` samples compatible authored training scenarios on each episode reset while preserving fixed-track `F1RLEnv` instances and the shared non-track training contract. Rollout-size validation remains in `PPOModel` at environment binding time. Repeatable training orchestration is also implemented: root `config.yml` resolves into `TrainingRunConfig`, `TrainingRunner` supports explicit single-track and multi-track runs, and generated checkpoints/state/final models live under `artifacts/runs/`.

## Next layers

- independent evaluation metrics;
- unseen-track generalization experiments;
- reward-shaping ablation.

Evaluation and generalization are not implied by the PPO baseline smoke pipeline. Sandbox remains reusable without importing the learning package. Policy-controlled braking remains a separate future action-contract change.
