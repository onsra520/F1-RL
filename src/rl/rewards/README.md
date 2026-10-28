# `src.rl.rewards`

## Purpose

Deterministic, composable reward shaping for reinforcement-learning experiments.

## V1 contract

- `core.py` defines `RewardTransition`, contribution/result DTOs, `RewardTerm`, and `RewardComposer`.
- `terms.py` defines `TargetProgressReward`, `CollisionPenalty`, and `TimePenalty`.
- One transition uses the navigation target active before the step for both before/after distance measurements.
- Progress uses signed 3D Euclidean distance change.
- Collision is penalized once per transition even when overlapping obstacles emit several hits.
- Time pressure scales by elapsed seconds.
- Composer output preserves declaration order and exposes an inspectable contribution breakdown.
- Reward values are not clipped or normalized in this semantic layer.

Game score remains separate from RL reward. Score pickups, checkpoint-entry, and finish-line events create no implicit V1 reward. Valid checkpoint/lap progression and terminal finish semantics belong to a future route-progress/episode layer.

Production reward code depends only on sandbox domain/event types plus the Python standard library; it does not own simulation stepping or depend on Gymnasium, NumPy, PyTorch, or PPO.
