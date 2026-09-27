# `f1rl.learning.algorithms`

## Purpose

Small adapters around reinforcement-learning algorithms and libraries.

## Planned files

- `base.py` — common train/save/load/predict contract used by F1-RL.
- `ppo.py` — PPO adapter, initially backed by Stable-Baselines3.

Algorithm wrappers must not own simulation, reward, or experiment-reporting responsibilities.
