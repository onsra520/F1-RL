# `f1rl.learning.algorithms`

## Purpose

Thin adapters around reinforcement-learning algorithm libraries. The algorithm layer does not own simulation, reward design, route semantics, training orchestration, or evaluation.

## PPO baseline

`ppo.py` implements the Stable-Baselines3 PPO baseline through:

- `PPOConfig` — validated hyperparameters and YAML loading;
- `PPOModel` — create, predict, cumulative learn, save, and load;
- fixed `MlpPolicy` for the vector observation baseline.

Install the optional RL stack with:

```text
pip install -e ".[rl]"
```

Editable defaults live in `configs/training/ppo-baseline.yml`.

## Device modes

`device` accepts `cpu`, `cuda`, or `auto`. The checked-in baseline uses `cpu`.

CUDA is a hybrid mode: F1-RL simulation, sensors, rewards, and Gym environment stepping remain on CPU while the PPO policy/value network may run on the GPU. Explicit unavailable `cuda` fails rather than silently falling back; `auto` delegates device selection to SB3/PyTorch.

## Persistence and scope

Training is cumulative across `learn()` calls. `save()` and `load()` use Stable-Baselines3 `.zip` archives; generated policies belong under ignored artifact directories such as `artifacts/policies/` or `artifacts/checkpoints/`.

Vectorized environments, multi-track sampling, training orchestration, callbacks, evaluation, and generalization experiments remain later milestones.
