# `f1rl.learning.training`

## Purpose

Coordinates repeatable RL training runs.

## Planned files

- `trainer.py` — constructs environments and algorithms, then executes training.
- `curriculum.py` — optional staged environment difficulty and randomization schedules.
- `callbacks.py` — logging, checkpointing, and training-time evaluation hooks.

Training configuration lives under `configs/training/`; generated outputs go to `artifacts/`.
