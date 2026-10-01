# `configs`

## Purpose

Human-readable configuration files that define reproducible F1-RL runs and scientific experiment protocols without hard-coding parameters.

## Subdirectories

- `training/` — lower-level PPO and vector-environment runtime/training configuration.
- `experiments/` — canonical scientific experiment definitions with explicit draft/frozen lifecycle and protocol versioning.

Root `config.yml` remains the editable training-orchestration input. A checked-in experiment definition may reference it while the experiment is still a draft, but final scientific training settings are frozen separately after pilot calibration rather than inferred from mutable runtime defaults.
