# `f1rl.learning.rewards`

## Purpose

Composable reward shaping for reinforcement learning experiments.

## Planned files

- `progress.py` — rewards progress toward the active checkpoint/goal.
- `safety.py` — collision and unsafe-proximity penalties.
- `racing.py` — speed, timing, racing-line, boost, and race-specific incentives.
- `composer.py` — combines configured reward terms into the final scalar reward and exposes per-term diagnostics.

Reward code must consume observable/domain state without mutating the sandbox.
