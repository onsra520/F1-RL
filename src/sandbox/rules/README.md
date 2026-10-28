# `f1rl.sandbox.rules`

## Purpose

Race rules derived from sandbox state and domain events.

## Implemented

- `scoring.py` — immutable game/demo score aggregation from `ScorePickupCollected` events.

## Planned

- `collision.py` — higher-fidelity collision detection/result handling beyond V1 reference-point blocking.
- `timing.py` — elapsed time and lap/session timing.

RL rewards are intentionally separate under `f1rl.learning.rewards`.
