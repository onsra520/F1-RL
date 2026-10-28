# `f1rl.sandbox.interactions`

## Purpose

Deterministic interaction semantics derived from accepted TrackWorld queries.

## Implemented files

- `state.py` — active checkpoint and finish-line membership.
- `events.py` — checkpoint, finish, and obstacle-hit domain events.
- `resolver.py` — edge-triggered entry events, grounded track-effect aggregation, and V1 obstacle rollback.

Resolvers never query the world themselves and never calculate RL rewards.
