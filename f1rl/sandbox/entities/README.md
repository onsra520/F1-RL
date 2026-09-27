# `f1rl.sandbox.entities`

## Purpose

Domain objects placed inside a track that affect navigation or race state.

## Planned files

- `obstacle.py` — collidable static or runtime-injected obstacles.
- `boost_pad.py` — temporary speed-enhancing zone.
- `slow_zone.py` — speed-reducing zone.
- `checkpoint.py` — ordered progress marker.
- `finish_line.py` — race completion trigger.

Entities describe world behavior; they do not contain RL policy logic.
