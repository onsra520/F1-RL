# `src.simulation.entities`

## Purpose

Immutable domain objects placed inside a track that affect navigation, vehicle modifiers, or race state.

## Planned files

- `obstacle.py` — static solid geometry described by a spatial volume.
- `boost_pad.py` — persistent acceleration-enhancing trigger zone.
- `slow_zone.py` — persistent drag-enhancing trigger zone.
- `checkpoint.py` — ordered progress trigger.
- `finish_line.py` — race-completion trigger.
- `score_pickup.py` — consumable collectible that carries game/demo score points.

Entities describe world behavior only. Collision response belongs in `src.simulation.rules`, collectible consumption state belongs in `src.simulation.track`, and RL reward logic belongs in `src.rl.rewards`.