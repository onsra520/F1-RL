from __future__ import annotations

from dataclasses import dataclass

from ..entities.boost_pad import BoostPad
from ..entities.checkpoint import Checkpoint
from ..entities.finish_line import FinishLine
from ..entities.obstacle import Obstacle
from ..entities.score_pickup import ScorePickup
from ..entities.slow_zone import SlowZone
from ..surface import SurfaceContact


@dataclass(frozen=True, slots=True)
class TrackQuery:
    contact: SurfaceContact | None
    obstacles: tuple[Obstacle, ...]
    boost_pads: tuple[BoostPad, ...]
    slow_zones: tuple[SlowZone, ...]
    checkpoints: tuple[Checkpoint, ...]
    finish_lines: tuple[FinishLine, ...]
    score_pickups: tuple[ScorePickup, ...]
