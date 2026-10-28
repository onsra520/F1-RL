from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..entities.boost_pad import BoostPad
from ..entities.checkpoint import Checkpoint
from ..entities.finish_line import FinishLine
from ..entities.obstacle import Obstacle
from ..entities.score_pickup import ScorePickup
from ..entities.slow_zone import SlowZone
from ..geometry.vector import Vec3
from ..surface import SurfaceContact
from .patch import PlanarSurfacePatch
from .query import TrackQuery
from .state import TrackState


@dataclass(frozen=True, slots=True)
class TrackWorld:
    surfaces: tuple[PlanarSurfacePatch, ...] = ()
    obstacles: tuple[Obstacle, ...] = ()
    boost_pads: tuple[BoostPad, ...] = ()
    slow_zones: tuple[SlowZone, ...] = ()
    checkpoints: tuple[Checkpoint, ...] = ()
    finish_lines: tuple[FinishLine, ...] = ()
    score_pickups: tuple[ScorePickup, ...] = ()

    def __post_init__(self) -> None:
        field_names = (
            "surfaces",
            "obstacles",
            "boost_pads",
            "slow_zones",
            "checkpoints",
            "finish_lines",
            "score_pickups",
        )
        all_ids: list[str] = []
        for field_name in field_names:
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise ValueError(f"{field_name} must be a tuple")
            ordered = tuple(sorted(values, key=lambda item: item.id))
            object.__setattr__(self, field_name, ordered)
            all_ids.extend(item.id for item in ordered)

        if len(all_ids) != len(set(all_ids)):
            raise ValueError("track world ids must be unique across all categories")

    def surface_at(self, position: Vec3, max_surface_distance: float) -> SurfaceContact | None:
        if not isfinite(max_surface_distance) or max_surface_distance < 0.0:
            raise ValueError("max_surface_distance must be finite and non-negative")

        candidates: list[tuple[float, str, SurfaceContact]] = []
        for patch in self.surfaces:
            contact = patch.sample(position, max_surface_distance)
            if contact is None:
                continue
            distance = abs((position - contact.point).dot(contact.normal))
            candidates.append((distance, patch.id, contact))

        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    def query_point(
        self,
        position: Vec3,
        max_surface_distance: float,
        state: TrackState,
    ) -> TrackQuery:
        contact = self.surface_at(position, max_surface_distance)
        obstacles = tuple(item for item in self.obstacles if item.volume.contains(position))
        boost_pads = tuple(item for item in self.boost_pads if item.volume.contains(position))
        slow_zones = tuple(item for item in self.slow_zones if item.volume.contains(position))
        checkpoints = tuple(item for item in self.checkpoints if item.volume.contains(position))
        finish_lines = tuple(item for item in self.finish_lines if item.volume.contains(position))
        score_pickups = tuple(
            item
            for item in self.score_pickups
            if item.id not in state.collected_pickup_ids and item.volume.contains(position)
        )

        return TrackQuery(
            contact=contact,
            obstacles=obstacles,
            boost_pads=boost_pads,
            slow_zones=slow_zones,
            checkpoints=checkpoints,
            finish_lines=finish_lines,
            score_pickups=score_pickups,
        )
