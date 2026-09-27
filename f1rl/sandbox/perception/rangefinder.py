from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..geometry.vector import Vec3
from ..track.model import TrackWorld
from ..vehicle import VehicleState

_UNIT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class RangefinderSpec:
    id: str
    local_direction: Vec3
    max_distance: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("rangefinder id must be a non-empty string")
        if abs(self.local_direction.magnitude - 1.0) > _UNIT_TOLERANCE:
            raise ValueError("local_direction must be unit length")
        if isinstance(self.max_distance, bool) or not isinstance(self.max_distance, (int, float)):
            raise ValueError("max_distance must be a finite positive number")
        if not isfinite(self.max_distance) or self.max_distance <= 0.0:
            raise ValueError("max_distance must be a finite positive number")


@dataclass(frozen=True, slots=True)
class RangefinderReading:
    sensor_id: str
    distance: float
    hit_id: str | None


def sense_rangefinder(
    spec: RangefinderSpec,
    track: TrackWorld,
    vehicle: VehicleState,
) -> RangefinderReading:
    world_direction = vehicle.orientation.rotate(spec.local_direction)
    candidates: list[tuple[float, str]] = []
    for obstacle in track.obstacles:
        distance = obstacle.volume.ray_distance(
            vehicle.position,
            world_direction,
            spec.max_distance,
        )
        if distance is not None:
            candidates.append((distance, obstacle.id))

    if not candidates:
        return RangefinderReading(spec.id, spec.max_distance, None)

    distance, obstacle_id = min(candidates, key=lambda item: (item[0], item[1]))
    return RangefinderReading(spec.id, distance, obstacle_id)
