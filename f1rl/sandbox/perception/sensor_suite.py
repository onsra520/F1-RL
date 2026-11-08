from __future__ import annotations

from dataclasses import dataclass

from ..track.model import TrackWorld
from ..vehicle import VehicleState
from .rangefinder import RangefinderReading, RangefinderSpec, sense_rangefinder


@dataclass(frozen=True, slots=True)
class SensorSuite:
    rangefinders: tuple[RangefinderSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rangefinders, tuple):
            raise ValueError("rangefinders must be a tuple")
        ids = tuple(spec.id for spec in self.rangefinders)
        if len(ids) != len(set(ids)):
            raise ValueError("rangefinder ids must be unique")

    def sense(
        self,
        track: TrackWorld,
        vehicle: VehicleState,
    ) -> tuple[RangefinderReading, ...]:
        return tuple(
            sense_rangefinder(spec, track, vehicle)
            for spec in self.rangefinders
        )
