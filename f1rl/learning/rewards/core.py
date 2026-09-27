from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from ...sandbox.geometry.vector import Vec3
from ...sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from ...sandbox.track.state import ScorePickupCollected
from ...sandbox.vehicle import VehicleState

_ALLOWED_EVENTS = (ObstacleHit, CheckpointEntered, FinishLineEntered, ScorePickupCollected)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class RewardTransition:
    previous_vehicle: VehicleState
    next_vehicle: VehicleState
    target_position: Vec3 | None
    events: tuple[object, ...]
    dt: float

    def __post_init__(self) -> None:
        if not isinstance(self.previous_vehicle, VehicleState) or not isinstance(self.next_vehicle, VehicleState):
            raise ValueError("vehicle states must be VehicleState instances")
        if self.target_position is not None and not isinstance(self.target_position, Vec3):
            raise ValueError("target_position must be Vec3 or None")
        if not isinstance(self.events, tuple) or not all(isinstance(event, _ALLOWED_EVENTS) for event in self.events):
            raise ValueError("events must be a tuple of current WorldEvent variants")
        dt = _finite_number(self.dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive")


@dataclass(frozen=True, slots=True)
class RewardContribution:
    name: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        _finite_number(self.value, "value")


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    contributions: tuple[RewardContribution, ...]

    def __post_init__(self) -> None:
        _finite_number(self.total, "total")
        if not isinstance(self.contributions, tuple):
            raise ValueError("contributions must be a tuple")


class RewardTerm(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self, transition: RewardTransition) -> RewardContribution: ...
