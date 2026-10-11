from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from ...sandbox.geometry.vector import Vec3
from ...sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from ...sandbox.track.state import ScorePickupCollected
from ...sandbox.vehicle import VehicleState
from ...sandbox.world import WorldEvent

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
    events: tuple[WorldEvent, ...]
    dt: float
    route_progress_delta_meters: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.previous_vehicle, VehicleState) or not isinstance(self.next_vehicle, VehicleState):
            raise ValueError("vehicle states must be VehicleState instances")
        if self.target_position is not None and not isinstance(self.target_position, Vec3):
            raise ValueError("target_position must be Vec3 or None")
        if not isinstance(self.events, tuple) or not all(isinstance(event, _ALLOWED_EVENTS) for event in self.events):
            raise ValueError("events must be a tuple of current WorldEvent variants")
        if _finite_number(self.dt, "dt") <= 0.0:
            raise ValueError("dt must be strictly positive")
        if self.route_progress_delta_meters is not None and _finite_number(
            self.route_progress_delta_meters, "route_progress_delta_meters"
        ) < 0.0:
            raise ValueError("route_progress_delta_meters must be non-negative")


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


@dataclass(frozen=True, slots=True)
class RewardComposer:
    terms: tuple[RewardTerm, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.terms, tuple):
            raise ValueError("terms must be a tuple")
        names: list[str] = []
        for term in self.terms:
            name = getattr(term, "name", None)
            evaluate = getattr(term, "evaluate", None)
            if not isinstance(name, str) or not name.strip() or not callable(evaluate):
                raise ValueError("each term must expose a non-empty name and callable evaluate")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("reward term names must be unique")

    def evaluate(self, transition: RewardTransition) -> RewardResult:
        contributions: list[RewardContribution] = []
        total = 0.0
        for term in self.terms:
            contribution = term.evaluate(transition)
            if not isinstance(contribution, RewardContribution):
                raise ValueError("reward term must return RewardContribution")
            if contribution.name != term.name:
                raise ValueError("reward contribution name must match term name")
            _finite_number(contribution.value, "reward contribution value")
            contributions.append(contribution)
            total += contribution.value
        return RewardResult(total, tuple(contributions))
