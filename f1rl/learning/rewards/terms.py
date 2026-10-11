from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ...sandbox.interactions.events import ObstacleHit
from .core import RewardContribution, RewardTransition


def _validate_non_negative(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class TargetProgressReward:
    weight_per_meter: float

    def __post_init__(self) -> None:
        _validate_non_negative(self.weight_per_meter, "weight_per_meter")

    @property
    def name(self) -> str:
        return "target_progress"

    def evaluate(self, transition: RewardTransition) -> RewardContribution:
        if self.weight_per_meter == 0.0:
            return RewardContribution(self.name, 0.0)
        target = transition.target_position
        if target is None:
            return RewardContribution(self.name, 0.0)
        previous_distance = (target - transition.previous_vehicle.position).magnitude
        next_distance = (target - transition.next_vehicle.position).magnitude
        return RewardContribution(self.name, self.weight_per_meter * (previous_distance - next_distance))


@dataclass(frozen=True, slots=True)
class RouteProgressReward:
    """Positive credit only for genuinely new checkpoint-gated arc progress."""
    weight_per_meter: float

    def __post_init__(self) -> None:
        _validate_non_negative(self.weight_per_meter, "weight_per_meter")

    @property
    def name(self) -> str:
        return "route_progress"

    def evaluate(self, transition: RewardTransition) -> RewardContribution:
        delta = transition.route_progress_delta_meters
        if delta is None:
            raise ValueError("route_progress reward requires authored route delta evidence")
        return RewardContribution(self.name, self.weight_per_meter * delta)


@dataclass(frozen=True, slots=True)
class CollisionPenalty:
    penalty: float

    def __post_init__(self) -> None:
        _validate_non_negative(self.penalty, "penalty")

    @property
    def name(self) -> str:
        return "collision"

    def evaluate(self, transition: RewardTransition) -> RewardContribution:
        if self.penalty == 0.0:
            return RewardContribution(self.name, 0.0)
        hit = any(isinstance(event, ObstacleHit) for event in transition.events)
        return RewardContribution(self.name, -self.penalty if hit else 0.0)


@dataclass(frozen=True, slots=True)
class TimePenalty:
    penalty_per_second: float

    def __post_init__(self) -> None:
        _validate_non_negative(self.penalty_per_second, "penalty_per_second")

    @property
    def name(self) -> str:
        return "time"

    def evaluate(self, transition: RewardTransition) -> RewardContribution:
        if self.penalty_per_second == 0.0:
            return RewardContribution(self.name, 0.0)
        return RewardContribution(self.name, -self.penalty_per_second * transition.dt)
