from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..sandbox.geometry.vector import Vec3
from ..sandbox.perception.sensor_suite import SensorSuite

_FIXED_FEATURE_NAMES = (
    "velocity_local_x",
    "velocity_local_y",
    "velocity_local_z",
    "target_direction_local_x",
    "target_direction_local_y",
    "target_direction_local_z",
    "target_distance",
    "target_available",
    "grounded",
)


def _validate_positive_scale(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    sensor_suite: SensorSuite
    velocity_scale: float
    target_distance_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_suite, SensorSuite):
            raise ValueError("sensor_suite must be a SensorSuite")
        _validate_positive_scale(self.velocity_scale, "velocity_scale")
        _validate_positive_scale(self.target_distance_scale, "target_distance_scale")


@dataclass(frozen=True, slots=True)
class NavigationTarget:
    position: Vec3


@dataclass(frozen=True, slots=True)
class ObservationEncoder:
    config: ObservationConfig

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *(f"rangefinder:{spec.id}" for spec in self.config.sensor_suite.rangefinders),
            *_FIXED_FEATURE_NAMES,
        )

    @property
    def size(self) -> int:
        return len(self.feature_names)
