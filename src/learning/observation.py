from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..sandbox.geometry.vector import Vec3
from ..sandbox.perception.sensor_suite import SensorSuite
from ..sandbox.track.model import TrackWorld
from ..sandbox.vehicle import VehicleState

_LOCAL_RIGHT = Vec3(1.0, 0.0, 0.0)
_LOCAL_UP = Vec3(0.0, 1.0, 0.0)
_LOCAL_FORWARD = Vec3(0.0, 0.0, 1.0)
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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


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

    def encode(
        self,
        track: TrackWorld,
        vehicle: VehicleState,
        target: NavigationTarget | None,
    ) -> tuple[float, ...]:
        readings = self.config.sensor_suite.sense(track, vehicle)
        sensor_features = tuple(
            _clamp(reading.distance / spec.max_distance, 0.0, 1.0)
            for spec, reading in zip(self.config.sensor_suite.rangefinders, readings, strict=True)
        )
        right = vehicle.orientation.rotate(_LOCAL_RIGHT)
        up = vehicle.orientation.rotate(_LOCAL_UP)
        forward = vehicle.orientation.rotate(_LOCAL_FORWARD)
        velocity_features = (
            _clamp(vehicle.velocity.dot(right) / self.config.velocity_scale, -1.0, 1.0),
            _clamp(vehicle.velocity.dot(up) / self.config.velocity_scale, -1.0, 1.0),
            _clamp(vehicle.velocity.dot(forward) / self.config.velocity_scale, -1.0, 1.0),
        )

        if target is None:
            target_direction = (0.0, 0.0, 0.0)
            target_distance = 0.0
            target_available = 0.0
        else:
            delta = target.position - vehicle.position
            distance = delta.magnitude
            if distance == 0.0:
                target_direction = (0.0, 0.0, 0.0)
            else:
                direction = delta / distance
                target_direction = (
                    _clamp(direction.dot(right), -1.0, 1.0),
                    _clamp(direction.dot(up), -1.0, 1.0),
                    _clamp(direction.dot(forward), -1.0, 1.0),
                )
            target_distance = _clamp(distance / self.config.target_distance_scale, 0.0, 1.0)
            target_available = 1.0

        return (
            *sensor_features,
            *velocity_features,
            *target_direction,
            target_distance,
            target_available,
            1.0 if vehicle.grounded else 0.0,
        )
