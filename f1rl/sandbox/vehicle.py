from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

from .geometry.rotation import Quaternion
from .geometry.vector import Vec3


@dataclass(frozen=True, slots=True)
class VehicleState:
    position: Vec3
    velocity: Vec3
    orientation: Quaternion
    angular_velocity: Vec3
    grounded: bool

    @property
    def speed(self) -> float:
        return self.velocity.magnitude


@dataclass(frozen=True, slots=True)
class VehicleControl:
    steering: float
    throttle: float
    brake: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.steering <= 1.0:
            raise ValueError("steering must be in [-1, 1]")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be in [0, 1]")
        if not 0.0 <= self.brake <= 1.0:
            raise ValueError("brake must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class VehicleParameters:
    wheelbase: float
    ride_height: float
    max_steering_angle: float
    max_acceleration: float
    max_braking: float
    rolling_drag: float
    air_drag: float
    max_ground_speed: float
    gravity: float

    def __post_init__(self) -> None:
        values = (
            self.wheelbase,
            self.ride_height,
            self.max_steering_angle,
            self.max_acceleration,
            self.max_braking,
            self.rolling_drag,
            self.air_drag,
            self.max_ground_speed,
            self.gravity,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("vehicle parameters must be finite")
        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if self.ride_height <= 0.0:
            raise ValueError("ride_height must be positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if not 0.0 <= self.max_steering_angle < pi / 2.0:
            raise ValueError("max_steering_angle must be in [0, pi/2)")
        for name in (
            "max_acceleration",
            "max_braking",
            "rolling_drag",
            "air_drag",
            "max_ground_speed",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
