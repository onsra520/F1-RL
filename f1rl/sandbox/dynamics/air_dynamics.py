from __future__ import annotations

from dataclasses import dataclass

from ..geometry.vector import Vec3
from ..vehicle import VehicleControl, VehicleParameters, VehicleState
from .modifiers import MotionModifiers


@dataclass(frozen=True, slots=True)
class AirMotion:
    linear_acceleration: Vec3
    angular_velocity: Vec3


def compute_air_motion(
    state: VehicleState,
    control: VehicleControl,
    parameters: VehicleParameters,
    modifiers: MotionModifiers | None = None,
) -> AirMotion:
    del control, modifiers
    gravity = Vec3(0.0, -parameters.gravity, 0.0)
    drag = state.velocity * (-parameters.air_drag)
    return AirMotion(gravity + drag, state.angular_velocity)
