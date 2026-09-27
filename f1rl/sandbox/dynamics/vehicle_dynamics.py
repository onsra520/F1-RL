from __future__ import annotations

from dataclasses import dataclass

from ..geometry.vector import Vec3
from ..surface import SurfaceContact
from ..vehicle import VehicleControl, VehicleParameters, VehicleState
from .air_dynamics import compute_air_motion
from .ground_dynamics import compute_ground_motion


@dataclass(frozen=True, slots=True)
class MotionDerivatives:
    linear_acceleration: Vec3
    angular_velocity: Vec3


def compute_motion(
    state: VehicleState,
    control: VehicleControl,
    parameters: VehicleParameters,
    contact: SurfaceContact | None,
) -> MotionDerivatives:
    if contact is None:
        motion = compute_air_motion(state, control, parameters)
    else:
        motion = compute_ground_motion(state, control, parameters, contact)
    return MotionDerivatives(motion.linear_acceleration, motion.angular_velocity)
