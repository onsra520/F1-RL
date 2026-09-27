from __future__ import annotations

from dataclasses import dataclass
from math import tan

from ..geometry.vector import Vec3
from ..surface import SurfaceContact
from ..vehicle import VehicleControl, VehicleParameters, VehicleState


LOCAL_FORWARD = Vec3(0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class GroundMotion:
    linear_acceleration: Vec3
    angular_velocity: Vec3


def compute_ground_motion(
    state: VehicleState,
    control: VehicleControl,
    parameters: VehicleParameters,
    contact: SurfaceContact,
) -> GroundMotion:
    tangent_velocity = state.velocity.reject_from(contact.normal)
    forward = state.orientation.rotate(LOCAL_FORWARD).reject_from(contact.normal)
    if forward.magnitude <= 1e-12:
        forward = tangent_velocity
    if forward.magnitude <= 1e-12:
        reference = Vec3(0.0, 0.0, 1.0) if abs(contact.normal.z) < 0.9 else Vec3(1.0, 0.0, 0.0)
        forward = reference.reject_from(contact.normal)
    forward = forward.normalized()
    forward_speed = max(0.0, tangent_velocity.dot(forward))

    longitudinal = (
        control.throttle * parameters.max_acceleration
        - control.brake * parameters.max_braking
        - parameters.rolling_drag * forward_speed
    )
    if forward_speed >= parameters.max_ground_speed and longitudinal > 0.0:
        longitudinal = 0.0

    gravity = Vec3(0.0, -parameters.gravity, 0.0)
    gravity_tangent = gravity.reject_from(contact.normal)
    linear_acceleration = forward * longitudinal + gravity_tangent

    steering_angle = control.steering * parameters.max_steering_angle
    yaw_rate = 0.0
    if parameters.wheelbase > 0.0:
        yaw_rate = forward_speed / parameters.wheelbase * tan(steering_angle)
    angular_velocity = contact.normal * yaw_rate

    return GroundMotion(linear_acceleration, angular_velocity)
