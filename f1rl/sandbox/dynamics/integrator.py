from __future__ import annotations

from ..geometry.rotation import Quaternion
from ..geometry.vector import Vec3
from ..surface import SurfaceContact
from ..vehicle import VehicleParameters, VehicleState
from .vehicle_dynamics import MotionDerivatives


LOCAL_FORWARD = Vec3(0.0, 0.0, 1.0)
WORLD_FORWARD = Vec3(0.0, 0.0, 1.0)
WORLD_RIGHT = Vec3(1.0, 0.0, 0.0)
_EPSILON = 1e-12


def _advance_orientation(
    orientation: Quaternion,
    angular_velocity: Vec3,
    dt: float,
) -> Quaternion:
    angular_speed = angular_velocity.magnitude
    if angular_speed <= _EPSILON:
        return orientation
    delta = Quaternion.from_axis_angle(angular_velocity / angular_speed, angular_speed * dt)
    return delta * orientation


def _surface_aligned_orientation(state: VehicleState, normal: Vec3) -> Quaternion:
    forward = state.orientation.rotate(LOCAL_FORWARD).reject_from(normal)
    if forward.magnitude <= _EPSILON:
        forward = state.velocity.reject_from(normal)
    if forward.magnitude <= _EPSILON:
        reference = WORLD_FORWARD if abs(normal.z) < 0.9 else WORLD_RIGHT
        forward = reference.reject_from(normal)

    forward = forward.normalized()
    right = normal.cross(forward).normalized()
    forward = right.cross(normal).normalized()
    return Quaternion.from_basis(right, normal, forward)


def _constrain_ground_velocity(
    velocity: Vec3,
    forward: Vec3,
    normal: Vec3,
    max_speed: float,
) -> Vec3:
    tangent = velocity.reject_from(normal)
    longitudinal = tangent.dot(forward)
    if longitudinal < 0.0:
        tangent = tangent - forward * longitudinal

    speed = tangent.magnitude
    if speed > max_speed:
        if max_speed == 0.0:
            return Vec3(0.0, 0.0, 0.0)
        tangent = tangent * (max_speed / speed)
    return tangent


def integrate(
    state: VehicleState,
    derivatives: MotionDerivatives,
    parameters: VehicleParameters,
    contact: SurfaceContact | None,
    dt: float,
) -> VehicleState:
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    velocity = state.velocity + derivatives.linear_acceleration * dt

    if contact is None:
        orientation = _advance_orientation(state.orientation, derivatives.angular_velocity, dt)
        position = state.position + velocity * dt
        return VehicleState(position, velocity, orientation, derivatives.angular_velocity, False)

    aligned = _surface_aligned_orientation(state, contact.normal)
    forward = aligned.rotate(LOCAL_FORWARD)
    velocity = _constrain_ground_velocity(
        velocity,
        forward,
        contact.normal,
        parameters.max_ground_speed,
    )
    position = state.position + velocity * dt
    signed_distance = (position - contact.point).dot(contact.normal)
    position = position + contact.normal * (parameters.ride_height - signed_distance)
    orientation = _advance_orientation(aligned, derivatives.angular_velocity, dt)

    return VehicleState(position, velocity, orientation, derivatives.angular_velocity, True)
