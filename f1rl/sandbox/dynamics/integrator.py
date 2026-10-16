from __future__ import annotations

from math import isfinite, pi, sin, tan

from ..geometry.rotation import Quaternion
from ..geometry.vector import Vec3
from ..surface import SurfaceContact
from ..vehicle import VehicleControl, VehicleParameters, VehicleState
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
    previous_forward: Vec3,
    next_forward: Vec3,
    normal: Vec3,
    max_speed: float,
) -> Vec3:
    # Ground contact enforces a bicycle-model no-slip constraint. Retain only
    # forward motion along the prior heading, then align it to the steered
    # heading without introducing artificial speed loss on turns.
    tangent = velocity.reject_from(normal)
    longitudinal = min(max(tangent.dot(previous_forward), 0.0), max_speed)
    return next_forward * longitudinal


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
    previous_forward = aligned.rotate(LOCAL_FORWARD)
    orientation = _advance_orientation(aligned, derivatives.angular_velocity, dt)
    next_forward = orientation.rotate(LOCAL_FORWARD).reject_from(contact.normal).normalized()
    velocity = _constrain_ground_velocity(
        velocity,
        previous_forward,
        next_forward,
        contact.normal,
        parameters.max_ground_speed,
    )
    position = state.position + velocity * dt
    signed_distance = (position - contact.point).dot(contact.normal)
    position = position + contact.normal * (parameters.ride_height - signed_distance)

    return VehicleState(position, velocity, orientation, derivatives.angular_velocity, True)


def _distance_ground_frame(state: VehicleState,parameters: VehicleParameters):
    aligned = _surface_aligned_orientation(state,Vec3(0.,1.,0.))
    forward = aligned.rotate(LOCAL_FORWARD)
    tangent = Vec3(state.velocity.x,0.,state.velocity.z)
    speed = tangent.dot(forward)
    if (speed < -1e-8 or speed > parameters.max_ground_speed+1e-8
            or (tangent-forward*speed).magnitude>1e-8):
        raise ValueError('distance ground integration requires bounded forward no-slip velocity')
    return aligned,forward,max(0.,min(speed,parameters.max_ground_speed))


def integrate_ground_distance(
    state: VehicleState, derivatives: MotionDerivatives, control: VehicleControl,
    parameters: VehicleParameters, contact: SurfaceContact, dt: float,
) -> VehicleState:
    """Flat no-slip bicycle step with distance/yaw from one force-frozen law.

    Longitudinal force is frozen for the tick as in compute_ground_motion.
    Stop/speed-cap event times split the distance integral. This is a reduced
    constrained bicycle model, not tire-slip or rigid-body force validation.
    """
    if (type(dt) not in (int,float) or not isfinite(dt) or dt<=0
            or contact.normal!=Vec3(0.,1.,0.)):
        raise ValueError('distance ground integration requires finite dt and horizontal contact')
    aligned,forward,speed = _distance_ground_frame(state,parameters)
    acceleration = derivatives.linear_acceleration.dot(forward)
    moving_dt = dt
    tail_speed = None
    if acceleration<0 and speed+acceleration*dt<0:
        moving_dt = -speed/acceleration
        tail_speed = 0.
    elif acceleration>0 and speed+acceleration*dt>parameters.max_ground_speed:
        moving_dt = (parameters.max_ground_speed-speed)/acceleration
        tail_speed = parameters.max_ground_speed
    end_speed = max(0.,min(speed+acceleration*dt,parameters.max_ground_speed))
    distance = speed*moving_dt+.5*acceleration*moving_dt*moving_dt
    if tail_speed is not None:
        distance += tail_speed*(dt-moving_dt)
    curvature = tan(control.steering*parameters.max_steering_angle)/parameters.wheelbase
    angle = curvature*distance
    if not isfinite(distance) or not isfinite(angle) or abs(angle)>=pi:
        raise ValueError('distance ground step exceeds finite shortest-yaw numeric domain')
    half = angle/2
    sinc = sin(half)/half if half else 1.
    midpoint_forward = Quaternion.from_axis_angle(contact.normal,half).rotate(forward)
    position = state.position+midpoint_forward*(distance*sinc)
    position = Vec3(position.x,contact.point.y+parameters.ride_height,position.z)
    orientation = Quaternion.from_axis_angle(contact.normal,angle)*aligned
    velocity = orientation.rotate(LOCAL_FORWARD)*end_speed
    return VehicleState(position,velocity,orientation,contact.normal*(curvature*end_speed),True)
