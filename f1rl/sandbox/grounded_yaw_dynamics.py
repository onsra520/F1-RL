"""Optional WP7 grounded planar yaw with explicit tire-force limits.

Research-only dynamic-bicycle approximation, not full 6DOF grounded contact.
World uses legacy drive/brake longitudinal candidate; this module owns only
lateral slip and yaw angular acceleration. Both axles' force magnitudes are
clamped by mu*N; the mass distribution is the symmetric axle split below.
Fails closed at speeds where slip angle/kinematic transition is undefined.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2,cos,isfinite,sin

from .geometry.vector import Vec3
from .geometry.rotation import Quaternion
from .vehicle import VehicleState

_Y=Vec3(0.,1.,0.)
_Z=Vec3(0.,0.,1.)


class GroundedYawResolutionError(RuntimeError):
    """Yaw/tire state not representable inside declared dynamics domain."""


@dataclass(frozen=True,slots=True)
class GroundedYawPolicy:
    mass_kg: float
    yaw_inertia_kg_m2: float
    front_axle_distance_m: float
    rear_axle_distance_m: float
    front_cornering_stiffness_nprad: float
    rear_cornering_stiffness_nprad: float
    tire_friction_mu: float
    min_longitudinal_speed_mps: float=1.
    max_yaw_rate_radps: float=1.
    max_slip_angle_rad: float=.3

    def __post_init__(self):
        for name in (
            "mass_kg","yaw_inertia_kg_m2",
            "front_axle_distance_m","rear_axle_distance_m",
            "front_cornering_stiffness_nprad","rear_cornering_stiffness_nprad",
            "tire_friction_mu","min_longitudinal_speed_mps",
            "max_yaw_rate_radps","max_slip_angle_rad",
        ):
            v=getattr(self,name)
            if type(v) not in (int,float) or not isfinite(v) or v<=0.:
                raise ValueError(name+" must be finite positive")
        if self.tire_friction_mu>2. or self.max_slip_angle_rad>.5:
            raise ValueError("tire domain outside registered research bounds")


@dataclass(frozen=True,slots=True)
class GroundedYawAudit:
    front_slip_rad: float
    rear_slip_rad: float
    front_force_n: float
    rear_force_n: float
    yaw_moment_nm: float
    lateral_speed_before_mps: float
    lateral_speed_after_mps: float


def advance_grounded_yaw(
    previous: VehicleState,
    prescribed: VehicleState,
    steering_rad: float,
    policy: GroundedYawPolicy,
    *,
    dt: float,
    gravity_mps2: float,
)->tuple[VehicleState,GroundedYawAudit]:
    if (not isinstance(previous,VehicleState) or not isinstance(prescribed,VehicleState)
            or not previous.grounded or not prescribed.grounded
            or not isinstance(policy,GroundedYawPolicy)):
        raise ValueError("two grounded vehicle states and explicit yaw policy required")
    for name,val in (
        ("dt",dt),("gravity",gravity_mps2),("steering",steering_rad),
    ):
        if type(val) not in (int,float) or not isfinite(val):
            raise ValueError(name+" must be finite")
    if dt<=0. or gravity_mps2<=0. or abs(steering_rad)>.7:
        raise ValueError("invalid yaw timestep/gravity/steering domain")
    # Recover yaw only, ignoring tilt already owned by suspension model.
    forward=previous.orientation.rotate(_Z)
    horizontal=Vec3(forward.x,0.,forward.z)
    if horizontal.magnitude<1e-9:
        raise GroundedYawResolutionError("vertical heading unsupported")
    forward=horizontal.normalized()
    right=_Y.cross(forward)
    yaw_rate=previous.angular_velocity.y
    if abs(yaw_rate)>policy.max_yaw_rate_radps:
        raise GroundedYawResolutionError("incoming yaw spin exceeds research range")
    longitudinal=prescribed.velocity.dot(forward)
    lateral=previous.velocity.dot(right)
    if longitudinal < policy.min_longitudinal_speed_mps:
        raise GroundedYawResolutionError("low speed requires a separate kinematic blending law")
    front_angle=atan2(lateral+yaw_rate*policy.front_axle_distance_m,longitudinal)-steering_rad
    rear_angle=atan2(lateral-yaw_rate*policy.rear_axle_distance_m,longitudinal)
    if max(abs(front_angle),abs(rear_angle))>policy.max_slip_angle_rad:
        raise GroundedYawResolutionError("slip angle beyond linear tire model")
    base=policy.front_axle_distance_m+policy.rear_axle_distance_m
    front_normal=(policy.mass_kg*gravity_mps2*policy.rear_axle_distance_m/base)
    rear_normal=(policy.mass_kg*gravity_mps2*policy.front_axle_distance_m/base)
    front=max(-policy.tire_friction_mu*front_normal,
              min(policy.tire_friction_mu*front_normal,
                  -policy.front_cornering_stiffness_nprad*front_angle))
    rear=max(-policy.tire_friction_mu*rear_normal,
             min(policy.tire_friction_mu*rear_normal,
                 -policy.rear_cornering_stiffness_nprad*rear_angle))
    moment=policy.front_axle_distance_m*front-policy.rear_axle_distance_m*rear
    rate=yaw_rate+(moment/policy.yaw_inertia_kg_m2)*dt
    if abs(rate)>policy.max_yaw_rate_radps:
        raise GroundedYawResolutionError("next yaw spin outside research domain")
    next_lateral=lateral+((front+rear)/policy.mass_kg-longitudinal*yaw_rate)*dt
    heading=Quaternion.from_axis_angle(_Y,rate*dt)
    next_forward=heading.rotate(forward)
    next_right=_Y.cross(next_forward)
    horizontal_velocity=next_forward*longitudinal+next_right*next_lateral
    position=Vec3(
        previous.position.x+horizontal_velocity.x*dt,
        prescribed.position.y,
        previous.position.z+horizontal_velocity.z*dt,
    )
    yaw=Quaternion.from_basis(next_right,_Y,next_forward)
    result=VehicleState(
        position,Vec3(horizontal_velocity.x,prescribed.velocity.y,horizontal_velocity.z),
        yaw,Vec3(0.,rate,0.),True,
    )
    return result,GroundedYawAudit(
        front_angle,rear_angle,front,rear,moment,lateral,next_lateral,
    )
