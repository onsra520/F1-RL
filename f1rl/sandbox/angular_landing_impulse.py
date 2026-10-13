"""Prospective unilateral *single-wheel* normal-impact impulse (WP7 research).

This is an auditable impulsive rigid-body velocity update at a prescribed
wheel-reference point. A real multi-wheel contact manifold, tire radius,
friction and body restitution are NOT solved. Any ambiguous co-contact must
fail closed in the integrating flight kernel, never be treated as this solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .geometry.vector import Vec3
from .geometry.rotation import Quaternion
from .independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, _horizontal_yaw,
)
from .vehicle import VehicleState

_EPS = 1e-9
_X = Vec3(1., 0., 0.)
_Y = Vec3(0., 1., 0.)
_Z = Vec3(0., 0., 1.)


class ImpulseResolutionError(RuntimeError):
    """Impact outcome unsupported/unsafe, never an earned collision or reward."""


@dataclass(frozen=True, slots=True)
class AngularLandingImpulsePolicy:
    """All numeric knobs are research parameters, not science acceptance."""
    restitution: float = 0.
    max_pre_angular_speed_radps: float = 2.
    max_post_angular_speed_radps: float = 2.
    event_bracket_segments: int = 64
    event_bisections: int = 32
    event_time_tolerance_s: float = 1e-8
    unique_contact_gap_m: float = 1e-5
    max_ground_yaw_rate_radps: float = .02

    def __post_init__(self) -> None:
        for name in ("restitution", "max_pre_angular_speed_radps",
                     "max_post_angular_speed_radps", "event_time_tolerance_s",
                     "unique_contact_gap_m", "max_ground_yaw_rate_radps"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not isfinite(v):
                raise ValueError(f"{name} must be finite")
            if name == "restitution":
                if v < 0. or v > 1.:
                    raise ValueError("restitution must lie within [0,1]")
            elif v <= 0.:
                raise ValueError(f"{name} must be positive")
        if (type(self.event_bracket_segments) is not int
                or not 8 <= self.event_bracket_segments <= 1024):
            raise ValueError("event_bracket_segments must be [8,1024]")
        if (type(self.event_bisections) is not int
                or not 8 <= self.event_bisections <= 64):
            raise ValueError("event_bisections must be [8,64]")


@dataclass(frozen=True, slots=True)
class WheelImpulseAudit:
    wheel_index: int
    patch_id: str
    magnitude_ns: float
    normal_speed_before_mps: float
    normal_speed_after_mps: float
    pre_kinetic_energy_j: float
    post_kinetic_energy_j: float
    linear_delta_mps: Vec3
    angular_delta_radps: Vec3

    def __post_init__(self) -> None:
        if type(self.wheel_index) is not int or not 0 <= self.wheel_index < 4:
            raise ValueError("invalid wheel index")
        if (not self.patch_id or not all(isfinite(x) for x in (
                self.magnitude_ns, self.normal_speed_before_mps,
                self.normal_speed_after_mps, self.pre_kinetic_energy_j,
                self.post_kinetic_energy_j))):
            raise ValueError("invalid wheel impulse audit")


def _yaw_inertia_inverse(vector: Vec3, yaw: Quaternion,
                         physics: IndependentSuspensionDynamicsPolicy) -> Vec3:
    """Inverse world inertia of a *research-restricted X/Z-only* impulse.

    The 3-DOF chassis model has no independently registered yaw moment of
    inertia. A nonzero yaw torque is therefore unsupported, not silently
    inverted with a guessed parameter. A yaw-frame vector is transformed by
    the inverse yaw quaternion; its X/Z components use registered I_pitch and
    I_roll, then the result returns to world space.
    """
    local = Quaternion(yaw.w, -yaw.x, -yaw.y, -yaw.z).rotate(vector)
    if abs(local.y) > 1e-7:
        raise ImpulseResolutionError("yaw torque requires registered yaw inertia")
    return yaw.rotate(Vec3(local.x / physics.pitch_inertia_kg_m2,
                           0., local.z / physics.roll_inertia_kg_m2))


def _kinetic(v: Vec3, w: Vec3, yaw: Quaternion,
             physics: IndependentSuspensionDynamicsPolicy) -> float:
    local = Quaternion(yaw.w, -yaw.x, -yaw.y, -yaw.z).rotate(w)
    if abs(local.y) > 1e-7:
        raise ImpulseResolutionError("yaw spin requires registered yaw inertia")
    return .5*(
        physics.mass_kg*v.dot(v)
        + physics.pitch_inertia_kg_m2*local.x**2
        + physics.roll_inertia_kg_m2*local.z**2
    )


def solve_one_wheel_normal_impulse(
    airborne: VehicleState,
    wheel_local: Vec3,
    wheel_index: int,
    patch_id: str,
    normal: Vec3,
    physics: IndependentSuspensionDynamicsPolicy,
    policy: AngularLandingImpulsePolicy,
) -> tuple[VehicleState, WheelImpulseAudit]:
    """Apply frictionless normal J at an isolated closing wheel reference.

    J = -(1+e) v_contact_n / [1/m + (r x n) * I^-1(r x n)].

    Uses the existing yaw-aligned pitch/roll inertia only, and rejects any
    yaw torque/spin. Velocity update conserves tangential *impulse* (zero
    tangential force), and decreases kinetic energy for restitution in [0,1].
    """
    if (not isinstance(airborne, VehicleState) or airborne.grounded
            or not isinstance(wheel_local, Vec3)
            or not isinstance(physics, IndependentSuspensionDynamicsPolicy)
            or not isinstance(policy, AngularLandingImpulsePolicy)
            or not isinstance(normal, Vec3) or type(wheel_index) is not int
            or not 0 <= wheel_index < 4 or not patch_id):
        raise ValueError("explicit airborne wheel/physical impact inputs required")
    if abs(normal.magnitude-1.) > 1e-8:
        raise ValueError("normal must be unit length")
    if normal.y < 1.-1e-9:
        raise ImpulseResolutionError("sloped angular impulse needs yaw inertia")
    yaw = _horizontal_yaw(airborne.orientation)
    w = airborne.angular_velocity
    if w.magnitude > policy.max_pre_angular_speed_radps:
        raise ImpulseResolutionError("pre-impact angular speed exceeds configured bound")
    r = airborne.orientation.rotate(wheel_local)
    torque_axis = r.cross(normal)
    inverse = _yaw_inertia_inverse(torque_axis,yaw,physics)
    k = 1./physics.mass_kg + torque_axis.dot(inverse)
    if k <= 0. or not isfinite(k):
        raise ImpulseResolutionError("invalid impulse effective inverse mass")
    vn = (airborne.velocity + w.cross(r)).dot(normal)
    if vn >= -_EPS:
        raise ImpulseResolutionError("wheel is not approaching road")
    energy_before = _kinetic(airborne.velocity,w,yaw,physics)
    J = -(1.+policy.restitution)*vn/k
    if J <= 0. or not isfinite(J):
        raise ImpulseResolutionError("invalid unilateral normal impulse")
    delta_v = normal*(J/physics.mass_kg)
    delta_w = inverse*J
    after_v = airborne.velocity+delta_v
    after_w = w+delta_w
    if after_w.magnitude > policy.max_post_angular_speed_radps:
        raise ImpulseResolutionError("post-impact spin outside registered bound")
    vn_after = (after_v + after_w.cross(r)).dot(normal)
    if abs(vn_after + policy.restitution*vn) > 1e-7:
        raise ImpulseResolutionError("normal restitution identity failed")
    energy_after = _kinetic(after_v,after_w,yaw,physics)
    if energy_after > energy_before + 1e-6*max(1.,energy_before):
        raise ImpulseResolutionError("impulse increased kinetic energy")
    impact = VehicleState(
        airborne.position,after_v,airborne.orientation,after_w,False,
    )
    audit = WheelImpulseAudit(
        wheel_index,patch_id,J,vn,vn_after,
        energy_before,energy_after,delta_v,delta_w,
    )
    return impact,audit
