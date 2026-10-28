"""Opt-in WP7 research: deterministic four-wheel contact impulses with full yaw inertia.

The constrained projected sequential impulse solver uses prescribed wheel
reference points, a diagonal body-frame inertia tensor and normal/Coulomb
friction impulses. It rejects non-convergence, energy gain, invalid geometry,
and unsupported downstream transitions rather than fabricating a collision.

NOT a measured tire patch, exact complementarity solution, or accepted
scientific calibration. No dependencies on learning, TRAIN or HELD-OUT.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from .geometry.vector import Vec3
from .geometry.rotation import Quaternion
from .independent_suspension_dynamics import IndependentSuspensionDynamicsPolicy
from .vehicle import VehicleState

_ZERO = Vec3(0., 0., 0.)
_UP = Vec3(0., 1., 0.)


class ManifoldResolutionError(RuntimeError):
    """Unknown/unsafe contact manifold; never an episode success."""


@dataclass(frozen=True, slots=True)
class ContactManifoldPolicy:
    yaw_inertia_kg_m2: float
    friction_coefficient: float = 0.
    restitution: float = 0.
    sweeps: int = 128
    velocity_tolerance_mps: float = 1e-5
    energy_tolerance_j: float = 1e-5
    simultaneous_gap_m: float = 1e-4
    rebound_separation_mps: float = 1e-3
    max_post_angular_speed_radps: float = 5.
    max_ground_yaw_radps: float = .02
    max_impacts_per_step: int = 1
    coplanar_flat_contacts: bool = False

    def __post_init__(self) -> None:
        if type(self.coplanar_flat_contacts) is not bool:
            raise ValueError("coplanar_flat_contacts must be bool")
        for name in ("yaw_inertia_kg_m2", "friction_coefficient", "restitution",
                     "velocity_tolerance_mps", "energy_tolerance_j",
                     "simultaneous_gap_m", "rebound_separation_mps",
                     "max_post_angular_speed_radps", "max_ground_yaw_radps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in ("friction_coefficient", "restitution"):
                if value < 0. or value > (1. if name == "restitution" else 2.):
                    raise ValueError(f"{name} outside registered bounds")
            elif value <= 0.:
                raise ValueError(f"{name} must be positive")
        if type(self.sweeps) is not int or not 1 <= self.sweeps <= 1024:
            raise ValueError("sweeps must be integer in [1,1024]")
        if type(self.max_impacts_per_step) is not int or not 1 <= self.max_impacts_per_step <= 8:
            raise ValueError("max_impacts_per_step must be integer in [1,8]")


@dataclass(frozen=True, slots=True)
class WheelManifoldContact:
    wheel_index: int
    patch_id: str
    wheel_local: Vec3
    normal: Vec3

    def __post_init__(self) -> None:
        if (type(self.wheel_index) is not int or not 0 <= self.wheel_index < 4
                or not self.patch_id or not isinstance(self.wheel_local, Vec3)
                or not isinstance(self.normal, Vec3)
                or abs(self.normal.magnitude - 1.) > 1e-8
                or self.normal.y < .5):
            raise ValueError("invalid registered wheel plane contact")


@dataclass(frozen=True, slots=True)
class ManifoldContactAudit:
    wheel_index: int
    normal_impulse_ns: float
    tangential_impulse_ns: Vec3
    incoming_normal_speed_mps: float
    outgoing_normal_speed_mps: float
    patch_id: str = ""


@dataclass(frozen=True, slots=True)
class ManifoldImpulseAudit:
    contacts: tuple[ManifoldContactAudit, ...]
    pre_kinetic_energy_j: float
    post_kinetic_energy_j: float
    total_linear_impulse_ns: Vec3
    total_angular_impulse_nms: Vec3
    iterations: int


def _world_inverse_inertia(torque: Vec3, q: Quaternion,
                           phys: IndependentSuspensionDynamicsPolicy,
                           policy: ContactManifoldPolicy) -> Vec3:
    inv = Quaternion(q.w, -q.x, -q.y, -q.z)
    local = inv.rotate(torque)
    return q.rotate(Vec3(
        local.x/phys.pitch_inertia_kg_m2,
        local.y/policy.yaw_inertia_kg_m2,
        local.z/phys.roll_inertia_kg_m2,
    ))


def _energy(v: Vec3, w: Vec3, q: Quaternion,
            phys: IndependentSuspensionDynamicsPolicy,
            policy: ContactManifoldPolicy) -> float:
    inv = Quaternion(q.w, -q.x, -q.y, -q.z)
    local = inv.rotate(w)
    return .5*(phys.mass_kg*v.dot(v)
               + phys.pitch_inertia_kg_m2*local.x**2
               + policy.yaw_inertia_kg_m2*local.y**2
               + phys.roll_inertia_kg_m2*local.z**2)


def solve_wheel_contact_manifold(
    airborne: VehicleState,
    contacts: tuple[WheelManifoldContact, ...],
    physics: IndependentSuspensionDynamicsPolicy,
    policy: ContactManifoldPolicy,
) -> tuple[VehicleState, ManifoldImpulseAudit]:
    """PGS normal complementarity plus clamped isotropic Coulomb friction.

    For each accumulated normal impulse jn>=0, solve the target incoming
    normal restitution and clamp total tangent impulse to ||jt||<=mu*jn.
    Reject if residuals, unilateral condition or energy audits fail.
    """
    if (not isinstance(airborne, VehicleState) or airborne.grounded
            or not isinstance(physics, IndependentSuspensionDynamicsPolicy)
            or not isinstance(policy, ContactManifoldPolicy)
            or not isinstance(contacts, tuple) or not 1 <= len(contacts) <= 4
            or any(not isinstance(c, WheelManifoldContact) for c in contacts)):
        raise ValueError("requires opt-in airborne state and 1-4 wheel contacts")
    if len({c.wheel_index for c in contacts}) != len(contacts):
        raise ValueError("duplicate wheel contact")
    # Solve in canonical order; caller order must not change physics output.
    sorted_contacts = tuple(sorted(contacts,key=lambda c:(c.wheel_index,c.patch_id)))
    if any(c.wheel_local != physics.contact.wheels.positions[c.wheel_index]
           for c in sorted_contacts):
        raise ValueError("contact wheel geometry differs from physics registration")
    q = airborne.orientation
    r = tuple(q.rotate(c.wheel_local) for c in sorted_contacts)
    vn_initial = tuple(
        (airborne.velocity + airborne.angular_velocity.cross(arm)).dot(c.normal)
        for c,arm in zip(sorted_contacts,r,strict=True)
    )
    if not any(v < -policy.velocity_tolerance_mps for v in vn_initial):
        raise ManifoldResolutionError("no approaching manifold wheel")
    target = tuple(
        -policy.restitution*min(0.,v) if v < -policy.velocity_tolerance_mps else 0.
        for v in vn_initial
    )
    pre = _energy(airborne.velocity,airborne.angular_velocity,q,physics,policy)
    v,w = airborne.velocity,airborne.angular_velocity
    normal_acc = [0.]*len(sorted_contacts)
    tangent_acc = [_ZERO]*len(sorted_contacts)

    def contact_velocity(index):
        return v + w.cross(r[index])

    def inv_k(arm, direction):
        axis=arm.cross(direction)
        value=1./physics.mass_kg+axis.dot(_world_inverse_inertia(axis,q,physics,policy))
        if not isfinite(value) or value<=0.:
            raise ManifoldResolutionError("invalid contact effective mass")
        return value

    def apply(arm, impulse):
        nonlocal v,w
        v = v + impulse/physics.mass_kg
        w = w + _world_inverse_inertia(arm.cross(impulse),q,physics,policy)

    niter=0
    for iteration in range(policy.sweeps):
        niter=iteration+1
        for i,c in enumerate(sorted_contacts):
            normal=c.normal
            closing=contact_velocity(i).dot(normal)
            delta=max(-normal_acc[i],(target[i]-closing)/inv_k(r[i],normal))
            if delta:
                normal_acc[i]+=delta
                apply(r[i],normal*delta)
            if policy.friction_coefficient>0. and normal_acc[i]>0.:
                # Incremental tangent solver with a circular Coulomb cone.
                vt=contact_velocity(i).reject_from(normal)
                speed=vt.magnitude
                if speed>1e-13:
                    direction=vt/speed
                    trial=tangent_acc[i] - direction*(speed/inv_k(r[i],direction))
                    max_friction=policy.friction_coefficient*normal_acc[i]
                    if trial.magnitude>max_friction:
                        trial=trial*(max_friction/trial.magnitude)
                    tangential_delta=trial-tangent_acc[i]
                    tangent_acc[i]=trial
                    apply(r[i],tangential_delta)

    post=_energy(v,w,q,physics,policy)
    if post>pre+policy.energy_tolerance_j:
        raise ManifoldResolutionError("manifold impulse increased kinetic energy")
    if w.magnitude>policy.max_post_angular_speed_radps:
        raise ManifoldResolutionError("post-impulse angular speed exceeds registered limit")
    audit_contacts=[]
    total_force=_ZERO
    total_torque=_ZERO
    for i,c in enumerate(sorted_contacts):
        vn=contact_velocity(i).dot(c.normal)
        if vn+policy.velocity_tolerance_mps < target[i]:
            raise ManifoldResolutionError("normal restitution/complementarity not converged")
        if normal_acc[i]<-1e-9:
            raise ManifoldResolutionError("negative unilateral impulse")
        if tangent_acc[i].magnitude > policy.friction_coefficient*normal_acc[i]+1e-7:
            raise ManifoldResolutionError("Coulomb friction cone violated")
        impulse=c.normal*normal_acc[i]+tangent_acc[i]
        total_force=total_force+impulse
        total_torque=total_torque+r[i].cross(impulse)
        audit_contacts.append(ManifoldContactAudit(
            c.wheel_index, normal_acc[i], tangent_acc[i], vn_initial[i],vn,c.patch_id,
        ))
    if (v-airborne.velocity-total_force/physics.mass_kg).magnitude > 1e-7:
        raise ManifoldResolutionError("linear momentum audit failed")
    expected_w=_world_inverse_inertia(total_torque,q,physics,policy)
    if (w-airborne.angular_velocity-expected_w).magnitude > 1e-7:
        raise ManifoldResolutionError("angular momentum audit failed")
    result=VehicleState(airborne.position,v,q,w,False)
    audit=ManifoldImpulseAudit(
        tuple(audit_contacts),pre,post,total_force,total_torque,niter,
    )
    return result,audit
