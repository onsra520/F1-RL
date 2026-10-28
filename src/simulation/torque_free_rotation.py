"""Prospective torque-free rigid-body orientation evolution with diagonal inertia.

This isolated 6-DOF angular kernel integrates Euler's equations in the BODY
frame using fixed-step RK4 and a normalized quaternion. Translation, gravity,
contact forces and collision are separate. It does not authorize a scientific
physics freeze or replace prescribed ground steering/yaw.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from .geometry.rotation import Quaternion
from .geometry.vector import Vec3

_EPS=1e-12


class RigidBodyIntegrationError(RuntimeError):
    """Angular physics certification unavailable; no successful trajectory."""


@dataclass(frozen=True,slots=True)
class TorqueFreeRotationPolicy:
    pitch_inertia_kg_m2: float
    yaw_inertia_kg_m2: float
    roll_inertia_kg_m2: float
    integration_steps: int=32
    max_angular_speed_radps: float=10.
    energy_relative_tolerance: float=1e-5
    momentum_relative_tolerance: float=1e-5

    def __post_init__(self):
        for name in (
            "pitch_inertia_kg_m2","yaw_inertia_kg_m2","roll_inertia_kg_m2",
            "max_angular_speed_radps","energy_relative_tolerance",
            "momentum_relative_tolerance",
        ):
            val=getattr(self,name)
            if type(val) not in (int,float) or not isfinite(val) or val<=0.:
                raise ValueError(f"{name} must be finite positive")
        if type(self.integration_steps) is not int or not 1<=self.integration_steps<=1024:
            raise ValueError("integration_steps must be [1,1024]")


@dataclass(frozen=True,slots=True)
class AngularRigidBodyState:
    orientation: Quaternion
    world_angular_velocity_radps: Vec3


@dataclass(frozen=True,slots=True)
class AngularRigidBodyAudit:
    initial_energy_j: float
    final_energy_j: float
    initial_world_angular_momentum: Vec3
    final_world_angular_momentum: Vec3
    steps: int


def _quat_from_tuple(q):
    return Quaternion(q[0],q[1],q[2],q[3])


def _body_omega(q,world):
    conjugate=Quaternion(q.w,-q.x,-q.y,-q.z)
    return conjugate.rotate(world)


def _energy(w,policy):
    return .5*(
        policy.pitch_inertia_kg_m2*w.x*w.x
        +policy.yaw_inertia_kg_m2*w.y*w.y
        +policy.roll_inertia_kg_m2*w.z*w.z
    )


def _angular_momentum(q,w,policy):
    return q.rotate(Vec3(
        policy.pitch_inertia_kg_m2*w.x,
        policy.yaw_inertia_kg_m2*w.y,
        policy.roll_inertia_kg_m2*w.z,
    ))


def advance_torque_free_rotation(
    state: AngularRigidBodyState,
    policy: TorqueFreeRotationPolicy,
    duration_s: float,
)->tuple[AngularRigidBodyState,AngularRigidBodyAudit]:
    """RK4 for body angular velocity and quaternion (qdot=0.5*q*[0,w]).

    At every substep enforce positive finite quaternion norm and registered
    spin magnitude; at the end verify energy and world angular-momentum drift.
    No damping/aero/contact torque is implied.
    """
    if not isinstance(state,AngularRigidBodyState) or not isinstance(policy,TorqueFreeRotationPolicy):
        raise ValueError("registered rigid-body state and policy required")
    if type(duration_s) not in (int,float) or not isfinite(duration_s) or duration_s<0.:
        raise ValueError("finite nonnegative duration required")
    q=state.orientation
    w=_body_omega(q,state.world_angular_velocity_radps)
    if w.magnitude>policy.max_angular_speed_radps:
        raise RigidBodyIntegrationError("initial angular speed outside physics policy")
    initial_energy=_energy(w,policy)
    initial_momentum=_angular_momentum(q,w,policy)
    iq=(q.w,q.x,q.y,q.z)
    iv=(w.x,w.y,w.z)
    ix,iy,iz=(policy.pitch_inertia_kg_m2,policy.yaw_inertia_kg_m2,policy.roll_inertia_kg_m2)
    def deriv(vec):
        qw,qx,qy,qz,wx,wy,wz=vec
        return (
            -.5*(qx*wx+qy*wy+qz*wz),
            .5*(qw*wx+qy*wz-qz*wy),
            .5*(qw*wy+qz*wx-qx*wz),
            .5*(qw*wz+qx*wy-qy*wx),
            (iy-iz)*wy*wz/ix,
            (iz-ix)*wz*wx/iy,
            (ix-iy)*wx*wy/iz,
        )
    y=(*iq,*iv)
    h=duration_s/policy.integration_steps
    for _ in range(policy.integration_steps):
        a=deriv(y)
        b=deriv(tuple(v+h*k*.5 for v,k in zip(y,a,strict=True)))
        c=deriv(tuple(v+h*k*.5 for v,k in zip(y,b,strict=True)))
        d=deriv(tuple(v+h*k for v,k in zip(y,c,strict=True)))
        y=tuple(v+h*(ka+2*kb+2*kc+kd)/6.
                for v,ka,kb,kc,kd in zip(y,a,b,c,d,strict=True))
        if not all(isfinite(z) for z in y):
            raise RigidBodyIntegrationError("angular integrator produced nonfinite state")
        norm=sqrt(sum(x*x for x in y[:4]))
        if norm<=_EPS:
            raise RigidBodyIntegrationError("angular quaternion singularity")
        y=(*(v/norm for v in y[:4]),*y[4:])
        if Vec3(*y[4:]).magnitude>policy.max_angular_speed_radps:
            raise RigidBodyIntegrationError("angular speed exceeds physics policy")
    last_q=_quat_from_tuple(y[:4])
    last_w=Vec3(*y[4:])
    final_energy=_energy(last_w,policy)
    final_momentum=_angular_momentum(last_q,last_w,policy)
    if abs(final_energy-initial_energy)>policy.energy_relative_tolerance*max(1.,initial_energy):
        raise RigidBodyIntegrationError("torque-free energy conservation failed")
    if (final_momentum-initial_momentum).magnitude>policy.momentum_relative_tolerance*max(1.,initial_momentum.magnitude):
        raise RigidBodyIntegrationError("torque-free world angular momentum conservation failed")
    result=AngularRigidBodyState(last_q,last_q.rotate(last_w))
    return result,AngularRigidBodyAudit(
        initial_energy,final_energy,initial_momentum,final_momentum,
        policy.integration_steps,
    )
