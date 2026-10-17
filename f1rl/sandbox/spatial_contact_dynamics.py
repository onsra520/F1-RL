"""Supported spatial rigid body with massless spring/contact references.

Position, quaternion, linear velocity and world angular momentum share one
force field. Tire forces use a friction circle and dissipative lateral slip;
there is no wheel-spin, static-friction, partial-support or flight claim.
Every accepted RK4 leaf must pass work/impulse and numerical body/road sweeps.
"""
from dataclasses import dataclass
from math import asin, atan2, cos, isfinite, sin, tanh

from .body_collision import VehicleBodyGeometry
from .center_trajectory import CenterSegment
from .coupled_longitudinal_dynamics import LongitudinalContactPolicy
from .dynamics.modifiers import MotionModifiers
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsResult,
    SuspensionDynamicsState, SuspensionResolutionError, WheelForceSample,
)
from .multi_surface_support import _atlas, _contacts, certify_multisurface_support
from .vehicle import VehicleControl, VehicleParameters, VehicleState

UP, RIGHT, FORWARD = Vec3(0.,1.,0.), Vec3(1.,0.,0.), Vec3(0.,0.,1.)


@dataclass(frozen=True, slots=True)
class SpatialContactPolicy:
    traction: LongitudinalContactPolicy
    yaw_inertia_kg_m2: float
    lateral_damping_nspm: float
    max_angular_speed_radps: float = 10.

    def __post_init__(self):
        if not isinstance(self.traction,LongitudinalContactPolicy):
            raise ValueError('explicit spatial traction policy required')
        if any(type(v) not in (int,float) or not isfinite(v) or v<0. for v in
               (self.yaw_inertia_kg_m2,self.lateral_damping_nspm,self.max_angular_speed_radps)):
            raise ValueError('finite nonnegative spatial contact parameters required')
        if self.yaw_inertia_kg_m2<=0. or not 0.<self.max_angular_speed_radps<=20.:
            raise ValueError('spatial inertia/angular domain must be positive and bounded')


@dataclass(frozen=True, slots=True)
class SpatialWheelAudit:
    wheel_index: int
    patch_id: str
    normal_force_n: float
    longitudinal_force_n: float
    lateral_force_n: float
    longitudinal_speed_mps: float
    lateral_speed_mps: float


@dataclass(frozen=True, slots=True)
class SpatialSubstepAudit:
    start_time_s: float
    end_time_s: float
    force_impulse_ns: Vec3
    torque_impulse_nms: Vec3
    linear_impulse_residual_ns: float
    angular_impulse_residual_nms: float
    initial_energy_j: float
    final_energy_j: float
    actuator_work_j: float
    dissipation_j: float
    energy_residual_j: float
    energy_tolerance_j: float
    contact_stages: tuple[tuple[SpatialWheelAudit, ...], ...]


@dataclass(frozen=True, slots=True)
class SpatialGroundAudit:
    version: str
    substeps: tuple[SpatialSubstepAudit, ...]
    actuator_work_j: float
    dissipation_j: float
    energy_residual_j: float
    max_linear_impulse_residual_ns: float
    max_angular_impulse_residual_nms: float


@dataclass(frozen=True, slots=True)
class SpatialGroundResult:
    suspension: SuspensionDynamicsResult
    audit: SpatialGroundAudit
    center_segments: tuple[CenterSegment, ...]


def _suspension_state(vehicle):
    forward = vehicle.orientation.rotate(FORWARD)
    right = vehicle.orientation.rotate(RIGHT)
    up = vehicle.orientation.rotate(UP)
    pitch = asin(max(-1.,min(1.,-forward.y)))
    yaw = atan2(forward.x,forward.z)
    roll = atan2(right.y,up.y)
    if abs(cos(pitch))<1e-6:
        raise SuspensionResolutionError('spatial suspension coordinates singular')
    rotation = Quaternion.from_axis_angle(UP,yaw)
    local_rate = Quaternion(rotation.w,-rotation.x,-rotation.y,-rotation.z).rotate(vehicle.angular_velocity)
    return SuspensionDynamicsState(vehicle.position.y,vehicle.velocity.y,pitch,local_rate.x,
                                   roll,local_rate.z/cos(pitch))


@dataclass(frozen=True, slots=True)
class _Stage:
    derivative: tuple[float, ...]
    force: Vec3
    torque: Vec3
    spring_energy: float
    actuator_power: float
    dissipation_power: float
    samples: tuple[WheelForceSample, ...]
    contacts: tuple[SpatialWheelAudit, ...]


def integrate_spatial_contact(previous, physics, body, surfaces, parameters, control,
                              policy, *, dt, modifiers=None):
    """Force-coupled supported motion; world Hdot=tau prevents hidden yaw projection.

    Quaternion RK4 is normalized at each accepted candidate. Work refinement
    controls a numerical defect, not the error against a calibrated physical ODE.
    Missing wheel support or an uncertain swept certificate rejects the step.
    """
    if (not isinstance(previous,VehicleState) or not isinstance(physics,IndependentSuspensionDynamicsPolicy)
            or physics.force_model!='road_relative_vertical_v2'
            or not isinstance(body,VehicleBodyGeometry) or not isinstance(parameters,VehicleParameters)
            or not isinstance(control,VehicleControl) or not isinstance(policy,SpatialContactPolicy)
            or type(dt) not in (int,float) or not isfinite(dt) or dt<=0.):
        raise ValueError('registered spatial contact inputs required')
    if not previous.grounded:
        raise SuspensionResolutionError('spatial contact requires four supported reference wheels')
    inertia=Vec3(physics.pitch_inertia_kg_m2,policy.yaw_inertia_kg_m2,physics.roll_inertia_kg_m2)
    if 2.*max(inertia.x,inertia.y,inertia.z)>inertia.x+inertia.y+inertia.z+1e-8:
        raise ValueError('spatial principal inertia violates rigid-body triangle inequality')
    atlas=_atlas(surfaces,physics.contact.seam_height_tolerance_m)
    if atlas is None:
        raise SuspensionResolutionError('spatial contact requires continuous 2.5D road atlas')
    patches={p.id:p for p in atlas[0]}
    modifiers=modifiers or MotionModifiers()
    if not isinstance(modifiers,MotionModifiers):
        raise ValueError('motion modifiers required')
    mass,gravity=physics.mass_kg,parameters.gravity
    preload=mass*gravity/4.
    traction=policy.traction

    def body_vector(q,world):
        return Quaternion(q.w,-q.x,-q.y,-q.z).rotate(world)

    def angular_momentum(q,omega):
        w=body_vector(q,omega)
        return q.rotate(Vec3(inertia.x*w.x,inertia.y*w.y,inertia.z*w.z))

    def vehicle(y):
        q=Quaternion(*y[6:10])
        h=body_vector(q,Vec3(*y[10:13]))
        omega=q.rotate(Vec3(h.x/inertia.x,h.y/inertia.y,h.z/inertia.z))
        return VehicleState(Vec3(*y[:3]),Vec3(*y[3:6]),q,omega,True)

    def stage(y):
        if not all(isfinite(v) for v in y):
            raise SuspensionResolutionError('nonfinite spatial state')
        state=vehicle(y)
        dynamic=_suspension_state(state)
        if (max(abs(dynamic.pitch_x_rad),abs(dynamic.roll_z_rad))>physics.max_abs_tilt_rad
                or state.speed>parameters.max_ground_speed+1e-8
                or state.angular_velocity.magnitude>policy.max_angular_speed_radps):
            raise SuspensionResolutionError('spatial state exceeds tilt/speed/angular domain')
        contacts=_contacts(state,physics.contact,atlas)
        if contacts is None:
            raise SuspensionResolutionError('spatial wheel support/travel missing')
        drag=state.velocity*(-mass*parameters.rolling_drag*modifiers.rolling_drag_multiplier)
        boost=state.orientation.rotate(FORWARD)*(mass*modifiers.longitudinal_acceleration_bonus)
        force=UP*(-mass*gravity)+drag+boost
        torque=state.angular_velocity*(-physics.angular_damping_nmsprad)
        actuator=boost.dot(state.velocity)
        dissipation=-drag.dot(state.velocity)+physics.angular_damping_nmsprad*state.angular_velocity.dot(state.angular_velocity)
        spring_energy=0.
        samples,records=[],[]
        for index,(local,contact) in enumerate(zip(physics.contact.wheels.positions,contacts,strict=True)):
            patch=patches[contact.patch_id]
            arm=state.orientation.rotate(local)
            point_velocity=state.velocity+state.angular_velocity.cross(arm)
            gradient=patch.normal/patch.normal.y
            gap_rate=gradient.dot(point_velocity)
            raw_elastic=preload+physics.spring_stiffness_npm*contact.deflection_m
            elastic=max(0.,raw_elastic)
            load=max(0.,raw_elastic-physics.damper_nspm*gap_rate)
            normal_force=load/patch.normal.y
            angle=control.steering*parameters.max_steering_angle if local.z>0. else 0.
            heading=state.orientation.rotate(Vec3(sin(angle),0.,cos(angle))).reject_from(patch.normal)
            if heading.magnitude<1e-8:
                raise SuspensionResolutionError('wheel tangent direction unresolved')
            tangent=heading.normalized()
            lateral=patch.normal.cross(tangent)
            speed,slip=point_velocity.dot(tangent),point_velocity.dot(lateral)
            longitudinal=mass*(control.throttle*parameters.max_acceleration
                -control.brake*parameters.max_braking*tanh(speed/traction.brake_speed_scale_mps))/4.
            side=-policy.lateral_damping_nspm*slip
            magnitude=(longitudinal*longitudinal+side*side)**.5
            limit=traction.traction_mu*normal_force
            if magnitude>limit:
                scale=limit/magnitude
                longitudinal,side=longitudinal*scale,side*scale
            if longitudinal*speed>traction.max_drive_power_w/4.:
                longitudinal=traction.max_drive_power_w/(4.*speed)
            wheel_force=gradient*load+tangent*longitudinal+lateral*side
            force=force+wheel_force
            torque=torque+arm.cross(wheel_force)
            actuator+=longitudinal*speed
            normal_loss=(elastic-load)*gap_rate
            if normal_loss < -1e-8 or side*slip>1e-8:
                raise SuspensionResolutionError('spatial contact produced damping energy')
            dissipation+=max(0.,normal_loss)-side*slip
            spring_energy+=elastic*elastic/(2.*physics.spring_stiffness_npm)
            samples.append(WheelForceSample(contact,load,gap_rate))
            records.append(SpatialWheelAudit(index,patch.id,normal_force,longitudinal,side,speed,slip))
        if not any(s.upward_force_n>0. for s in samples):
            raise SuspensionResolutionError('unloaded spatial contact requires airborne solver')
        qw,qx,qy,qz=y[6:10]
        wx,wy,wz=state.angular_velocity.x,state.angular_velocity.y,state.angular_velocity.z
        qdot=(-.5*(wx*qx+wy*qy+wz*qz),.5*(wx*qw+wy*qz-wz*qy),
              .5*(-wx*qz+wy*qw+wz*qx),.5*(wx*qy-wy*qx+wz*qw))
        derivative=(state.velocity.x,state.velocity.y,state.velocity.z,
                    force.x/mass,force.y/mass,force.z/mass,*qdot,torque.x,torque.y,torque.z)
        return _Stage(derivative,force,torque,spring_energy,actuator,dissipation,tuple(samples),tuple(records))

    def kinetic(y):
        s=vehicle(y)
        return .5*mass*s.velocity.dot(s.velocity)+.5*s.angular_velocity.dot(Vec3(*y[10:13]))

    def energy(y,f):
        return kinetic(y)+mass*gravity*y[1]+f.spring_energy

    q=previous.orientation
    momentum=angular_momentum(q,previous.angular_velocity)
    state=(previous.position.x,previous.position.y,previous.position.z,
           previous.velocity.x,previous.velocity.y,previous.velocity.z,
           q.w,q.x,q.y,q.z,momentum.x,momentum.y,momentum.z)
    poses,history,leaves,times=[previous],[],[],[0.]

    def shifted(y,d,h):
        return tuple(v+h*k for v,k in zip(y,d,strict=True))

    def advance(y,start,h,depth):
        a=stage(y)
        b=stage(shifted(y,a.derivative,h/2.))
        c=stage(shifted(y,b.derivative,h/2.))
        d=stage(shifted(y,c.derivative,h))
        stages=(a,b,c,d)
        weights=(1.,2.,2.,1.)
        end=tuple(v+h/6.*(ka+2.*kb+2.*kc+kd)
                  for v,ka,kb,kc,kd in zip(y,a.derivative,b.derivative,c.derivative,d.derivative,strict=True))
        q=Quaternion(*end[6:10])
        end=(*end[:6],q.w,q.x,q.y,q.z,*end[10:13])
        final=stage(end)
        work=h/6.*sum(w*f.actuator_power for w,f in zip(weights,stages,strict=True))
        lost=h/6.*sum(w*f.dissipation_power for w,f in zip(weights,stages,strict=True))
        e0,e1=energy(y,a),energy(end,final)
        residual=e1-e0-work+lost
        scale=max(1.,kinetic(y),kinetic(end),a.spring_energy,final.spring_energy,
                  abs(mass*gravity*(end[1]-y[1])),abs(work),lost)
        tolerance=traction.energy_absolute_tolerance_j+traction.energy_relative_tolerance*scale
        if abs(residual)>tolerance:
            if depth>=traction.max_refinements:
                raise SuspensionResolutionError('spatial energy balance unresolved at refinement limit')
            middle=advance(y,start,h/2.,depth+1)
            return advance(middle,times[-1],h/2.,depth+1)
        first,last=vehicle(y),vehicle(end)
        verdict=certify_multisurface_support(first,last,body,physics.contact,surfaces,
                                            body_clearance_version='bounded_patch_planes_v2')
        if not verdict.valid:
            raise SuspensionResolutionError(f'spatial body/contact sweep {verdict.status}: {verdict.reason}')
        impulse=sum((f.force*w for w,f in zip(weights,stages,strict=True)),Vec3(0.,0.,0.))*(h/6.)
        torque_impulse=sum((f.torque*w for w,f in zip(weights,stages,strict=True)),Vec3(0.,0.,0.))*(h/6.)
        linear_error=((last.velocity-first.velocity)*mass-impulse).magnitude
        angular_error=(Vec3(*end[10:13])-Vec3(*y[10:13])-torque_impulse).magnitude
        if linear_error>1e-7 or angular_error>1e-7:
            raise SuspensionResolutionError('spatial momentum audit failed')
        if len(leaves)>=4096:
            raise SuspensionResolutionError('spatial substep budget exhausted')
        leaves.append(SpatialSubstepAudit(start,start+h,impulse,torque_impulse,linear_error,angular_error,
            e0,e1,work,lost,residual,tolerance,tuple(f.contacts for f in stages)))
        poses.append(last)
        history.append(a.samples)
        times.append(start+h)
        return end

    h=dt/physics.integration_substeps
    for _ in range(physics.integration_substeps):
        state=advance(state,times[-1],h,0)
    last=poses[-1]
    suspension=SuspensionDynamicsResult(last,_suspension_state(last),tuple(poses),tuple(history))
    audit=SpatialGroundAudit('spatial_contact_world_momentum_rk4_v1',tuple(leaves),
        sum(s.actuator_work_j for s in leaves),sum(s.dissipation_j for s in leaves),
        sum(s.energy_residual_j for s in leaves),max(s.linear_impulse_residual_ns for s in leaves),
        max(s.angular_impulse_residual_nms for s in leaves))
    centers=tuple(CenterSegment(a,b,p.position,r.position) for a,b,p,r in zip(times,times[1:],poses,poses[1:]))
    return SpatialGroundResult(suspension,audit,centers)
