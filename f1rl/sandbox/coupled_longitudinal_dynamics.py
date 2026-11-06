"""Coupled world-Z/heave/pitch contact dynamics with audited RK4 work balance.

The planar rigid body has exact translational and principal-pitch inertia.
Four massless suspension reference contacts provide conservative normal spring
forces and dissipative damping. Tangential actuation obeys explicit friction/
power bounds; regularized braking does not claim static friction. Steering,
roll, lateral motion and airborne recovery remain unsupported.
"""
from dataclasses import dataclass
from math import isfinite, tanh

from .body_collision import VehicleBodyGeometry
from .center_trajectory import CenterSegment
from .dynamics.modifiers import MotionModifiers
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsResult, SuspensionDynamicsState,
    SuspensionResolutionError, WheelForceSample,
)
from .multi_surface_support import _atlas, _contacts, certify_multisurface_support
from .vehicle import VehicleControl, VehicleParameters, VehicleState

UP, FORWARD, RIGHT = Vec3(0.,1.,0.), Vec3(0.,0.,1.), Vec3(1.,0.,0.)


@dataclass(frozen=True, slots=True)
class LongitudinalContactPolicy:
    traction_mu: float
    max_drive_power_w: float
    brake_speed_scale_mps: float = .05
    energy_relative_tolerance: float = 1e-8
    energy_absolute_tolerance_j: float = 1e-7
    # Damper force jumps at a continuous-height road-normal seam; isolate
    # that interval without weakening the numerical work tolerance.
    max_refinements: int = 20

    def __post_init__(self):
        for name in ('traction_mu','max_drive_power_w','brake_speed_scale_mps',
                     'energy_relative_tolerance','energy_absolute_tolerance_j'):
            value = getattr(self,name)
            if type(value) not in (int,float) or not isfinite(value) or value < 0.:
                raise ValueError('finite nonnegative longitudinal contact parameters required')
        if (self.max_drive_power_w <= 0. or not 0. < self.brake_speed_scale_mps <= 1.
                or not 0. < self.energy_relative_tolerance <= 1e-5
                or not 0. < self.energy_absolute_tolerance_j <= 1e-3
                or type(self.max_refinements) is not int or not 0 <= self.max_refinements <= 24):
            raise ValueError('longitudinal contact parameters exceed numerical domain')


@dataclass(frozen=True, slots=True)
class ContactForceAudit:
    wheel_index: int
    patch_id: str
    normal_force_n: float
    tangent_force_n: float
    tangent_speed_mps: float


@dataclass(frozen=True, slots=True)
class LongitudinalSubstepAudit:
    start_time_s: float
    end_time_s: float
    force_impulse_ns: Vec3
    pitch_impulse_nms: float
    linear_impulse_residual_ns: float
    angular_impulse_residual_nms: float
    initial_energy_j: float
    final_energy_j: float
    actuator_work_j: float
    dissipation_j: float
    energy_residual_j: float
    energy_tolerance_j: float
    contact_stages: tuple[tuple[ContactForceAudit, ...], ...]


@dataclass(frozen=True, slots=True)
class CoupledGroundAudit:
    version: str
    substeps: tuple[LongitudinalSubstepAudit, ...]
    actuator_work_j: float
    dissipation_j: float
    energy_residual_j: float
    max_linear_impulse_residual_ns: float
    max_angular_impulse_residual_nms: float


@dataclass(frozen=True, slots=True)
class CoupledGroundResult:
    suspension: SuspensionDynamicsResult
    audit: CoupledGroundAudit
    center_segments: tuple[CenterSegment, ...]


def _longitudinal_frame(vehicle, initial):
    if not isinstance(vehicle,VehicleState) or not isinstance(initial,SuspensionDynamicsState):
        raise ValueError('vehicle and persistent suspension state required')
    q = Quaternion.from_axis_angle(RIGHT,initial.pitch_x_rad)
    actual = vehicle.orientation
    # q and -q encode the same attitude; a dot-product tolerance alone
    # hides first-order orientation errors because 1-cos(angle) is quadratic.
    attitude_error_squared = min(sum((a-sign*b)**2 for a,b in zip(
        (q.w,q.x,q.y,q.z),(actual.w,actual.x,actual.y,actual.z))) for sign in (1.,-1.))
    if (not vehicle.grounded or abs(initial.roll_z_rad)>1e-10 or abs(initial.roll_rate_radps)>1e-10
            or abs(vehicle.velocity.x)>1e-10 or abs(vehicle.angular_velocity.y)>1e-10
            or abs(vehicle.angular_velocity.z)>1e-10 or attitude_error_squared>1e-20
            or abs(vehicle.position.y-initial.height_m)>1e-8
            or abs(vehicle.velocity.y-initial.heave_velocity_mps)>1e-8
            or abs(vehicle.angular_velocity.x-initial.pitch_rate_radps)>1e-8):
        raise SuspensionResolutionError('coupled contact requires consistent world-Z/heave/pitch state')
    # Internal coordinates are (world z, height, pitch), not a spatial vector.
    return Vec3(vehicle.position.z,vehicle.position.y,initial.pitch_x_rad), Vec3(
        vehicle.velocity.z,vehicle.velocity.y,initial.pitch_rate_radps)


@dataclass(frozen=True, slots=True)
class _Forces:
    acceleration: Vec3
    force: Vec3
    pitch_torque: float
    spring_energy: float
    actuator_power: float
    dissipation_power: float
    samples: tuple[WheelForceSample, ...]
    contacts: tuple[ContactForceAudit, ...]


def integrate_coupled_longitudinal(previous, initial, physics, body, surfaces,
                                   parameters, control, policy, *, dt, modifiers=None):
    """Advance bounded supported planar contact; reject any unaudited candidate.

    Numerical energy tolerance controls deterministic substep refinement, not
    empirical physics acceptance. Accepted sweeps cover numerical chords/SLERP.
    """
    if (not isinstance(physics,IndependentSuspensionDynamicsPolicy)
            or physics.force_model != 'road_relative_vertical_v2'
            or not isinstance(body,VehicleBodyGeometry) or not isinstance(parameters,VehicleParameters)
            or not isinstance(control,VehicleControl) or not isinstance(policy,LongitudinalContactPolicy)
            or type(dt) not in (int,float) or not isfinite(dt) or dt<=0.):
        raise ValueError('registered coupled contact inputs required')
    if control.steering != 0.:
        raise ValueError('coupled longitudinal contact requires zero steering')
    q0,v0 = _longitudinal_frame(previous,initial)
    atlas = _atlas(surfaces,physics.contact.seam_height_tolerance_m)
    if atlas is None or any(abs(p.normal.x)>1e-10 for p in atlas[0]):
        raise SuspensionResolutionError('coupled contact requires longitudinal 2.5D road atlas')
    modifiers = modifiers or MotionModifiers()
    if not isinstance(modifiers,MotionModifiers):
        raise ValueError('motion modifiers required')
    mass,inertia,gravity = physics.mass_kg,physics.pitch_inertia_kg_m2,parameters.gravity
    preload = mass*gravity/4.
    patches = {p.id:p for p in atlas[0]}

    def pose(q,v):
        return VehicleState(Vec3(previous.position.x,q.y,q.x),Vec3(0.,v.y,v.x),
            Quaternion.from_axis_angle(RIGHT,q.z),RIGHT*v.z,True)

    def forces(q,v):
        state = pose(q,v)
        if abs(q.z)>physics.max_abs_tilt_rad or state.speed>parameters.max_ground_speed+1e-8:
            raise SuspensionResolutionError('coupled state exceeds tilt/speed domain')
        contacts = _contacts(state,physics.contact,atlas)
        if contacts is None:
            raise SuspensionResolutionError('coupled wheel support/travel missing')
        drag = state.velocity*(-mass*parameters.rolling_drag*modifiers.rolling_drag_multiplier)
        boost = FORWARD*(mass*modifiers.longitudinal_acceleration_bonus)
        total = UP*(-mass*gravity)+drag+boost
        torque = RIGHT*(-physics.angular_damping_nmsprad*v.z)
        actuator = boost.dot(state.velocity)
        dissipation = -drag.dot(state.velocity)+physics.angular_damping_nmsprad*v.z*v.z
        spring_energy = 0.
        samples,records = [],[]
        for index,(local,contact) in enumerate(zip(physics.contact.wheels.positions,contacts,strict=True)):
            patch = patches[contact.patch_id]
            arm = state.orientation.rotate(local)
            point_velocity = state.velocity+state.angular_velocity.cross(arm)
            gradient = patch.normal/patch.normal.y
            gap_rate = gradient.dot(point_velocity)
            raw_elastic = preload+physics.spring_stiffness_npm*contact.deflection_m
            elastic = max(0.,raw_elastic)
            load = max(0.,raw_elastic-physics.damper_nspm*gap_rate)
            reaction = gradient*load
            tangent = FORWARD.reject_from(patch.normal).normalized()
            speed = point_velocity.dot(tangent)
            request = mass*(control.throttle*parameters.max_acceleration
                -control.brake*parameters.max_braking*tanh(speed/policy.brake_speed_scale_mps))/4.
            normal_force = load/patch.normal.y
            limit = policy.traction_mu*normal_force
            traction = max(-limit,min(limit,request))
            if traction*speed>policy.max_drive_power_w/4.:
                traction = policy.max_drive_power_w/(4.*speed)
            wheel_force = reaction+tangent*traction
            total = total+wheel_force
            torque = torque+arm.cross(wheel_force)
            actuator += traction*speed
            loss = (elastic-load)*gap_rate
            if loss < -1e-8:
                raise SuspensionResolutionError('normal damper produced energy')
            dissipation += max(0.,loss)
            spring_energy += elastic*elastic/(2.*physics.spring_stiffness_npm)
            samples.append(WheelForceSample(contact,load,gap_rate))
            records.append(ContactForceAudit(index,patch.id,normal_force,traction,speed))
        if abs(total.x)>1e-7 or abs(torque.y)>1e-7 or abs(torque.z)>1e-7:
            raise SuspensionResolutionError('nonplanar contact force/torque is unsupported')
        if not any(s.upward_force_n>0. for s in samples):
            raise SuspensionResolutionError('unloaded contact requires an airborne solver')
        acceleration = Vec3(total.z/mass,total.y/mass,torque.x/inertia)
        return _Forces(acceleration,total,torque.x,spring_energy,actuator,dissipation,
                       tuple(samples),tuple(records))

    def kinetic(v):
        return .5*mass*(v.x*v.x+v.y*v.y)+.5*inertia*v.z*v.z

    def energy(q,v,f):
        return kinetic(v)+mass*gravity*q.y+f.spring_energy

    poses,history,segments,times = [previous],[],[],[0.]

    def advance(q,v,start,h,depth):
        f1 = forces(q,v)
        v2 = v+f1.acceleration*(h/2.)
        f2 = forces(q+v*(h/2.),v2)
        v3 = v+f2.acceleration*(h/2.)
        f3 = forces(q+v2*(h/2.),v3)
        v4 = v+f3.acceleration*h
        f4 = forces(q+v3*h,v4)
        stages = (f1,f2,f3,f4)
        weights = (1.,2.,2.,1.)
        end_q = q+(v+v2*2.+v3*2.+v4)*(h/6.)
        end_v = v+(f1.acceleration+f2.acceleration*2.+f3.acceleration*2.+f4.acceleration)*(h/6.)
        final = forces(end_q,end_v)
        work = h/6.*sum(w*f.actuator_power for w,f in zip(weights,stages,strict=True))
        lost = h/6.*sum(w*f.dissipation_power for w,f in zip(weights,stages,strict=True))
        initial_energy,final_energy = energy(q,v,f1),energy(end_q,end_v,final)
        residual = final_energy-initial_energy-work+lost
        # Exclude absolute gravitational potential: shifting world height must
        # not loosen an energy tolerance. This is a numerical error contract.
        scale = max(1.,kinetic(v),kinetic(end_v),f1.spring_energy,final.spring_energy,
                    abs(mass*gravity*(end_q.y-q.y)),abs(work),lost)
        tolerance = policy.energy_absolute_tolerance_j+policy.energy_relative_tolerance*scale
        if abs(residual)>tolerance:
            if depth>=policy.max_refinements:
                raise SuspensionResolutionError('coupled energy balance unresolved at refinement limit')
            mid_q,mid_v = advance(q,v,start,h/2.,depth+1)
            return advance(mid_q,mid_v,times[-1],h/2.,depth+1)
        first,last = pose(q,v),pose(end_q,end_v)
        verdict = certify_multisurface_support(first,last,body,physics.contact,surfaces,
                                               body_clearance_version='bounded_patch_planes_v2')
        if not verdict.valid:
            raise SuspensionResolutionError(f'coupled body/contact sweep {verdict.status}: {verdict.reason}')
        impulse = (f1.force+f2.force*2.+f3.force*2.+f4.force)*(h/6.)
        pitch_impulse = h/6.*sum(w*f.pitch_torque for w,f in zip(weights,stages,strict=True))
        linear_error = ((last.velocity-first.velocity)*mass-impulse).magnitude
        angular_error = abs((end_v.z-v.z)*inertia-pitch_impulse)
        if linear_error>1e-7 or angular_error>1e-7:
            raise SuspensionResolutionError('coupled momentum audit failed')
        if len(segments)>=4096:
            raise SuspensionResolutionError('coupled substep budget exhausted')
        segments.append(LongitudinalSubstepAudit(start,start+h,impulse,pitch_impulse,
            linear_error,angular_error,initial_energy,final_energy,work,lost,residual,tolerance,
            tuple(f.contacts for f in stages)))
        history.append(f1.samples)
        poses.append(last)
        times.append(start+h)
        return end_q,end_v

    q,v = q0,v0
    h = dt/physics.integration_substeps
    for _ in range(physics.integration_substeps):
        q,v = advance(q,v,times[-1],h,0)
    dynamics = SuspensionDynamicsState(q.y,v.y,q.z,v.z)
    suspension = SuspensionDynamicsResult(poses[-1],dynamics,tuple(poses),tuple(history))
    audit = CoupledGroundAudit('coupled_longitudinal_rk4_work_v1',tuple(segments),
        sum(s.actuator_work_j for s in segments),sum(s.dissipation_j for s in segments),
        sum(s.energy_residual_j for s in segments),
        max(s.linear_impulse_residual_ns for s in segments),max(s.angular_impulse_residual_nms for s in segments))
    centers = tuple(CenterSegment(a,b,p.position,r.position)
                    for a,b,p,r in zip(times,times[1:],poses,poses[1:]))
    return CoupledGroundResult(suspension,audit,centers)
