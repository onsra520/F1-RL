"""Supported spatial rigid body with massless spring/contact references.

Position, quaternion, linear velocity and world angular momentum share one
force field. Tire forces use a friction circle and dissipative lateral slip;
there is no wheel-spin or static-friction claim. The separate compliant
lifecycle entry point selects bounded fixed-attitude or world-H rotating flight above the atlas.
Every accepted RK4 leaf must pass work/impulse and numerical body/road sweeps.
"""
from dataclasses import dataclass, replace
from math import asin, atan2, cos, isfinite, pi, sin, tanh, ulp

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
from .multi_surface_support import (
    _WHEEL_NAMES, _atlas, _contacts, _intersection, _square, _within_shape,
    certify_multisurface_support,
)
from .spatial_partial_wheel_contact import sample_spatial_partial_wheels
from .finite_tire_footprint_v4 import sample_finite_wheel_coverage
from .wheel_edge_release_energy import certify_wheel_edge_release_energy
from .wheel_bump_stop_impact import certify_wheel_bump_stop
from .roll_envelope_terminal import RollEnvelopeStop,RollEnvelopeSource,CertifiedRollEnvelopeExit
from .wheel_edge_overtravel_impact import (
    incoming_edge,entry_margin,apply_edge_overtravel_impulse_at_prefix,
)
from .rigid_body_impact_response import RigidBodyImpactMass
from .accepted_contact_transition import apply_bump_impulse_at_certified_clear_pose
from .accepted_road_body_contact import apply_first_road_body_contact
from .physical_crash import PhysicalCrash
from .swept_surface_clearance import certify_swept_chassis_clearance, certify_swept_chassis_patch_intersection_free, certify_first_bounded_road_contact
from .body_collision import _slerp
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
    road_edge_release_dissipation_j: float = 0.


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
class SpatialPhaseBracket:
    kind: str
    lower_time_s: float
    upper_time_s: float

    def __post_init__(self):
        if (self.kind not in ('takeoff','landing')
                or any(type(t) not in (int,float) or not isfinite(t) for t in (self.lower_time_s,self.upper_time_s))
                or not 0.<=self.lower_time_s<self.upper_time_s
                or self.upper_time_s-self.lower_time_s>1e-8):
            raise ValueError('finite bounded compliant phase bracket required')


@dataclass(frozen=True, slots=True)
class SpatialLifecycleAudit(SpatialGroundAudit):
    phase_brackets: tuple[SpatialPhaseBracket, ...]


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
    compressed: bool


def integrate_spatial_contact(previous, physics, body, surfaces, parameters, control,
                              policy, *, dt, modifiers=None):
    return _integrate_spatial(previous,physics,body,surfaces,parameters,control,policy,
                              dt=dt,modifiers=modifiers,lifecycle=False)


def integrate_spatial_lifecycle(previous, physics, body, surfaces, parameters, control,
                                policy, *, dt, modifiers=None, phase_domain='horizontal_v1'):
    """Compliant release/recontact above a continuously covered road atlas.

    Free extension must end before the registered droop stop. Rotating flight
    requires atlas_rotating_v3; road-edge departure remains unsupported.
    """
    if phase_domain not in ('horizontal_v1','atlas_gradient_v2','atlas_rotating_v3'):
        raise ValueError('unregistered spatial phase domain')
    if not isinstance(physics,IndependentSuspensionDynamicsPolicy) or physics.force_model != 'road_relative_unilateral_v3':
        raise ValueError('spatial lifecycle requires unilateral contact')
    return _integrate_spatial(previous,physics,body,surfaces,parameters,control,policy,
                              dt=dt,modifiers=modifiers,lifecycle=True,phase_domain=phase_domain)


def integrate_spatial_road_edge_lifecycle(previous, physics, body, surfaces, parameters, control,
                                         policy, *, dt, modifiers=None):
    """Opt-in spatial per-wheel release experiment; NEVER legacy v3."""
    if physics.force_model != 'road_relative_unilateral_v3':
        raise ValueError('spatial edge mode requires unilateral contact')
    return _integrate_spatial(previous,physics,body,surfaces,parameters,control,policy,
                              dt=dt,modifiers=modifiers,lifecycle=True,
                              phase_domain='atlas_rotating_edge_v4')


def _integrate_spatial(previous, physics, body, surfaces, parameters, control,
                        policy, *, dt, modifiers, lifecycle, phase_domain='horizontal_v1'):
    """Force-coupled supported motion; world Hdot=tau prevents hidden yaw projection.

    Quaternion RK4 is normalized at each accepted candidate. Work refinement
    controls a numerical defect, not the error against a calibrated physical ODE.
    Missing wheel support or an uncertain swept certificate rejects the step.
    """
    if (not isinstance(previous,VehicleState) or not isinstance(physics,IndependentSuspensionDynamicsPolicy)
            or physics.force_model not in ('road_relative_vertical_v2','road_relative_unilateral_v3')
            or not isinstance(body,VehicleBodyGeometry) or not isinstance(parameters,VehicleParameters)
            or not isinstance(control,VehicleControl) or not isinstance(policy,SpatialContactPolicy)
            or type(dt) not in (int,float) or not isfinite(dt) or dt<=0.):
        raise ValueError('registered spatial contact inputs required')
    if not lifecycle and not previous.grounded:
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
    unilateral=physics.force_model=='road_relative_unilateral_v3'
    if unilateral and (not isfinite(preload) or preload<=0.):
        raise SuspensionResolutionError('unilateral contact requires finite positive nominal preload')
    if lifecycle and preload/physics.spring_stiffness_npm > physics.contact.max_droop_m:
        raise SuspensionResolutionError('free spring extension exceeds registered droop stop')
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
            if phase_domain=='atlas_rotating_edge_v4':
                roll_only=(abs(dynamic.roll_z_rad)>physics.max_abs_tilt_rad
                    and abs(dynamic.pitch_x_rad)<=physics.max_abs_tilt_rad
                    and state.speed<=parameters.max_ground_speed+1e-8
                    and state.angular_velocity.magnitude<=policy.max_angular_speed_radps)
                fault=SuspensionResolutionError(
                    f'v4 spatial state outside certified domain: '
                    f'pitch={dynamic.pitch_x_rad:.9g} roll={dynamic.roll_z_rad:.9g} '
                    f'tilt_limit={physics.max_abs_tilt_rad:.9g} '
                    f'speed={state.speed:.9g} speed_limit={parameters.max_ground_speed:.9g} '
                    f'angular={state.angular_velocity.magnitude:.9g} '
                    f'angular_limit={policy.max_angular_speed_radps:.9g}',
                    failure_code='roll_envelope_exit' if roll_only else 'other_domain_exit')
                if roll_only:
                    fault.rejected_vehicle=state
                    fault.roll_limit=physics.max_abs_tilt_rad
                raise fault
            raise SuspensionResolutionError('spatial state exceeds tilt/speed/angular domain')
        if phase_domain=='atlas_rotating_edge_v4':
            # A wheel enters/leaves a finite supporting tire footprint
            # continuously; source the potential gradient from the same
            # smooth road-atlas coverage as the wheel's normal load.
            footprints=sample_finite_wheel_coverage(state,physics.contact,atlas)
            contacts=tuple(f.contact for f in footprints)
        else:
            footprints=None
            contacts=_contacts(state,physics.contact,atlas,free_extension=lifecycle)
        if contacts is None:
            if lifecycle:
                outside_wheel = None
                for name, local in zip(_WHEEL_NAMES, physics.contact.wheels.positions, strict=True):
                    point = state.position + state.orientation.rotate(local)
                    if not any(_within_shape(point.x, point.z, polygon) for polygon in atlas[1]):
                        outside_wheel = name
                        break
                if outside_wheel is not None:
                    raise SuspensionResolutionError(
                        f'track_boundary_exit: {outside_wheel} wheel center left road support',
                        failure_code='track_boundary_exit',
                    )
            raise SuspensionResolutionError('spatial wheel support/travel missing')
        compressed=any(c is not None and preload+physics.spring_stiffness_npm*c.deflection_m>0.
                       for c in contacts)
        if lifecycle and phase_domain not in ('atlas_rotating_v3','atlas_rotating_edge_v4') and not compressed and state.angular_velocity.magnitude>1e-9:
            raise SuspensionResolutionError('spatial compliant flight requires fixed airborne attitude')
        ground_effect=1. if not lifecycle or compressed else 0.
        drag=state.velocity*(-mass*parameters.rolling_drag*modifiers.rolling_drag_multiplier*ground_effect)
        boost=state.orientation.rotate(FORWARD)*(mass*modifiers.longitudinal_acceleration_bonus*ground_effect)
        force=UP*(-mass*gravity)+drag+boost
        torque=state.angular_velocity*(-physics.angular_damping_nmsprad*ground_effect)
        actuator=boost.dot(state.velocity)
        dissipation=-drag.dot(state.velocity)+physics.angular_damping_nmsprad*ground_effect*state.angular_velocity.dot(state.angular_velocity)
        spring_energy=0.
        samples,records=[],[]
        for index,(local,contact) in enumerate(zip(physics.contact.wheels.positions,contacts,strict=True)):
            if contact is None:
                # Physically released/unloaded wheel: zero contact, no tire or damper work.
                continue
            patch=patches[contact.patch_id]
            arm=state.orientation.rotate(local)
            point_velocity=state.velocity+state.angular_velocity.cross(arm)
            gradient=contact.surface_gradient or patch.normal/patch.normal.y
            normal=gradient.normalized()
            gap_rate=gradient.dot(point_velocity)
            raw_elastic=preload+physics.spring_stiffness_npm*contact.deflection_m
            elastic=max(0.,raw_elastic)
            if unilateral:
                # Hunt-Crossley vanishes at zero compression; d=c/preload preserves
                # the registered damper's small-velocity slope at nominal load.
                factor=1.-(physics.damper_nspm/preload)*gap_rate if elastic>0. else 0.
                if not isfinite(factor):
                    raise SuspensionResolutionError('unilateral dissipation factor is nonfinite')
                load=elastic*max(0.,factor)
            else:
                load=max(0.,raw_elastic-physics.damper_nspm*gap_rate)
            coverage=footprints[index].coverage if footprints is not None else 1.
            edge_gradient=(footprints[index].coverage_gradient
                           if footprints is not None else Vec3(0.,0.,0.))
            normal_force=coverage*load*gradient.magnitude
            angle=control.steering*parameters.max_steering_angle if local.z>0. and ground_effect else 0.
            heading=state.orientation.rotate(Vec3(sin(angle),0.,cos(angle))).reject_from(normal)
            if heading.magnitude<1e-8:
                raise SuspensionResolutionError('wheel tangent direction unresolved')
            tangent=heading.normalized()
            lateral=normal.cross(tangent)
            speed,slip=point_velocity.dot(tangent),point_velocity.dot(lateral)
            longitudinal=coverage*mass*(control.throttle*parameters.max_acceleration
                -control.brake*parameters.max_braking*tanh(speed/traction.brake_speed_scale_mps))/4.
            side=-coverage*policy.lateral_damping_nspm*slip
            magnitude=(longitudinal*longitudinal+side*side)**.5
            limit=traction.traction_mu*normal_force
            if magnitude>limit:
                scale=limit/magnitude
                longitudinal,side=longitudinal*scale,side*scale
            if longitudinal*speed>traction.max_drive_power_w/4.:
                longitudinal=traction.max_drive_power_w/(4.*speed)
            stored=elastic*elastic/(2.*physics.spring_stiffness_npm)
            # Conservative edge force is -d(coverage*stored)/d(x,z).
            # Omitting it would create energy as a compressed tire enters.
            wheel_force=(gradient*(coverage*load)+tangent*longitudinal
                         +lateral*side-edge_gradient*stored)
            force=force+wheel_force
            torque=torque+arm.cross(wheel_force)
            actuator+=longitudinal*speed
            normal_loss=coverage*(elastic-load)*gap_rate
            if normal_loss < -1e-8 or side*slip>1e-8:
                raise SuspensionResolutionError('spatial contact produced damping energy')
            dissipation+=max(0.,normal_loss)-side*slip
            spring_energy+=coverage*stored
            samples.append(WheelForceSample(contact,coverage*load,gap_rate))
            records.append(SpatialWheelAudit(index,patch.id,normal_force,longitudinal,side,speed,slip))
        if not lifecycle and not any(s.upward_force_n>0. for s in samples):
            raise SuspensionResolutionError('unloaded spatial contact requires airborne solver')
        qw,qx,qy,qz=y[6:10]
        wx,wy,wz=state.angular_velocity.x,state.angular_velocity.y,state.angular_velocity.z
        qdot=(-.5*(wx*qx+wy*qy+wz*qz),.5*(wx*qw+wy*qz-wz*qy),
              .5*(-wx*qz+wy*qw+wz*qx),.5*(wx*qy-wy*qx+wz*qw))
        derivative=(state.velocity.x,state.velocity.y,state.velocity.z,
                    force.x/mass,force.y/mass,force.z/mass,*qdot,torque.x,torque.y,torque.z)
        return _Stage(derivative,force,torque,spring_energy,actuator,dissipation,tuple(samples),tuple(records),compressed)

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
    brackets,ballistic_leaves=[],[]

    def shifted(y,d,h):
        return tuple(v+h*k for v,k in zip(y,d,strict=True))

    def checked_stage(trial_y,start_y,elapsed):
        try:
            return stage(trial_y)
        except SuspensionResolutionError as error:
            if (phase_domain=='atlas_rotating_edge_v4' and
                getattr(error,'failure_code',None)=='roll_envelope_exit'):
                error.trial_y=trial_y
                error.trial_duration_s=elapsed
                raise
            if (phase_domain!='atlas_rotating_edge_v4' or
                getattr(error,'failure_code',None)!='over_bump_travel' or
                not hasattr(error,'wheel_local')):
                raise
            try:
                # Exact original rejected RK4 trial supplied as a *diagnostic*
                # interpolation only. The event is NOT a valid accepted step.
                witness=certify_wheel_bump_stop(
                    vehicle(start_y),vehicle(trial_y),
                    wheel_local=error.wheel_local,road=error.road_patch,
                    max_bump_m=physics.contact.max_bump_m,
                    mass=RigidBodyImpactMass(mass,inertia),duration_s=elapsed,
                    resolution_s=min(1e-10,elapsed/128.))
            except ValueError as proof_error:
                # Distinct event: the tire is OUTSIDE a polygon footprint at
                # the accepted start, already BELOW the vertical hard stop.
                # Its first physical encounter is a horizontal tyre sidewall
                # against that polygon's finite-radius boundary, not a
                # fictitious crossing of the already-penetrated y=stop plane.
                original=vehicle(start_y)
                rejected=vehicle(trial_y)
                try:
                    patch_index=next(i for i,p in enumerate(atlas[0])
                                     if p.id==error.road_patch.id)
                    polygon=atlas[1][patch_index]
                    edge_index=incoming_edge(
                        original,rejected,error.wheel_local,error.road_patch,
                        polygon,max_bump_m=physics.contact.max_bump_m)
                except (ValueError,StopIteration):
                    pass  # Keep the original fail-closed vertical error.
                else:
                    fault=SuspensionResolutionError(
                        'source-witnessed first compressed tire sidewall entry',
                        failure_code='edge_overtravel')
                    fault.wheel_index=error.wheel_index
                    fault.wheel_local=error.wheel_local
                    fault.road_patch=error.road_patch
                    fault.edge_polygon=polygon
                    fault.edge_index=edge_index
                    fault.trial_y=trial_y
                    fault.trial_duration_s=elapsed
                    raise fault from error
                n=error.road_patch.normal
                def signed_gap(p):
                    return (physics.contact.max_bump_m*n.y+
                        (p.position+p.orientation.rotate(error.wheel_local)
                         -error.road_patch.origin).dot(n))
                s0,s1=vehicle(start_y),vehicle(trial_y)
                speed0=(s0.velocity+s0.angular_velocity.cross(
                    s0.orientation.rotate(error.wheel_local))).dot(n)
                speed1=(s1.velocity+s1.angular_velocity.cross(
                    s1.orientation.rotate(error.wheel_local))).dot(n)
                raise SuspensionResolutionError(
                    f'{error}; unscored hard-stop trial not certified: {proof_error}'
                    f' clear_gap_m={signed_gap(s0):.12g}'
                    f' rejected_gap_m={signed_gap(s1):.12g}'
                    f' vnormal0={speed0:.12g} vnormal1={speed1:.12g}'
                    f' elapsed_s={elapsed:.12g}',
                    failure_code='over_bump_travel') from error
            fault=SuspensionResolutionError(
                f'{error}; provisional hard-stop witness time_s='
                f'({witness.first_clear_time_s:.9g},{witness.first_violating_time_s:.9g}) '
                f'normal_impulse_ns={witness.impulse_n_s} '
                f'angular_impulse_nms={witness.angular_impulse_nms} '
                f'energy_lost_j={witness.dissipated_kinetic_j:.9g}; '
                f'NOT_APPLIED_NOT_SCORED',failure_code='over_bump_travel')
            fault.wheel_index=error.wheel_index
            fault.wheel_local=error.wheel_local
            fault.road_patch=error.road_patch
            fault.trial_y=trial_y
            fault.trial_duration_s=elapsed
            raise fault from error

    def advance(y,start,h,depth,phase_depth=0):
        if phase_domain in ('atlas_rotating_v3','atlas_rotating_edge_v4') and h*max(policy.max_angular_speed_radps,
                Vec3(*y[10:13]).magnitude/min(inertia.x,inertia.y,inertia.z))>=pi:
            if depth>=traction.max_refinements:
                raise SuspensionResolutionError('spatial rotation interval cannot exclude a full turn')
            middle=advance(y,start,h/2.,depth+1,phase_depth)
            return advance(middle,times[-1],h/2.,depth+1,phase_depth)
        a=stage(y)
        try:
            b=checked_stage(shifted(y,a.derivative,h/2.),y,h/2.)
            c=checked_stage(shifted(y,b.derivative,h/2.),y,h/2.)
            d=checked_stage(shifted(y,c.derivative,h),y,h)
            stages=(a,b,c,d)
            weights=(1.,2.,2.,1.)
            end=tuple(v+h/6.*(ka+2.*kb+2.*kc+kd)
                      for v,ka,kb,kc,kd in zip(y,a.derivative,b.derivative,c.derivative,d.derivative,strict=True))
            q=Quaternion(*end[6:10])
            end=(*end[:6],q.w,q.x,q.y,q.z,*end[10:13])
            final=checked_stage(end,y,h)
        except SuspensionResolutionError as fault:
            if (phase_domain!='atlas_rotating_edge_v4' or
                getattr(fault,'failure_code',None) not in
                    ('over_bump_travel','edge_overtravel','roll_envelope_exit')
                or not hasattr(fault,'trial_y')):
                raise
            # Refine the accepted RK4 prefix rather than applying an impulse
            # against a rejected large-step candidate. Event resolution is
            # only possible in a tiny, monotone source-bounded window.
            if h>1e-5 and phase_depth<30:
                middle=advance(y,start,h/2.,depth,phase_depth+1)
                return advance(middle,times[-1],h/2.,depth,phase_depth+1)
            if h>1e-5:
                raise
            # Find the event on FULL short-step RK4, not the rejected
            # intermediate stage's linear interpolation. Every bisection's
            # admitted prefix is computed from the original accepted state.
            def short_rk4(duration):
                ka=stage(y)
                kb=stage(shifted(y,ka.derivative,duration/2.))
                kc=stage(shifted(y,kb.derivative,duration/2.))
                kd=stage(shifted(y,kc.derivative,duration))
                end_y=tuple(v+duration/6.*(u+2.*v2+2.*v3+w)
                    for v,u,v2,v3,w in zip(
                        y,ka.derivative,kb.derivative,kc.derivative,kd.derivative,
                        strict=True))
                q0=Quaternion(*end_y[6:10])
                end_y=(*end_y[:6],q0.w,q0.x,q0.y,q0.z,*end_y[10:13])
                kfin=stage(end_y)
                return end_y,(ka,kb,kc,kd,kfin)
            high=fault.trial_duration_s
            low=0.
            prefix_y=None
            prefix_stages=None
            try:
                if high<=0. or high>h:
                    raise ValueError('rejected RK4 trial outside source interval')
                for _ in range(55):
                    if high-low<=1e-10:
                        break
                    probe=(low+high)/2.
                    try:
                        trial_y,trial_stages=short_rk4(probe)
                        # The accepted prefix must be clearly on the safe
                        # side of the original, UNCHANGED 0.12m hard stop.
                        proof=vehicle(trial_y)
                        if getattr(fault,'failure_code',None)=='roll_envelope_exit':
                            if abs(_suspension_state(proof).roll_z_rad)>physics.max_abs_tilt_rad:
                                high=probe
                                continue
                        else:
                            wheel=proof.position+proof.orientation.rotate(fault.wheel_local)
                            n=fault.road_patch.normal
                            clearance=physics.contact.max_bump_m*n.y+(
                                wheel-fault.road_patch.origin).dot(n)
                        if getattr(fault,'failure_code',None)=='edge_overtravel':
                            margin=entry_margin(proof,fault.wheel_local,
                                fault.edge_polygon,fault.edge_index)
                            # Reject interior penetration; only a genuine
                            # finite-tyre exterior prefix can be accepted.
                            if margin>0.:
                                high=probe
                                continue
                        elif getattr(fault,'failure_code',None)=='over_bump_travel' and clearance< -1e-9:
                            high=probe
                            continue
                    except SuspensionResolutionError as why:
                        if getattr(fault,'failure_code',None)=='roll_envelope_exit':
                            if getattr(why,'failure_code',None)!='roll_envelope_exit':
                                raise
                            fault.rejected_vehicle=why.rejected_vehicle
                        elif (getattr(why,'failure_code',None)!='over_bump_travel'
                            or getattr(why,'wheel_index',None)!=getattr(fault,'wheel_index',None)):
                            raise
                        high=probe
                        continue
                    low=probe
                    prefix_y,prefix_stages=trial_y,trial_stages
                if prefix_y is None or low<=0.:
                    raise ValueError('no accepted pre-impact RK4 prefix to hard stop')
                if getattr(fault,'failure_code',None)=='roll_envelope_exit':
                    # Stop at the final FULL-RK4 accepted pose, before the
                    # solver's unchanged authored tilt domain. This is NOT a
                    # physical collision: no normal impulse is generated.
                    pre=vehicle(prefix_y)
                    rejected=fault.rejected_vehicle
                    original=vehicle(y)
                    r0=_suspension_state(original)
                    r1=_suspension_state(pre)
                    rbad=_suspension_state(rejected)
                    side=1 if rbad.roll_z_rad>0. else -1
                    if (side*rbad.roll_z_rad<=physics.max_abs_tilt_rad
                        or side*r1.roll_z_rad>physics.max_abs_tilt_rad
                        or side*r1.roll_z_rad<=0.
                        or side*r1.roll_rate_radps<=1e-6
                        or abs(r0.roll_z_rad)>=physics.max_abs_tilt_rad
                        or high-low>1.0001e-10):
                        raise ValueError('unwitnessed first entering roll-envelope boundary')
                    from .body_collision import obb_separation_lower_bound
                    from .track.volume import OrientedBoxVolume
                    for check in (original,pre,rejected):
                        box=body.box_at(check.position,check.orientation)
                        for road_patch in atlas[0]:
                            road_box=OrientedBoxVolume(road_patch.origin,
                                Vec3(road_patch.width/2.,1e-9,road_patch.length/2.),
                                Quaternion.from_basis(road_patch.right,
                                    road_patch.normal,road_patch.forward))
                            if obb_separation_lower_bound(box,road_box)<=1e-6:
                                raise ValueError('roll limit lacks distinct road-free witness')
                    weights=(1.,2.,2.,1.)
                    forces=sum((k.force*w for k,w in zip(
                        prefix_stages[:4],weights,strict=True)),Vec3(0.,0.,0.))*(low/6.)
                    torques=sum((k.torque*w for k,w in zip(
                        prefix_stages[:4],weights,strict=True)),Vec3(0.,0.,0.))*(low/6.)
                    H=angular_momentum(pre.orientation,pre.angular_velocity)
                    linear=((pre.velocity-original.velocity)*mass-forces).magnitude
                    angular=(H-Vec3(*y[10:13])-torques).magnitude
                    work=low/6.*sum(w*k.actuator_power for w,k in zip(
                        weights,prefix_stages[:4],strict=True))
                    loss=low/6.*sum(w*k.dissipation_power for w,k in zip(
                        weights,prefix_stages[:4],strict=True))
                    e0=energy(y,a)
                    e1=energy(prefix_y,prefix_stages[4])
                    balance=e1-e0-work+loss
                    scale=max(1.,kinetic(y),kinetic(prefix_y),
                        a.spring_energy,prefix_stages[4].spring_energy,abs(work),loss)
                    tolerance=traction.energy_absolute_tolerance_j+traction.energy_relative_tolerance*scale
                    if (abs(balance)>tolerance or linear>1e-7 or angular>1e-7):
                        raise ValueError('roll-prefix impulse or energy ledger invalid')
                    when=start+low
                    prefix=tuple(CenterSegment(t0,t1,p.position,q.position,
                        p.velocity if air else None,gravity if air else 0.)
                        for t0,t1,p,q,air in zip(times[:-1],times[1:],
                            poses[:-1],poses[1:],ballistic_leaves,strict=True))
                    prefix+=(CenterSegment(start,when,original.position,pre.position),)
                    if when<dt:
                        prefix+=(CenterSegment(when,dt,pre.position,pre.position),)
                    leaf=SpatialSubstepAudit(start,when,forces,torques,
                        linear,angular,e0,e1,work,loss,balance,tolerance,
                        tuple(k.contacts for k in prefix_stages[:4])+
                        (prefix_stages[4].contacts,))
                    source=RollEnvelopeSource(original,rejected,start,low,high,
                        tuple(poses),tuple(times),tuple(leaves),leaf,body,atlas)
                    stop=RollEnvelopeStop(when,start+high,
                        physics.max_abs_tilt_rad,side,r0.roll_z_rad,
                        r1.roll_z_rad,rbad.roll_z_rad,pre,rejected,prefix,source)
                    raise CertifiedRollEnvelopeExit(stop)
                pre=vehicle(prefix_y)
                if getattr(fault,'failure_code',None)=='edge_overtravel':
                    impact=apply_edge_overtravel_impulse_at_prefix(
                        pre,RigidBodyImpactMass(mass,inertia),
                        wheel_local=fault.wheel_local,road=fault.road_patch,
                        polygon=fault.edge_polygon,edge_index=fault.edge_index,
                        max_bump_m=physics.contact.max_bump_m,
                        first_clear_s=low,first_witness_s=high)
                else:
                    impact=apply_bump_impulse_at_certified_clear_pose(
                        pre,RigidBodyImpactMass(mass,inertia),
                        wheel_local=fault.wheel_local,road=fault.road_patch,
                        max_bump_m=physics.contact.max_bump_m,
                        first_clear_s=low,first_witness_s=high)
                post=impact.postimpact
                q=post.orientation
                H=angular_momentum(q,post.angular_velocity)
                after=(post.position.x,post.position.y,post.position.z,
                       post.velocity.x,post.velocity.y,post.velocity.z,
                       q.w,q.x,q.y,q.z,H.x,H.y,H.z)
                after_stage=stage(after)
                pre_a,pre_b,pre_c,pre_d,pre_final=prefix_stages
                weights=(1.,2.,2.,1.)
                source_force=sum((k.force*w for k,w in zip(
                    prefix_stages[:4],weights,strict=True)),Vec3(0.,0.,0.))*(low/6.)
                source_torque=sum((k.torque*w for k,w in zip(
                    prefix_stages[:4],weights,strict=True)),Vec3(0.,0.,0.))*(low/6.)
                total_force=source_force+impact.impulse_n_s
                total_torque=source_torque+impact.angular_impulse_nms
                lin=((post.velocity-vehicle(y).velocity)*mass-total_force).magnitude
                ang=(H-Vec3(*y[10:13])-total_torque).magnitude
                work=low/6.*sum(w*k.actuator_power for w,k in zip(weights,prefix_stages[:4],strict=True))
                regular_loss=low/6.*sum(w*k.dissipation_power for w,k in zip(weights,prefix_stages[:4],strict=True))
                loss=regular_loss+impact.energy_lost_j
                E0=energy(y,a)
                E1=energy(after,after_stage)
                balance=E1-E0-work+loss
                scale=max(1.,kinetic(y),kinetic(after),a.spring_energy,
                          after_stage.spring_energy,abs(work),loss)
                tolerance=(traction.energy_absolute_tolerance_j+
                           traction.energy_relative_tolerance*scale)
                if (abs(balance)>tolerance or lin>1e-7 or ang>1e-7
                    or not start<start+low<=start+h):
                    raise ValueError(f'accepted RK4 hard-stop conservation failed: '
                        f'energy={balance:.9g}/{tolerance:.9g} '
                        f'linear_ns={lin:.9g} angular_nms={ang:.9g}')
            except (ValueError,SuspensionResolutionError) as reason:
                raise SuspensionResolutionError(
                    f'v4 hard-stop remains invalid: {reason}',
                    failure_code='over_bump_travel') from fault
            if phase_domain=='atlas_rotating_edge_v4':
                # This first certified rigid hard-stop impulse terminates
                # immediately. Continuing the remaining tick without a
                # persistent unilateral constraint would let the very next
                # RK4 stage penetrate again and double-count the impulse.
                tick_hit=start+high
                if not start<tick_hit<=dt or not 0.<high-low<=1e-10+1e-14:
                    raise SuspensionResolutionError(
                        'wheel hard stop certified witness time outside tick')
                previous_prefix=tuple(
                    CenterSegment(t0,t1,p.position,q.position,
                        p.velocity if air else None,gravity if air else 0.)
                    for t0,t1,p,q,air in zip(
                        times[:-1],times[1:],poses[:-1],poses[1:],
                        ballistic_leaves,strict=True))
                prefix=previous_prefix+(
                    CenterSegment(start,tick_hit,vehicle(y).position,
                                  impact.preimpact.position),)
                if tick_hit<dt:
                    prefix+=(CenterSegment(tick_hit,dt,
                        impact.preimpact.position,impact.preimpact.position),)
                edge_hit=getattr(fault,'failure_code',None)=='edge_overtravel'
                crash=PhysicalCrash(
                    (f'wheel-edge-stop:{fault.road_patch.id}' if edge_hit else
                     f'wheel-hard-stop:{fault.road_patch.id}'),
                    tick_hit,start+low,tick_hit,len(leaves),impact,prefix,
                    version=('wp3_grounded_wheel_edge_stop_crash_v4_dev' if edge_hit
                             else 'wp3_grounded_wheel_hard_stop_crash_v4_dev'))
                impact_leaf=SpatialSubstepAudit(
                    start,start+low,total_force,total_torque,lin,ang,
                    E0,E1,work,loss,balance,tolerance,
                    tuple(k.contacts for k in prefix_stages[:4])+
                    (after_stage.contacts,))
                source=SpatialRoadCrashSource(
                    vehicle(y),vehicle(fault.trial_y),
                    start,fault.trial_duration_s,atlas,body,
                    RigidBodyImpactMass(mass,inertia),
                    tuple(poses),tuple(times),tuple(leaves),
                    tuple(ballistic_leaves),tuple(brackets),
                    fault.wheel_local,fault.road_patch,low,high,
                    impact_leaf,
                    fault.edge_polygon if edge_hit else None,
                    fault.edge_index if edge_hit else None)
                raise SpatialRoadCrash(crash,source)
            tick_hit=start+low
            leaves.append(SpatialSubstepAudit(start,tick_hit,
                total_force,total_torque,lin,ang,E0,E1,work,loss,balance,
                tolerance,tuple(k.contacts for k in prefix_stages[:4])+
                (after_stage.contacts,)))
            poses.append(post)
            history.append(a.samples)
            times.append(tick_hit)
            ballistic_leaves.append(False)
            rest=start+h-tick_hit
            if rest<=1e-13:
                return after
            return advance(after,tick_hit,rest,depth,phase_depth+1)
        phase_change=lifecycle and any(f.compressed!=a.compressed for f in (*stages,final))
        if phase_change and h>1e-8:
            if phase_depth>=24:
                raise SuspensionResolutionError('spatial phase time unresolved at refinement limit')
            middle=advance(y,start,h/2.,depth,phase_depth+1)
            return advance(middle,times[-1],h/2.,depth,phase_depth+1)
        if phase_change and a.compressed==final.compressed:
            raise SuspensionResolutionError('multiple spatial phase changes inside event bracket')
        work=h/6.*sum(w*f.actuator_power for w,f in zip(weights,stages,strict=True))
        lost=h/6.*sum(w*f.dissipation_power for w,f in zip(weights,stages,strict=True))
        e0,e1=energy(y,a),energy(end,final)
        residual=e1-e0-work+lost
        scale=max(1.,kinetic(y),kinetic(end),a.spring_energy,final.spring_energy,
                  abs(mass*gravity*(end[1]-y[1])),abs(work),lost)
        tolerance=traction.energy_absolute_tolerance_j+traction.energy_relative_tolerance*scale
        edge_release_loss=0.
        if phase_domain=='atlas_rotating_edge_v4' and abs(residual)>tolerance:
            signatures=tuple(frozenset(item.wheel_index for item in f.contacts)
                             for f in (*stages,final))
            # Explicit, bounded dissipative road-contact release ledger.
            # A released massless wheel cannot create a body momentum impulse.
            # Missing event-time proof or rebound must remain a physics failure.
            if h<=1e-8:
                try:
                    release=certify_wheel_edge_release_energy(
                        signatures,start_spring_energy_j=a.spring_energy,
                        end_spring_energy_j=final.spring_energy,
                        balance_residual_j=residual,
                        balance_tolerance_j=tolerance,bracket_duration_s=h)
                except ValueError:
                    pass  # Original energy audit below must still fail closed.
                else:
                    edge_release_loss=release.transferred_to_environment_j
                    lost+=edge_release_loss
                    residual=release.residual_after_transfer_j
        if abs(residual)>tolerance:
            if depth>=traction.max_refinements:
                if phase_domain=='atlas_rotating_edge_v4':
                    signatures=tuple(tuple(item.wheel_index for item in stage.contacts) for stage in (*stages,final))
                    raise SuspensionResolutionError(f'v4 spatial energy refinement exhausted: h={h:.3g} residual={residual:.9g} tol={tolerance:.9g} spring0={a.spring_energy:.8g} spring1={final.spring_energy:.8g} masks={signatures}')
                masks = tuple(tuple(item.patch_id for item in force.contacts) for force in (*stages, final))
                contact_samples = tuple(tuple(
                    (sample.contact.patch_id, sample.contact.ground_height_m,
                     sample.contact.deflection_m, sample.vertical_wheel_velocity_mps,
                     sample.contact.surface_gradient)
                    for sample in force.samples
                ) for force in (*stages, final))
                raise SuspensionResolutionError(
                    f'spatial energy balance unresolved at refinement limit: '
                    f'start={start:.9g} h={h:.9g} depth={depth} '
                    f'residual_j={residual:.9g} tolerance_j={tolerance:.9g} '
                    f'spring_start_j={a.spring_energy:.9g} spring_end_j={final.spring_energy:.9g} '
                    f'contact_masks={masks} contact_samples={contact_samples}'
                )
            middle=advance(y,start,h/2.,depth+1)
            return advance(middle,times[-1],h/2.,depth+1)
        first,last=vehicle(y),vehicle(end)
        if lifecycle:
            first,last=replace(first,grounded=a.compressed),replace(last,grounded=final.compressed)
        if phase_change and phase_domain!='atlas_rotating_edge_v4':
            # A SLERP wheel/plane gap has normalized second derivative bounded
            # by theta^2*|r|*|gradient|. Its secant exceeds every derivative's
            # deviation by half this bound, proving a unique phase crossing.
            relative=last.orientation*Quaternion(first.orientation.w,-first.orientation.x,
                                                  -first.orientation.y,-first.orientation.z)
            theta=2.*atan2(Vec3(relative.x,relative.y,relative.z).magnitude,abs(relative.w))
            direction=1. if final.compressed else -1.
            for local in physics.contact.wheels.positions:
                first_point=first.position+first.orientation.rotate(local)
                last_point=last.position+last.orientation.rotate(local)
                radius=(last.position-first.position).magnitude+theta*local.magnitude
                footprint=_square(first_point.x,first_point.z,radius)
                crossed=[p for p,poly in zip(*atlas) if _intersection(footprint,poly)]
                if phase_domain=='horizontal_v1':
                    if not crossed or any(p.normal!=UP for p in crossed) or len({p.origin.y for p in crossed})!=1:
                        raise SuspensionResolutionError('spatial phase uniqueness requires local horizontal road')
                    gradients=(UP,)
                else:
                    if not crossed:
                        raise SuspensionResolutionError('spatial phase atlas coverage unresolved')
                    gradients=tuple(p.normal/p.normal.y for p in crossed)
                # Continuous atlas heights agree at seams. Every potentially
                # crossed plane must have the same strict gap direction.
                for gradient in gradients:
                    secant=(first_point-last_point).dot(gradient)
                    curvature=theta*theta*local.magnitude*gradient.magnitude
                    roundoff=0.
                    if phase_domain!='horizontal_v1':
                        scale=max(1.,*(abs(v) for point in (first_point,last_point)
                                       for v in (point.x,point.y,point.z)))
                        roundoff=64.*ulp(scale)*(abs(gradient.x)+abs(gradient.y)+abs(gradient.z))
                    if direction*secant<=.5*curvature+roundoff:
                        raise SuspensionResolutionError('spatial phase crossing monotonicity unresolved')
        ballistic=lifecycle and not any(f.compressed for f in (*stages,final))
        if phase_domain=='atlas_rotating_edge_v4':
            # No all-wheel footprint demand; independently prove the complete
            # chassis does not cross any currently present road patch.
            # Unknown/grazing geometric intervals always reject the step.
            def pose_at(t):
                return VehicleState(first.position+(last.position-first.position)*t+
                                    Vec3(0.,.5*gravity*h*h*t*(1.-t) if ballistic else 0.,0.),
                                    first.velocity+(last.velocity-first.velocity)*t,
                                    _slerp(first.orientation,last.orientation,t),
                                    first.angular_velocity+(last.angular_velocity-first.angular_velocity)*t,
                                    first.grounded)
            try:
                certify_swept_chassis_patch_intersection_free(pose_at,0.,1.,body,atlas,
                    max_subdivisions=14,
                    gravity_mps2=gravity*h*h if ballistic else 0.)
            except SuspensionResolutionError as error:
                if (phase_domain=='atlas_rotating_edge_v4'
                    and 'bounded chassis witnessed physical road contact' in str(error)
                    and h>1e-8):
                    try:
                        early_contact=apply_first_road_body_contact(
                            first,last,body,atlas,RigidBodyImpactMass(mass,inertia),
                            duration_s=h,
                            ballistic_gravity_mps2=gravity if ballistic else 0.)
                    except ValueError as diagnostic_error:
                        if h>1e-5:
                            raise SuspensionResolutionError(
                                f'v4 first road proof diagnostic h={h:.9g} '
                                f'start_time={start:.9g} '
                                f'first={first.position} last={last.position} '
                                f'cause={diagnostic_error}') from error
                        early_contact=None  # May be a corner/seam, never invent a face.
                    if early_contact is not None:
                        time_s=start+early_contact.contact.witnessed_contact_s
                        original_prefix=tuple(
                            CenterSegment(t0,t1,p.position,r.position,
                                p.velocity if airborne else None,
                                gravity if airborne else 0.)
                            for t0,t1,p,r,airborne in zip(
                                times[:-1],times[1:],poses[:-1],poses[1:],
                                ballistic_leaves,strict=True))
                        if (time_s<=start or time_s>dt
                            or len(original_prefix)!=len(leaves)):
                            raise SuspensionResolutionError(
                                'first road contact source time is not an accepted prefix') from error
                        event_segment=CenterSegment(start,time_s,first.position,
                            early_contact.contact.preimpact.position,
                            first.velocity if ballistic else None,
                            gravity if ballistic else 0.)
                        prefix=original_prefix+(event_segment,)
                        if time_s<dt:
                            prefix+=(CenterSegment(time_s,dt,
                                early_contact.contact.preimpact.position,
                                early_contact.contact.preimpact.position),)
                        contact=early_contact.contact
                        crash=PhysicalCrash(
                            f'road-patch:{early_contact.patch_id}',time_s,
                            start+contact.first_clear_s,time_s,
                            len(original_prefix),contact,prefix,
                            version=('wp3_grounded_applied_road_crash_v4_dev'
                                if contact.preimpact.grounded else
                                'wp3_airborne_applied_road_crash_v4_dev'))
                        source=SpatialRoadCrashSource(
                            first,last,start,h,atlas,body,
                            RigidBodyImpactMass(mass,inertia),
                            tuple(poses),tuple(times),tuple(leaves),
                            tuple(ballistic_leaves),tuple(brackets))
                        raise SpatialRoadCrash(crash,source) from error
                if lifecycle and not phase_change and h>1e-8 and phase_depth<24:
                    middle=advance(y,start,h/2.,depth,phase_depth+1)
                    return advance(middle,times[-1],h/2.,depth,phase_depth+1)
                witness='no certified first bounded patch collision'
                if 'witnessed physical road contact' in str(error):
                    try:
                        event=certify_first_bounded_road_contact(first,last,body,atlas,
                            duration_s=h,resolution_s=1e-6,
                            ballistic_gravity_mps2=gravity if ballistic else 0.)
                        if event is not None:
                            witness=f'first bounded patch interval={event}'
                    except ValueError as proof_error:
                        witness=f'first patch contact unresolved: {proof_error}'
                raise SuspensionResolutionError(f'spatial edge chassis sweep uncertified: {error}; {witness}') from error
        else:
            verdict=certify_multisurface_support(first,last,body,physics.contact,surfaces,
                                            body_clearance_version='bounded_patch_planes_v2',
            contact_domain='unilateral_free_extension_v1' if lifecycle else 'four_supported_v1',
            ballistic_sag_m=gravity*h*h/8. if ballistic else 0.,
            maximum_deflection_m=-preload/physics.spring_stiffness_npm if ballistic else None,
            minimum_support_deflection_m=-preload/physics.spring_stiffness_npm
                if lifecycle and not ballistic and not phase_change else None,
            body_coverage_domain=(
                'wheel_supported_overhang_v1' if lifecycle else 'full_footprint_v1'
            ))
        if phase_domain!='atlas_rotating_edge_v4' and not verdict.valid:
            if lifecycle and not phase_change and h>1e-8 and phase_depth<24:
                middle=advance(y,start,h/2.,depth,phase_depth+1)
                return advance(middle,times[-1],h/2.,depth,phase_depth+1)
            if verdict.status == 'outside':
                raise SuspensionResolutionError(
                    f'spatial body/contact sweep {verdict.status}: {verdict.reason}',
                    failure_code='unsupported_contact',
                )
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
            e0,e1,work,lost,residual,tolerance,tuple(f.contacts for f in stages),edge_release_loss))
        if phase_change:
            brackets.append(SpatialPhaseBracket('landing' if final.compressed else 'takeoff',start,start+h))
        ballistic_leaves.append(lifecycle and not any(f.compressed for f in (*stages,final)))
        poses.append(last)
        history.append(a.samples)
        times.append(start+h)
        return end

    if lifecycle and previous.grounded != stage(state).compressed:
        raise SuspensionResolutionError('spatial lifecycle phase disagrees with spring compression')
    h=dt/physics.integration_substeps
    for _ in range(physics.integration_substeps):
        state=advance(state,times[-1],h,0)
    last=poses[-1]
    suspension=SuspensionDynamicsResult(last,_suspension_state(last),tuple(poses),tuple(history))
    audit=SpatialGroundAudit('spatial_contact_world_momentum_rk4_v1',tuple(leaves),
        sum(s.actuator_work_j for s in leaves),sum(s.dissipation_j for s in leaves),
        sum(s.energy_residual_j for s in leaves),max(s.linear_impulse_residual_ns for s in leaves),
        max(s.angular_impulse_residual_nms for s in leaves))
    if lifecycle:
        audit=SpatialLifecycleAudit('spatial_road_edge_partial_lifecycle_rk4_v4' if phase_domain=='atlas_rotating_edge_v4'
            else 'spatial_rotating_compliant_lifecycle_rk4_v3' if phase_domain=='atlas_rotating_v3'
            else 'spatial_compliant_atlas_lifecycle_rk4_v2' if phase_domain=='atlas_gradient_v2'
            else 'spatial_compliant_lifecycle_rk4_v1',audit.substeps,
            audit.actuator_work_j,audit.dissipation_j,audit.energy_residual_j,
            audit.max_linear_impulse_residual_ns,audit.max_angular_impulse_residual_nms,tuple(brackets))
    centers=tuple(CenterSegment(a,b,p.position,r.position,p.velocity if air else None,gravity if air else 0.)
        for a,b,p,r,air in zip(times[:-1],times[1:],poses[:-1],poses[1:],ballistic_leaves,strict=True))
    return SpatialGroundResult(suspension,audit,centers)


@dataclass(frozen=True,slots=True)
class SpatialRoadCrashSource:
    """Development-only source bind for a first bounded road impact."""
    first:VehicleState
    last:VehicleState
    segment_start_s:float
    segment_duration_s:float
    road_atlas:tuple
    body:VehicleBodyGeometry
    mass:RigidBodyImpactMass
    accepted_poses:tuple[VehicleState,...]
    accepted_times_s:tuple[float,...]
    accepted_leaves:tuple[SpatialSubstepAudit,...]
    accepted_ballistic_leaves:tuple[bool,...]
    accepted_phase_brackets:tuple[SpatialPhaseBracket,...]
    wheel_hard_stop_local:Vec3|None=None
    wheel_hard_stop_patch:object|None=None
    wheel_hard_stop_clear_s:float|None=None
    wheel_hard_stop_witness_s:float|None=None
    wheel_hard_stop_impact_audit:SpatialSubstepAudit|None=None
    wheel_edge_polygon:tuple[tuple[float,float],...]|None=None
    wheel_edge_index:int|None=None

class SpatialRoadCrash(RuntimeError):
    """Opt-in v4 physically absorbed road contact, not a scientific outcome."""
    def __init__(self,crash,source:SpatialRoadCrashSource):
        super().__init__("v4 accepted first road body contact: development absorbing terminal")
        self.crash=crash
        self.source=source
