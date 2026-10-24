"""Applied, energy-audited instantaneous WP3 contact updates.

A contact changes the actual rigid VehicleState at its bracketed event time.
Its tick remainder is NOT integrated here. An enclosing simulation may only
use it as an absorbing terminal with a source-registered, prefix-audited rule;
normal driving must resume using the resulting state and remaining dt.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

from .body_collision import VehicleBodyGeometry, _slerp
from .certified_face_impact import FaceImpactCertificate, certify_linear_face_impact
from .center_trajectory import CenterSegment
from .geometry.vector import Vec3
from .rigid_body_impact_response import RigidBodyImpactMass
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState
from .wheel_bump_stop_impact import WheelBumpStopCertificate, certify_wheel_bump_stop
from .track.patch import PlanarSurfacePatch


@dataclass(frozen=True,slots=True)
class AppliedContactState:
    first_clear_s:float
    witnessed_contact_s:float
    preimpact:VehicleState
    postimpact:VehicleState
    impulse_n_s:Vec3
    angular_impulse_nms:Vec3
    kinetic_before_j:float
    kinetic_after_j:float
    energy_lost_j:float
    kind:str
    version:str="wp3_applied_contact_state_v1"


def _interpolate(first:VehicleState,last:VehicleState,t:float)->VehicleState:
    return VehicleState(
        first.position+(last.position-first.position)*t,
        first.velocity+(last.velocity-first.velocity)*t,
        _slerp(first.orientation,last.orientation,t),
        first.angular_velocity+(last.angular_velocity-first.angular_velocity)*t,
        first.grounded and last.grounded)


def _apply_mass_impulse(
    start:VehicleState, mass:RigidBodyImpactMass, impulse:Vec3, angular_impulse:Vec3,
    pre_j:float,post_j:float,lost_j:float,
    *,
    first_clear:float,witness:float,kind:str
)->AppliedContactState:
    new=VehicleState(
        start.position,start.velocity+impulse/mass.mass_kg,start.orientation,
        start.angular_velocity+mass.world_inertia_inverse(start.orientation,angular_impulse),
        start.grounded)
    def kinetic(v:Vec3,w:Vec3)->float:
        return .5*mass.mass_kg*v.dot(v)+.5*w.dot(mass.world_inertia(new.orientation,w))
    actual_before=kinetic(start.velocity,start.angular_velocity)
    actual_after=kinetic(new.velocity,new.angular_velocity)
    scale=max(1.,actual_before,actual_after)
    if (not 0.<=first_clear<witness or
        not all(isfinite(v) for v in (pre_j,post_j,lost_j,actual_before,actual_after))
        or not isclose(actual_before,pre_j,abs_tol=1e-7,rel_tol=1e-9)
        or not isclose(actual_after,post_j,abs_tol=1e-7,rel_tol=1e-9)
        or actual_after>actual_before+1e-9*scale
        or not isclose(actual_before-actual_after,lost_j,abs_tol=1e-7,rel_tol=1e-9)
        or ((new.velocity-start.velocity)*mass.mass_kg-impulse).magnitude>1e-7
        or (mass.world_inertia(new.orientation,new.angular_velocity-start.angular_velocity)-angular_impulse).magnitude>1e-7):
        raise ValueError("post-impact accepted state fails linear/angular impulse or kinetic energy audit")
    return AppliedContactState(
        first_clear,witness,start,new,impulse,angular_impulse,
        pre_j,post_j,lost_j,kind)


def apply_certified_face_impulse(
    first:VehicleState,last:VehicleState,hull:VehicleBodyGeometry,
    obstacle:OrientedBoxVolume,mass:RigidBodyImpactMass,*,duration_s:float
)->AppliedContactState:
    cert=certify_linear_face_impact(first,last,hull,obstacle,mass,duration_s=duration_s)
    pre=_interpolate(first,last,cert.first_witnessed_overlap_s/duration_s)
    r=cert.response
    return _apply_mass_impulse(pre,mass,r.impulse_n_s,r.angular_impulse_nms,
        r.before_kinetic_j,r.after_kinetic_j,r.energy_lost_j,
        first_clear=cert.first_certified_clear_s,
        witness=cert.first_witnessed_overlap_s,kind="obstacle_face")


def apply_certified_bump_stop_impulse(
    first:VehicleState,last:VehicleState,mass:RigidBodyImpactMass,
    *,wheel_local:Vec3,road:PlanarSurfacePatch,max_bump_m:float,
    duration_s:float,resolution_s:float=1e-6
)->AppliedContactState:
    cert=certify_wheel_bump_stop(first,last,wheel_local=wheel_local,road=road,
        max_bump_m=max_bump_m,mass=mass,duration_s=duration_s,
        resolution_s=resolution_s)
    pre=_interpolate(first,last,cert.first_violating_time_s/duration_s)
    return _apply_mass_impulse(pre,mass,cert.impulse_n_s,cert.angular_impulse_nms,
        cert.kinetic_before_j,cert.kinetic_after_j,cert.dissipated_kinetic_j,
        first_clear=cert.first_clear_time_s,
        witness=cert.first_violating_time_s,kind="wheel_hard_stop")


def absorb_crash_center_prefix(
    segments:tuple[CenterSegment,...],contact_time_s:float,
    *,
    fixed_dt_s:float,accepted:AppliedContactState
)->tuple[CenterSegment,...]:
    """Reject any event at/after collision; append immobile absorbing tail.

    This returned path is only a geometric timing certificate for gates and
    rewards; the physical response is the distinct AppliedContactState.
    """
    if (not segments or type(contact_time_s) not in (float,int)
        or not isfinite(contact_time_s) or not 0.<contact_time_s<=fixed_dt_s
        or abs(segments[0].start_time_s)>1e-10
        or abs(segments[-1].end_time_s-fixed_dt_s)>1e-10
        or any(abs(a.end_time_s-b.start_time_s)>1e-10 or
               (a.end-b.start).magnitude>1e-8
               for a,b in zip(segments,segments[1:]))):
        raise ValueError("complete chronological tick trajectory required")
    kept=[]
    for segment in segments:
        if segment.start_time_s>=contact_time_s:
            break
        if segment.end_time_s<=contact_time_s:
            kept.append(segment)
        else:
            pos=segment.position_at(contact_time_s)
            kept.append(CenterSegment(segment.start_time_s,contact_time_s,
                            segment.start,pos,segment.initial_velocity,
                            segment.gravity_mps2))
            break
    if not kept or (kept[-1].end-accepted.preimpact.position).magnitude>1e-6:
        raise ValueError("contact pose not on supplied source trajectory")
    if contact_time_s<fixed_dt_s:
        kept.append(CenterSegment(contact_time_s,fixed_dt_s,
                                  accepted.preimpact.position,accepted.preimpact.position))
    return tuple(kept)


def apply_bump_impulse_at_certified_clear_pose(
    pre:VehicleState,mass:RigidBodyImpactMass,*,wheel_local:Vec3,
    road:PlanarSurfacePatch,max_bump_m:float,
    first_clear_s:float,first_witness_s:float,
    maximum_road_gap_m:float=1e-6,
)->AppliedContactState:
    """Apply unilateral impact to a separately RK4-accepted clear prefix.

    The geometric bracket must have been witnessed on a rejected trial and
    independently replayed to its last known-clear instant. No extrapolated
    or overlapping state is accepted, and the original bump stop is intact.
    """
    from math import isfinite
    from .geometry.vector import Vec3
    if (not isinstance(pre,VehicleState) or not isinstance(mass,RigidBodyImpactMass)
        or not isinstance(wheel_local,Vec3) or not isinstance(road,PlanarSurfacePatch)
        or any(type(t) not in (int,float) or not isfinite(t) for t in
               (max_bump_m,first_clear_s,first_witness_s,maximum_road_gap_m))
        or not 0.<first_clear_s<first_witness_s
        or not 0.<max_bump_m or not 0.<maximum_road_gap_m<=1e-6):
        raise ValueError('uncertified accepted-prefix wheel stop event')
    n=road.normal
    if abs(n.magnitude-1.)>1e-8 or n.y<=0.:
        raise ValueError('road normal must be unit upward')
    r=pre.orientation.rotate(wheel_local)
    pt=pre.position+r
    gap=max_bump_m*n.y+(pt-road.origin).dot(n)
    if not 0.<=gap<=maximum_road_gap_m:
        raise ValueError(f'accepted prefix not at clear hard-stop boundary: gap_m={gap:.9g}')
    speed=(pre.velocity+pre.angular_velocity.cross(r)).dot(n)
    if speed>=-1e-6:
        raise ValueError('road hard-stop contact must be approaching')
    arm=r.cross(n)
    inv=mass.world_inertia_inverse(pre.orientation,arm)
    effective=1./mass.mass_kg+n.dot(inv.cross(r))
    if not isfinite(effective) or effective<=0.:
        raise ValueError('nonpositive road stop effective inverse mass')
    impulse=n*(-speed/effective)
    angular=r.cross(impulse)
    post_vel=pre.velocity+impulse/mass.mass_kg
    post_omega=pre.angular_velocity+mass.world_inertia_inverse(pre.orientation,angular)
    def kinetic(v,w):
        return .5*mass.mass_kg*v.dot(v)+.5*w.dot(mass.world_inertia(pre.orientation,w))
    before,after=kinetic(pre.velocity,pre.angular_velocity),kinetic(post_vel,post_omega)
    if after>before+1e-9*max(1.,before):
        raise ValueError('bump stop would generate kinetic energy')
    return _apply_mass_impulse(pre,mass,impulse,angular,before,after,max(0.,before-after),
        first_clear=first_clear_s,witness=first_witness_s,kind='wheel_hard_stop')
