"""Opt-in, conservative witnessed OBB EDGE-to-EDGE impact certificate.

This is not a fallback normal guess. A unique cross-product SAT axis must
dominate every other valid OBB separating axis at the actual first witnessed
contact, and both finite edge segments must have an interior closest pair.
The witness must be within 10 microns of contact. Anything else is INVALID.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

from .body_collision import VehicleBodyGeometry, _box_axes, _slerp, obb_separation_at_most
from .linear_first_body_contact import certify_first_linear_body_contact
from .geometry.vector import Vec3
from .rigid_body_impact_response import RigidBodyImpactMass, ImpactResponse, rigid_body_normal_impact
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState


@dataclass(frozen=True, slots=True)
class EdgeImpactCertificate:
    first_certified_clear_s: float
    first_witnessed_overlap_s: float
    normal_obstacle_to_car: Vec3
    car_contact_world: Vec3
    obstacle_contact_world: Vec3
    edge_indices: tuple[int, int]
    edge_parameter_fractions: tuple[float, float]
    distance_m: float
    uncertainty_s: float
    response: ImpactResponse
    schema_version: str = "wp3_unique_edge_edge_impact_v4_dev"


def _closest_interior_edges(a: OrientedBoxVolume, b: OrientedBoxVolume,
                            i: int, j: int, normal: Vec3):
    """Clamped nearest points on the extremal car/obstacle supporting edges."""
    aa,bb=_box_axes(a),_box_axes(b)
    ah=(a.half_extents.x,a.half_extents.y,a.half_extents.z)
    bh=(b.half_extents.x,b.half_extents.y,b.half_extents.z)
    ca=a.center
    cb=b.center
    for k in range(3):
        if k!=i:
            projection=normal.dot(aa[k])
            if abs(projection)<1e-8:
                raise ValueError("ambiguous car edge selection")
            ca=ca-aa[k]*(ah[k]*(1. if projection>0 else -1.))
        if k!=j:
            projection=normal.dot(bb[k])
            if abs(projection)<1e-8:
                raise ValueError("ambiguous obstacle edge selection")
            cb=cb+bb[k]*(bh[k]*(1. if projection>0 else -1.))
    p1,p2=ca-aa[i]*ah[i],ca+aa[i]*ah[i]
    q1,q2=cb-bb[j]*bh[j],cb+bb[j]*bh[j]
    u=p2-p1
    v=q2-q1
    w=p1-q1
    A,B,C=u.dot(u),u.dot(v),v.dot(v)
    D,E=u.dot(w),v.dot(w)
    denominator=A*C-B*B
    if denominator<=1e-12:
        raise ValueError("nearly parallel SAT edges cannot certify a contact point")
    s=max(0.,min(1.,(B*E-C*D)/denominator))
    t=max(0.,min(1.,(A*E-B*D)/denominator))
    for _ in range(8):
        s=max(0.,min(1.,(B*t-D)/A))
        t=max(0.,min(1.,(E+B*s)/C))
    if not 1e-6<s<1.-1e-6 or not 1e-6<t<1.-1e-6:
        raise ValueError("edge endpoint or edge-face requires separate contact proof")
    x,y=p1+u*s,q1+v*t
    return x,y,s,t


def certify_unique_linear_edge_impact(
    start: VehicleState, end: VehicleState, hull: VehicleBodyGeometry,
    obstacle: OrientedBoxVolume, mass: RigidBodyImpactMass,
    *, duration_s: float, resolution_s: float=1e-6,
    max_contact_distance_m: float=1e-5,
    minimum_axis_margin_m: float=1e-5,
    minimum_approach_speed_mps: float=1e-4,
) -> EdgeImpactCertificate:
    if (not all(type(x) in (int,float) and isfinite(x) for x in (
            duration_s,resolution_s,max_contact_distance_m,
            minimum_axis_margin_m,minimum_approach_speed_mps))
        or not 0.<resolution_s<=1e-5 or not 0.<duration_s
        or not 0.<max_contact_distance_m<=1e-5
        or not 0.<minimum_axis_margin_m<=1e-3
        or not 0.<minimum_approach_speed_mps):
        raise ValueError("edge impact positive finite bounded proof parameters required")
    hit=certify_first_linear_body_contact(
        start,end,hull,obstacle,duration_s=duration_s,
        resolution_s=resolution_s,max_refinements=22)
    if hit is None or not 0.<=hit.last_certified_clear_s<hit.first_witnessed_overlap_s<=duration_s:
        raise ValueError("edge impact missing first witnessed overlap bracket")
    ratio=hit.first_witnessed_overlap_s/duration_s
    from .accepted_contact_transition import _interpolate
    state=_interpolate(start,end,ratio)
    car=hull.box_at(state.position,state.orientation)
    if not obb_separation_at_most(car,obstacle,1e-9):
        raise ValueError("edge witness cannot be disjoint")
    ca,ob=_box_axes(car),_box_axes(obstacle)
    cha=(car.half_extents.x,car.half_extents.y,car.half_extents.z)
    oh=(obstacle.half_extents.x,obstacle.half_extents.y,obstacle.half_extents.z)
    delta=car.center-obstacle.center
    candidates=[]
    all_axes=[("car",i,-1,v) for i,v in enumerate(ca)]
    all_axes +=[("obstacle",-1,j,v) for j,v in enumerate(ob)]
    all_axes +=[("edge",i,j,a.cross(b)) for i,a in enumerate(ca) for j,b in enumerate(ob)]
    for feature,i,j,raw in all_axes:
        length=raw.magnitude
        if length<1e-8:continue
        axis=raw/length
        gap=(abs(delta.dot(axis))
             -sum(e*abs(axis.dot(v)) for e,v in zip(cha,ca,strict=True))
             -sum(e*abs(axis.dot(v)) for e,v in zip(oh,ob,strict=True)))
        candidates.append((gap,feature,i,j,axis))
    candidates.sort(key=lambda x:x[0],reverse=True)
    if (len(candidates)<2 or candidates[0][1]!="edge"
            or candidates[0][0] > max_contact_distance_m
            or candidates[0][0] < -max_contact_distance_m
            or candidates[0][0]-candidates[1][0]<minimum_axis_margin_m):
        raise ValueError("no uniquely dominant edge-edge SAT contact")
    gap,_,i,j,normal=candidates[0]
    if delta.dot(normal)<0:normal=normal*(-1.)
    x,y,s,t=_closest_interior_edges(car,obstacle,i,j,normal)
    distance=(x-y).magnitude
    if distance>max_contact_distance_m or abs((x-y).dot(normal)-gap)>max_contact_distance_m:
        raise ValueError("closest edge segments do not meet at witnessed TOI")
    if (state.velocity+state.angular_velocity.cross(x-state.position)).dot(normal)>=-minimum_approach_speed_mps:
        raise ValueError("edge impact normal is not approaching")
    response=rigid_body_normal_impact(
        state,hull,mass,contact_point_world=x,
        normal_obstacle_to_vehicle=normal,restitution=0.,
        face_tolerance_m=1e-6)
    if (response.energy_lost_j<=0.
            or response.after_kinetic_j>response.before_kinetic_j+1e-9
            or response.post_relative_normal_mps < -1e-7):
        raise ValueError("edge contact response fails passive collision audit")
    return EdgeImpactCertificate(
        hit.last_certified_clear_s,hit.first_witnessed_overlap_s,
        normal,x,y,(i,j),(s,t),distance,hit.uncertainty_s,response)


def apply_certified_edge_impulse(
    start: VehicleState, end: VehicleState, hull: VehicleBodyGeometry,
    obstacle: OrientedBoxVolume, mass: RigidBodyImpactMass, *,
    duration_s: float,
):
    """Apply the unique witnessed SAT edge impulse to accepted VehicleState."""
    from .accepted_contact_transition import _interpolate, _apply_mass_impulse
    cert=certify_unique_linear_edge_impact(
        start,end,hull,obstacle,mass,duration_s=duration_s)
    pre=_interpolate(start,end,cert.first_witnessed_overlap_s/duration_s)
    response=cert.response
    return _apply_mass_impulse(
        pre,mass,response.impulse_n_s,response.angular_impulse_nms,
        response.before_kinetic_j,response.after_kinetic_j,
        response.energy_lost_j,first_clear=cert.first_certified_clear_s,
        witness=cert.first_witnessed_overlap_s,kind="obstacle_edge")
