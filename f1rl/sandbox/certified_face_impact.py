"""Conservative rigid-body face-impact diagnostic with bounded TOI.

This combines a first *witnessed overlap* bracket with an explicit obstacle
face and car hull point; accepts only an unambiguous face-corner witness.
A vertex/edge manifold without a unique normal is rejected rather than
making a scored crash. The result is a diagnostic, NOT an official terminal.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .body_collision import VehicleBodyGeometry, _box_axes, _slerp, obb_separation_at_most
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .linear_first_body_contact import certify_first_linear_body_contact
from .rigid_body_impact_response import RigidBodyImpactMass, ImpactResponse, rigid_body_normal_impact
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState


@dataclass(frozen=True,slots=True)
class FaceImpactCertificate:
    first_certified_clear_s:float
    first_witnessed_overlap_s:float
    obstacle_face_normal:Vec3
    car_hull_contact_world:Vec3
    certified_contact_time_uncertainty_s:float
    response:ImpactResponse
    schema_version:str="wp3_face_impact_diagnostic_v1"


def certify_linear_face_impact(
    start:VehicleState,
    end:VehicleState,
    hull:VehicleBodyGeometry,
    obstacle:OrientedBoxVolume,
    mass:RigidBodyImpactMass,
    *,
    duration_s:float,
    resolution_s:float=1e-6,
    minimum_approach_speed_mps:float=1e-4,
)->FaceImpactCertificate:
    if (type(minimum_approach_speed_mps) not in (int,float)
        or not isfinite(minimum_approach_speed_mps)
        or minimum_approach_speed_mps<=0):
        raise ValueError("positive approach-speed floor required")
    hit=certify_first_linear_body_contact(
        start,end,hull,obstacle,duration_s=duration_s,
        resolution_s=resolution_s,max_refinements=22)
    if hit is None:
        raise ValueError("no witnessed obstacle impact on the supplied interval")
    t=hit.first_witnessed_overlap_s/duration_s
    q=_slerp(start.orientation,end.orientation,t)
    # Interpolated state is used ONLY on an explicitly certified linear + SLERP
    # segment, not a ballistic parabola or an RK4 curvilinear segment.
    pos=start.position+(end.position-start.position)*t
    vel=start.velocity+(end.velocity-start.velocity)*t
    ang=start.angular_velocity+(end.angular_velocity-start.angular_velocity)*t
    state=VehicleState(pos,vel,q,ang,start.grounded and end.grounded)
    body=hull.box_at(pos,q)
    if not obb_separation_at_most(body,obstacle,1e-9):
        raise ValueError("event witness not overlapping on the physical obstacle")
    delta=body.center-obstacle.center
    normals=_box_axes(obstacle)
    extents=(obstacle.half_extents.x,obstacle.half_extents.y,obstacle.half_extents.z)
    axes=_box_axes(body)
    h=(body.half_extents.x,body.half_extents.y,body.half_extents.z)
    options=[]
    # A face normal is valid only if the geometrically extremal CAR vertex
    # lies near the OBSTACLE plane AND within the obstacle's face interior.
    # Edge-on-edge and ambiguous plane choices fail closed.
    for i,(axis,e) in enumerate(zip(normals,extents,strict=True)):
        sign=1. if delta.dot(axis)>=0. else -1.
        normal=axis*sign
        support=body.center
        for car_axis,half in zip(axes,h,strict=True):
            projection=normal.dot(car_axis)
            support=support-car_axis*(half*(1. if projection>=0 else -1.))
        relative=support-obstacle.center
        face_distance=abs(relative.dot(normal)-e)
        tangent=[abs(relative.dot(a)) for j,a in enumerate(normals) if j!=i]
        tangent_extents=[b for j,b in enumerate(extents) if j!=i]
        if (face_distance<=max(1e-5,5e-4*minimum_approach_speed_mps) and
            all(a<=b-1e-6 for a,b in zip(tangent,tangent_extents,strict=True))):
            approach=(vel+ang.cross(support-pos)).dot(normal)
            if approach < -minimum_approach_speed_mps:
                options.append((face_distance,normal,support))
    if len(options)!=1:
        raise ValueError(f"ambiguous/non-face contact: {len(options)} eligible normals")
    _,n,contact=options[0]
    response=rigid_body_normal_impact(
        state,hull,mass,contact_point_world=contact,
        normal_obstacle_to_vehicle=n,restitution=0.,
        face_tolerance_m=1e-6)
    if (response.energy_lost_j<0. or
        response.after_kinetic_j>response.before_kinetic_j+1e-7 or
        response.pre_relative_normal_mps>=-minimum_approach_speed_mps or
        response.post_relative_normal_mps< -1e-7):
        raise ValueError("impact impulse/kinetic-energy check failed")
    return FaceImpactCertificate(
        hit.last_certified_clear_s,hit.first_witnessed_overlap_s,
        n,contact,hit.uncertainty_s,response)
