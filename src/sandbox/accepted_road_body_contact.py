"""First source-bounded road-chassis contact and applied impulse candidate.

This is a truthful finite-rectangle first-contact solver for an explicitly
linear+SLERP pose interval with an initially CLEAR start. It will not invent
a first contact if the supplied source segment starts already overlapping.
Mainline 6-DOF RK4 must supply an accepted preimpact state/segment and must
separately approve its event-time split before publishing a scored terminal.
"""
from __future__ import annotations

from dataclasses import dataclass,replace

from .accepted_contact_transition import AppliedContactState,apply_certified_face_impulse,_apply_mass_impulse
from .body_collision import VehicleBodyGeometry,obb_separation_lower_bound,_slerp
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .rigid_body_impact_response import RigidBodyImpactMass,rigid_body_normal_impact
from .obb_feature_contact import closest_obb_features,closest_point
from .swept_surface_clearance import certify_first_bounded_road_contact
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState


@dataclass(frozen=True,slots=True)
class AcceptedRoadBodyContact:
    patch_id:str
    contact:AppliedContactState
    normal:Vec3
    version:str="wp3_bounded_road_first_impact_v1_dev"


def apply_first_road_body_contact(
    first:VehicleState,last:VehicleState,body:VehicleBodyGeometry,
    road_atlas:tuple,mass:RigidBodyImpactMass,*,duration_s:float,
    ballistic_gravity_mps2:float=0.,
)->AcceptedRoadBodyContact|None:
    from math import isfinite
    if (type(ballistic_gravity_mps2) not in (int,float)
        or not isfinite(ballistic_gravity_mps2) or ballistic_gravity_mps2<0.):
        raise ValueError('invalid airborne road-contact gravity')
    if ballistic_gravity_mps2:
        if (first.grounded or last.grounded
            or (last.position-first.position-first.velocity*duration_s+
                Vec3(0.,.5*ballistic_gravity_mps2*duration_s**2,0.)).magnitude>1e-7
            or (last.velocity-first.velocity+
                Vec3(0.,ballistic_gravity_mps2*duration_s,0.)).magnitude>1e-7):
            raise ValueError('airborne contact source does not follow exact gravity')
    witness=certify_first_bounded_road_contact(
        first,last,body,road_atlas,duration_s=duration_s,resolution_s=1e-6,
        ballistic_gravity_mps2=ballistic_gravity_mps2)
    if witness is None:return None
    patch_id,tclear,thit,_=witness
    patch=next(p for p in road_atlas[0] if p.id==patch_id)
    thin=OrientedBoxVolume(patch.origin,
        Vec3(patch.width/2.,1e-9,patch.length/2.),
        Quaternion.from_basis(patch.right,patch.normal,patch.forward))
    try:
        if ballistic_gravity_mps2:
            raise ValueError('airborne curved contact requires source-feature impulse')
        contact=apply_certified_face_impulse(first,last,body,thin,mass,duration_s=duration_s)
        if (contact.first_clear_s>thit or contact.witnessed_contact_s<tclear):
            raise ValueError("face impulse outside first road contact bracket")
        normal=patch.normal
    except ValueError:
        # Finite OBB road EDGE / car CORNER events cannot be described by
        # a face-only normal. Witness closest box features at the last
        # certified clear pose, and carry that same body feature to impact.
        def interpolate(t):
            f=t/duration_s
            return VehicleState(first.position+(last.position-first.position)*f+
                Vec3(0.,.5*ballistic_gravity_mps2*duration_s**2*f*(1.-f),0.),
                first.velocity+(last.velocity-first.velocity)*f,
                _slerp(first.orientation,last.orientation,f),
                first.angular_velocity+(last.angular_velocity-first.angular_velocity)*f,
                first.grounded and last.grounded)
        clear=interpolate(tclear)
        hit=interpolate(thit)
        box_clear=body.box_at(clear.position,clear.orientation)
        box_hit=body.box_at(hit.position,hit.orientation)
        pair=closest_obb_features(box_clear,thin)
        normal=(pair.body_point-pair.obstacle_point).normalized()
        inv_clear=Quaternion(clear.orientation.w,-clear.orientation.x,
                             -clear.orientation.y,-clear.orientation.z)
        body_local=inv_clear.rotate(pair.body_point-box_clear.center)
        point_hit=box_hit.center+hit.orientation.rotate(body_local)
        gap=(point_hit-closest_point(thin,point_hit)).magnitude
        travel=(hit.position-clear.position).magnitude
        # Bounding angular point movement by the full source-interval
        # rotation angle would be too coarse; use original short SLERP
        # angle scaled by the witnessed event time fraction.
        from math import acos
        q0,q1=first.orientation,last.orientation
        theta=2.*acos(max(0.,min(1.,abs(
            q0.w*q1.w+q0.x*q1.x+q0.y*q1.y+q0.z*q1.z))))
        max_motion=(travel+theta*body.motion_radius_m*((thit-tclear)/duration_s)
            +.5*ballistic_gravity_mps2*duration_s*(thit-tclear))
        if (gap>pair.separation_m+max_motion+1e-8 or
            (hit.velocity+hit.angular_velocity.cross(point_hit-hit.position)).dot(normal)>=-1e-6):
            raise ValueError("OBB edge witness has no consistent incoming contact")
        response=rigid_body_normal_impact(hit,body,mass,
            contact_point_world=point_hit,normal_obstacle_to_vehicle=normal,
            restitution=0.)
        contact=_apply_mass_impulse(hit,mass,response.impulse_n_s,
            response.angular_impulse_nms,response.before_kinetic_j,
            response.after_kinetic_j,response.energy_lost_j,
            first_clear=tclear,witness=thit,kind="bounded_road_body")
    if contact.impulse_n_s.dot(normal)<=0.:
        raise ValueError("road body impulse must oppose certified incoming normal")
    return AcceptedRoadBodyContact(patch_id,replace(contact,kind="bounded_road_body"),normal)
