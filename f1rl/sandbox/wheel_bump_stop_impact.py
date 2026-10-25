"""Unilateral rigid chassis/road bump-stop impact certificate (scratch physics).

Explicitly models the wheel's hard suspension stop as a normal road impulse
at the wheel anchor on the chassis. The step must START above the stop and
CROSS it on a monotone pose segment. This is a bounded diagnostic event,
not a globally calibrated tire/contact model or registered Eval terminal.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from .body_collision import _slerp
from .geometry.vector import Vec3
from .rigid_body_impact_response import RigidBodyImpactMass
from .track.patch import PlanarSurfacePatch
from .vehicle import VehicleState


@dataclass(frozen=True,slots=True)
class WheelBumpStopCertificate:
    first_clear_time_s:float
    first_violating_time_s:float
    contact_point_world:Vec3
    road_normal:Vec3
    impulse_n_s:Vec3
    angular_impulse_nms:Vec3
    kinetic_before_j:float
    kinetic_after_j:float
    dissipated_kinetic_j:float
    normal_speed_before_mps:float
    normal_speed_after_mps:float
    version:str="wheel_bump_stop_contact_impulse_v1"


def certify_wheel_bump_stop(
    before:VehicleState, after:VehicleState, *,
    wheel_local:Vec3, road:PlanarSurfacePatch, max_bump_m:float,
    mass:RigidBodyImpactMass, duration_s:float,
    restitution:float=0., resolution_s:float=1e-6
)->WheelBumpStopCertificate:
    if (not isinstance(before,VehicleState) or not isinstance(after,VehicleState)
        or not isinstance(wheel_local,Vec3) or not isinstance(road,PlanarSurfacePatch)
        or not isinstance(mass,RigidBodyImpactMass)):
        raise ValueError("typed road/wheel/vehicle/mass required")
    fields=(max_bump_m,duration_s,resolution_s,restitution)
    if (any(type(x) not in (int,float) or not isfinite(x) for x in fields)
        or max_bump_m<=0. or duration_s<=0. or resolution_s<=0.
        or duration_s<resolution_s or not 0.<=restitution<=1.):
        raise ValueError("invalid wheel-stop contact domain")
    # Geometry is interpolated on the caller-provided model's accepted
    # short rigid segment. It is not a claim of exact RK4 ODE root solving.
    def pose(t:float)->VehicleState:
        fraction=t/duration_s
        return VehicleState(
            before.position+(after.position-before.position)*fraction,
            before.velocity+(after.velocity-before.velocity)*fraction,
            _slerp(before.orientation,after.orientation,fraction),
            before.angular_velocity+(after.angular_velocity-before.angular_velocity)*fraction,
            before.grounded and after.grounded)
    normal=road.normal
    if abs(normal.magnitude-1.)>1e-8 or normal.y<=0.:
        raise ValueError("authored road normal must be unit upward")
    # Road plane equation expressed via the patch's origin and upward normal:
    # gap = d*n_y + (wheel_pos - origin) dot n.
    def gap(t:float)->float:
        s=pose(t)
        pt=s.position+s.orientation.rotate(wheel_local)
        return max_bump_m*normal.y+(pt-road.origin).dot(normal)
    g0,g1=gap(0.),gap(duration_s)
    # RK4's separately accepted pose can be within 0.1 nm of an ideal
    # hard-stop boundary. Distinguish this bounded numerical contact from
    # any macroscopic existing penetration: the rejected endpoint must be
    # strictly beyond 1 nm, with approaching velocity proved below.
    if not (-1e-9<=g0 and g1<-1e-9):
        raise ValueError(f"hard stop must be crossed from certified clear/touching state: g0={g0:.14g},g1={g1:.14g},dt={duration_s:.14g}")
    # Physically monotonic only if both wheel anchor velocities are approaching
    # the ROAD plane, rejecting ambiguous oscillation/recontact.
    for s in (before,after):
        arm=s.orientation.rotate(wheel_local)
        speed=(s.velocity+s.angular_velocity.cross(arm)).dot(normal)
        if speed>=-1e-6:
            raise ValueError("wheel anchor is not monotonically approaching stop")
    lo,hi=0.,duration_s
    for _ in range(64):
        if hi-lo<=resolution_s:break
        mid=(lo+hi)/2.
        if gap(mid)>0.:
            lo=mid
        else:hi=mid
    if gap(lo)<-1e-7 or gap(hi)>1e-7 or hi-lo>resolution_s:
        raise ValueError("hard-stop time bracket unresolved")
    s=pose(hi)
    point=s.position+s.orientation.rotate(wheel_local)
    r=point-s.position
    pre_speed=(s.velocity+s.angular_velocity.cross(r)).dot(normal)
    if pre_speed>=-1e-6:
        raise ValueError("certified wheel stop has no incoming velocity")
    cross=r.cross(normal)
    inv_i=mass.world_inertia_inverse(s.orientation,cross)
    denominator=1./mass.mass_kg+normal.dot(inv_i.cross(r))
    if not isfinite(denominator) or denominator<=0.:
        raise ValueError("effective wheel/road mass invalid")
    scalar=-(1.+restitution)*pre_speed/denominator
    impulse=normal*scalar
    torque=r.cross(impulse)
    v_after=s.velocity+impulse/mass.mass_kg
    w_after=s.angular_velocity+mass.world_inertia_inverse(s.orientation,torque)
    def kinetic(v:Vec3,w:Vec3)->float:
        return .5*mass.mass_kg*v.dot(v)+.5*w.dot(mass.world_inertia(s.orientation,w))
    e0,e1=kinetic(s.velocity,s.angular_velocity),kinetic(v_after,w_after)
    new_speed=(v_after+w_after.cross(r)).dot(normal)
    if (e1>e0+1e-7*max(1.,e0) or
        abs(new_speed+restitution*pre_speed)>1e-7*max(1.,abs(pre_speed))
        or not isfinite(e1)):
        raise ValueError("wheel stop restitution/energy conservation violated")
    return WheelBumpStopCertificate(
        lo,hi,point,normal,impulse,torque,e0,e1,max(0.,e0-e1),
        pre_speed,new_speed)
