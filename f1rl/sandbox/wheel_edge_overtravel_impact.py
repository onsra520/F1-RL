"""Conservative first finite-tire sidewall contact at a road-edge overtravel.

A compressed wheel approaching an authored polygon FROM OUTSIDE may encounter
its physical finite tyre footprint before a vertical spring hard stop exists.
This is an independent development-only rigid sidewall impulse, never a
fictitious vertical bump-stop event or a scientific calibration.
"""
from __future__ import annotations
from math import hypot,isfinite
from .geometry.vector import Vec3
from .vehicle import VehicleState
from .track.patch import PlanarSurfacePatch
from .rigid_body_impact_response import RigidBodyImpactMass
from .accepted_contact_transition import _apply_mass_impulse
from .finite_tire_footprint_v4 import FOOTPRINT_RADIUS_M

def sidewall_edge(polygon, index, radius=FOOTPRINT_RADIUS_M):
    if (not isinstance(polygon,tuple) or len(polygon)<3 or
        type(index) is not int or not 0<=index<len(polygon) or
        type(radius) not in (float,int) or not 0.<radius<=.2):
        raise ValueError("source-bound finite tire polygon and edge required")
    a,b=polygon[index],polygon[(index+1)%len(polygon)]
    dx,dz=b[0]-a[0],b[1]-a[1]
    length=hypot(dx,dz)
    if length<=1e-9:raise ValueError("degenerate road edge")
    return Vec3(dz/length,0.,-dx/length),length

def entry_margin(pose,wheel_local,polygon,index,*,radius=FOOTPRINT_RADIUS_M):
    outward,_=sidewall_edge(polygon,index,radius)
    a=polygon[index]
    p=pose.position+pose.orientation.rotate(wheel_local)
    return radius-(p.x-a[0])*outward.x-(p.z-a[1])*outward.z

def incoming_edge(before,after,wheel_local,road,polygon,*,radius=FOOTPRINT_RADIUS_M,max_bump_m):
    """Find exactly one newly entered finite-footprint edge with overtravel.

    Other edges must remain inside throughout source chord, so no corner or
    ambiguous simultaneous contact is licensed.
    """
    if (not isinstance(before,VehicleState) or not isinstance(after,VehicleState)
        or not isinstance(road,PlanarSurfacePatch) or
        not isinstance(wheel_local,Vec3) or
        not 0.<max_bump_m<=.5 or not before.grounded or not after.grounded):
        raise ValueError("unregistered finite-tire sidewall event source")
    edges=[]
    for i in range(len(polygon)):
        a=entry_margin(before,wheel_local,polygon,i,radius=radius)
        b=entry_margin(after,wheel_local,polygon,i,radius=radius)
        if a<=0.<b:
            edges.append(i)
        elif a<=0. or b<=0.:
            # Other simultaneous outside edge -> corner, NOT unique.
            raise ValueError("corner or nonentering wheel edge is not unique")
    if len(edges)!=1:raise ValueError("no unique footprint sidewall entry")
    idx=edges[0]
    n,_=sidewall_edge(polygon,idx,radius)
    wp=before.position+before.orientation.rotate(wheel_local)
    wp2=after.position+after.orientation.rotate(wheel_local)
    def vert_gap(p):
        return max_bump_m*road.normal.y+(p-road.origin).dot(road.normal)
    if not (vert_gap(wp)<-1e-9 and vert_gap(wp2)<-1e-9):
        raise ValueError("not a source-witnessed below-hard-stop sidewall entry")
    for pose in (before,after):
        arm=pose.orientation.rotate(wheel_local)
        speed=(pose.velocity+pose.angular_velocity.cross(arm)).dot(n)
        if speed>=-1e-6:
            raise ValueError("sidewall contact must be monotonically approaching")
    return idx

def apply_edge_overtravel_impulse_at_prefix(
    pre,mass,*,wheel_local,road,polygon,edge_index,
    max_bump_m,first_clear_s,first_witness_s,
    radius=FOOTPRINT_RADIUS_M
):
    """Apply impulse to last accepted RK4 prefix adjacent to first entry.

    Requires exact edge index and <=1 micron entry bracket. Momentum and
    rigid kinetic energy are recomputed by the shared accepted-state auditor.
    """
    if (not isinstance(pre,VehicleState) or not isinstance(mass,RigidBodyImpactMass)
        or not isinstance(wheel_local,Vec3) or not isinstance(road,PlanarSurfacePatch)
        or not all(type(t) in (int,float) and isfinite(t)
            for t in (max_bump_m,first_clear_s,first_witness_s))
        or not 0.<first_clear_s<first_witness_s
        or first_witness_s-first_clear_s>1.0001e-10
        or not pre.grounded):
        raise ValueError("unaccepted sidewall source time or mass")
    n,_=sidewall_edge(polygon,edge_index,radius)
    clearance=entry_margin(pre,wheel_local,polygon,edge_index,radius=radius)
    if not -1e-6<=clearance<=1e-9:
        raise ValueError(f"not the last certified outside edge prefix: {clearance:.9g}")
    arm=pre.orientation.rotate(wheel_local)
    point=pre.position+arm
    vertical=max_bump_m*road.normal.y+(point-road.origin).dot(road.normal)
    if vertical>=-1e-9:
        raise ValueError("vertical hard stop was not exceeded at sidewall")
    incoming=(pre.velocity+pre.angular_velocity.cross(arm)).dot(n)
    if incoming>=-1e-6:
        raise ValueError("nonapproaching sidewall cannot absorb momentum")
    tangent=arm.cross(n)
    inv=mass.world_inertia_inverse(pre.orientation,tangent)
    denom=1./mass.mass_kg+n.dot(inv.cross(arm))
    if not isfinite(denom) or denom<=0.:
        raise ValueError("sidewall effective mass invalid")
    impulse=n*(-incoming/denom)
    angular=arm.cross(impulse)
    vafter=pre.velocity+impulse/mass.mass_kg
    wafter=pre.angular_velocity+mass.world_inertia_inverse(pre.orientation,angular)
    def ke(v,w):
        return .5*mass.mass_kg*v.dot(v)+.5*w.dot(mass.world_inertia(pre.orientation,w))
    before,after=ke(pre.velocity,pre.angular_velocity),ke(vafter,wafter)
    if after>=before or after>before+1e-9*max(1.,before):
        raise ValueError("sidewall contact cannot generate kinetic energy")
    if (vafter+wafter.cross(arm)).dot(n)<-1e-7:
        raise ValueError("unilateral edge normal response still penetrating")
    return _apply_mass_impulse(
        pre,mass,impulse,angular,before,after,before-after,
        first_clear=first_clear_s,witness=first_witness_s,
        kind="wheel_edge_hard_stop")
