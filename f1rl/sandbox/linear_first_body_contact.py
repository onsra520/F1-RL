"""Continuous first-witness collision on a linear/ballistic+SLERP trajectory.

Unlike a repeated-prefix sweep, this searches the first potentially contacting
time interval left-to-right with a conservative spatial motion bound and can
resolve a first witnessed overlap to a bounded timestamp. Ambiguous grazing
is rejected, NEVER scored. The caller must certify that its source motion is
the stated linear or exact-gravity/SLERP segment before accepting evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import acos,isfinite

from .body_collision import VehicleBodyGeometry,_slerp,obb_separation_lower_bound
from .geometry.vector import Vec3
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState


@dataclass(frozen=True,slots=True)
class LinearFirstContactWitness:
    last_certified_clear_s:float
    first_witnessed_overlap_s:float
    gap_checks:int
    version:str="wp3_first_linear_slerp_obb_witness_v1"

    @property
    def uncertainty_s(self)->float:
        return self.first_witnessed_overlap_s-self.last_certified_clear_s


def certify_first_linear_body_contact(
    start:VehicleState,end:VehicleState,body:VehicleBodyGeometry,
    obstacle:OrientedBoxVolume,*,duration_s:float,
    resolution_s:float=1e-6,max_refinements:int=20,
    ballistic_gravity_mps2:float=0.,
)->LinearFirstContactWitness|None:
    if (not isinstance(start,VehicleState) or not isinstance(end,VehicleState)
        or not isinstance(body,VehicleBodyGeometry)
        or not isinstance(obstacle,OrientedBoxVolume)):
        raise ValueError("typed continuous collision geometry required")
    if (type(duration_s) not in (int,float) or not isfinite(duration_s) or duration_s<=0
        or type(ballistic_gravity_mps2) not in (int,float)
        or not isfinite(ballistic_gravity_mps2) or ballistic_gravity_mps2<0.
        or type(resolution_s) not in (int,float) or not isfinite(resolution_s) or resolution_s<=0
        or type(max_refinements) is not int or not 1<=max_refinements<=24):
        raise ValueError("finite positive collision interval/resolution bounds")
    q0,q1=start.orientation,end.orientation
    cosine=min(1.,max(0.,abs(q0.w*q1.w+q0.x*q1.x+q0.y*q1.y+q0.z*q1.z)))
    rotation_arc=2.*acos(cosine)
    translation=(end.position-start.position).magnitude
    # A parabola is above its endpoint chord by g*h^2*f*(1-f)/2.
    # Its derivative in normalized time is bounded by g*h^2/2, so the
    # search's midpoint non-contact bound remains conservative.
    motion_bound=(translation+rotation_arc*body.motion_radius_m+
                  .5*ballistic_gravity_mps2*duration_s*duration_s)
    checks=0
    def gap(fraction:float)->float:
        nonlocal checks
        checks+=1
        pose=(start.position+(end.position-start.position)*fraction+
              Vec3(0.,.5*ballistic_gravity_mps2*duration_s**2*fraction*(1.-fraction),0.))
        orientation=_slerp(q0,q1,fraction)
        return obb_separation_lower_bound(body.box_at(pose,orientation),obstacle)
    if gap(0.)<=1e-10:
        raise ValueError("first contact search starts in an overlapping state")
    def search(lo:float,hi:float,remaining:int):
        center=(lo+hi)/2.
        if gap(center)>motion_bound*(hi-lo)/2.+1e-10:
            return None
        if (hi-lo)*duration_s<=resolution_s:
            left,right=gap(lo),gap(hi)
            if left>1e-10 and right<=1e-10:
                return lo,hi
            # A repeated end-clear / grazing interval cannot be declared a
            # valid certified terminal; require witnessed penetration.
            if left>1e-10 and right>1e-10:
                if gap(center)<=1e-10:
                    raise ValueError("contact/grazing smaller than certified interval")
                return None
            if left<=1e-10 and right<=1e-10:
                raise ValueError("first contact lies before unresolved interval")
            raise ValueError("first linear impact ordering uncertain")
        if remaining==0:
            raise ValueError("first linear contact could not be resolved")
        first=search(lo,center,remaining-1)
        if first is not None:
            return first
        return search(center,hi,remaining-1)
    interval=search(0.,1.,max_refinements)
    if interval is None:
        return None
    lo,hi=interval
    if not 0<=lo<hi<=1. or (hi-lo)*duration_s>resolution_s+1e-12:
        raise ValueError("invalid collision timestamp bracket")
    return LinearFirstContactWitness(lo*duration_s,hi*duration_s,checks)
