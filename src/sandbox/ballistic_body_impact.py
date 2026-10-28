"""Fail-closed first witnessed impact on a *known ballistic parabola*.

Engineering diagnostic primitive ONLY: this is not a collision response and
cannot create valid terminal outcomes. Unlike a linear+SLERP chord sweep, the
center follows exact constant-acceleration kinematics. A conservative bound
around each midpoint certifies separation of the entire time interval;
otherwise it subdivides left-first. Ambiguous grazing remains BLOCKED.

Assumptions: no external torque during ballistic interval, shortest-arc SLERP
orientation between certified endpoints; acceleration is a trusted explicit
constant in world coordinates. No road/obstacle response inside interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos,isfinite

from .body_collision import VehicleBodyGeometry,_slerp,obb_separation_lower_bound
from .geometry.vector import Vec3
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState

@dataclass(frozen=True,slots=True)
class BallisticImpactBracket:
    last_proven_clear_s:float
    first_witnessed_contact_by_s:float
    gap_checks:int
    position_at_witness:Vec3
    version:str="ballistic_body_first_witness_v1"

    @property
    def uncertainty_s(self)->float:
        return self.first_witnessed_contact_by_s-self.last_proven_clear_s


def first_ballistic_body_impact(
    previous:VehicleState,
    candidate:VehicleState,
    geometry:VehicleBodyGeometry,
    obstacle:OrientedBoxVolume,
    *,
    duration_s:float,
    acceleration_world:Vec3,
    max_refinements:int=18,
    time_tolerance_s:float=1e-6,
) -> BallisticImpactBracket|None:
    if not all(isinstance(v,t) for v,t in (
        (previous,VehicleState),(candidate,VehicleState),
        (geometry,VehicleBodyGeometry),(obstacle,OrientedBoxVolume),
        (acceleration_world,Vec3))):
        raise ValueError("explicit vehicle, body, obstacle and ballistic acceleration required")
    if (type(duration_s) not in (int,float) or not isfinite(duration_s) or duration_s<=0.
        or type(max_refinements) is not int or not 1<=max_refinements<=20
        or type(time_tolerance_s) not in (int,float) or not isfinite(time_tolerance_s)
        or time_tolerance_s<=0.):
        raise ValueError("finite positive ballistic time/refinement bounds required")
    expected=(previous.position+previous.velocity*duration_s+
              acceleration_world*(.5*duration_s*duration_s))
    tolerance=1e-7*(1.+expected.magnitude)
    if (candidate.position-expected).magnitude>tolerance:
        raise ValueError("ballistic endpoint does not match registered acceleration")
    dot=abs(sum(getattr(previous.orientation,k)*getattr(candidate.orientation,k)
                for k in ("w","x","y","z")))
    angular_arc=2.*acos(min(1.,max(0.,dot)))
    rotational_rate_bound=angular_arc/duration_s
    gap_checks=0
    def position(t:float)->Vec3:
        return (previous.position+previous.velocity*t+
                acceleration_world*(.5*t*t))
    def orientation(t:float):
        return _slerp(previous.orientation,candidate.orientation,t/duration_s)
    def gap(t:float)->float:
        nonlocal gap_checks
        gap_checks+=1
        return obb_separation_lower_bound(
            geometry.box_at(position(t),orientation(t)),obstacle)
    def separated_interval(left:float,right:float)->bool:
        m=(left+right)/2.
        half=(right-left)/2.
        center_velocity=previous.velocity+acceleration_world*m
        # A rigid hull point displacement about midpoint is bounded by:
        # linear tangent displacement + parabola sagitta + SLERP angular arc.
        motion=(center_velocity.magnitude*half+
                .5*acceleration_world.magnitude*half*half+
                rotational_rate_bound*half*geometry.motion_radius_m)
        return gap(m)>motion+1e-10
    if gap(0.)<=1e-10:
        raise ValueError("ballistic interval starts from already overlapping body")
    # Right endpoint may be clear even when trajectory intersects in midair.
    # Need left-to-right *interval* certification, not endpoint-only testing.
    def search(left:float,right:float,depth:int):
        if separated_interval(left,right):
            return None
        if right-left<=time_tolerance_s:
            if gap(right)<=1e-10 and gap(left)>1e-10:
                return left,right
            # Exact contact may be grazing with no penetration. Reject unknown.
            raise ValueError("ballistic first contact cannot be certified within tolerance")
        if depth>=max_refinements:
            raise ValueError("ballistic first impact unresolvable at refinement limit")
        mid=(left+right)/2.
        first=search(left,mid,depth+1)
        if first is not None:
            return first
        return search(mid,right,depth+1)
    bracket=search(0.,float(duration_s),0)
    if bracket is None:return None
    lo,hi=bracket
    if not 0.<=lo<hi<=duration_s or hi-lo>time_tolerance_s+1e-12:
        raise ValueError("ballistic contact bracket out of bounds")
    return BallisticImpactBracket(lo,hi,gap_checks,position(hi))
