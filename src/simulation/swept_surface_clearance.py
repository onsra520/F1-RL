"""Conservative full-chassis swept clearance against bounded planar road patches.

This geometry certificate tests the *body*, not wheel-contact validity. It
supports ramps and overhangs by comparing 3D OBB vertices with each patch
plane only where swept XZ footprints may overlap that patch. Insufficient
resolution rejects the candidate. A parabola contributes g*dt**2/8 sagitta.
"""
from __future__ import annotations

from math import acos, isfinite
from typing import Callable

from .body_collision import VehicleBodyGeometry
from .independent_suspension_dynamics import SuspensionResolutionError
from .multi_surface_support import _body_corners, _intersection
from .road_footprint import _inflate, body_xz_polygon
from .vehicle import VehicleState


def certify_swept_chassis_clearance(
    pose_at: Callable[[float], VehicleState],
    start_s: float,
    end_s: float,
    body: VehicleBodyGeometry,
    atlas: tuple,
    *,
    max_subdivisions: int = 12,
    gravity_mps2: float = 0.,
) -> int:
    """Certify over prescribed linear+SLERP or exact ballistic pose_at.

    For ballistic time sampling, gravitational curvature between endpoints is
    bounded by g*h²/8. Every movement bound is over-conservative, including
    projected XZ hull and plane-normal clearance. A witness or unresolved
    leaf raises SuspensionResolutionError (never marks collision as scored).
    """
    if not isinstance(body, VehicleBodyGeometry) or not callable(pose_at):
        raise ValueError("explicit body and pose sampler required")
    if (not isfinite(start_s) or not isfinite(end_s)
            or start_s < 0. or end_s < start_s):
        raise ValueError("invalid sweep time interval")
    if type(max_subdivisions) is not int or not 0 <= max_subdivisions <= 20:
        raise ValueError("invalid sweep subdivision budget")
    if (type(gravity_mps2) not in (float, int)
            or not isfinite(gravity_mps2) or gravity_mps2 < 0.):
        raise ValueError("invalid ballistic gravity")
    patches, polygons = atlas
    if len(patches) != len(polygons):
        raise ValueError("road atlas shape mismatch")
    visits = 0

    def witness(state: VehicleState):
        for patch, poly in zip(patches, polygons, strict=True):
            hull = body_xz_polygon(state, body)
            if not _intersection(hull, poly):
                continue
            clearance = min(
                (v - patch.origin).dot(patch.normal) for v in _body_corners(state, body)
            )
            if clearance < -1e-9:
                raise SuspensionResolutionError(
                    "swept chassis witnessed road penetration"
                )

    def visit(lo: float, hi: float, depth: int):
        nonlocal visits
        visits += 1
        a, b = pose_at(lo), pose_at(hi)
        witness(a)
        witness(b)
        half = (lo + hi) / 2.
        mid = pose_at(half)
        witness(mid)
        delta = b.position - a.position
        q0, q1 = a.orientation, b.orientation
        dot = abs(q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z)
        angle = 2.*acos(min(1., max(0., dot)))
        # Upper bound of distance from midpoint to any swept body point
        # under linear position, shortest-arc rotation and ballistic sagitta.
        bound = .5*(delta.magnitude + angle*body.motion_radius_m)
        bound += gravity_mps2*(hi-lo)**2/8.
        envelope = _inflate(body_xz_polygon(mid, body), bound)
        corners = _body_corners(mid, body)
        clear = True
        for patch, poly in zip(patches, polygons, strict=True):
            if not _intersection(envelope, poly):
                continue
            separation = min((v-patch.origin).dot(patch.normal) for v in corners)
            if separation <= bound+1e-9:
                clear = False
                break
        if clear:
            return
        if depth == 0 or half == lo or half == hi:
            raise SuspensionResolutionError(
                "swept chassis/road clearance unresolved at subdivision limit"
            )
        visit(lo, half, depth-1)
        visit(half, hi, depth-1)

    visit(start_s, end_s, max_subdivisions)
    return visits


def certify_swept_chassis_patch_intersection_free(
    pose_at: Callable[[float], VehicleState],
    start_s:float,end_s:float,body:VehicleBodyGeometry,atlas:tuple,
    *,max_subdivisions:int=14,gravity_mps2:float=0.,
)->int:
    """Check exact finite oriented PATCH surfaces rather than extrapolated infinite planes.

    The old corner-plane witness was not sufficient at overhang: the lowest
    body vertex can be *outside* a narrow road patch. We instead separate
    OBB(body) and a bounded road rectangle by all 15 SAT axes. A thickness
    of 1e-9m is numerical conservatism, NOT penetration tolerance widening.
    Each midpoint separation also bounds every point in the interval using
    linear+SLERP and specified ballistic sagitta. Unresolved stays invalid.
    """
    from .body_collision import obb_separation_lower_bound
    from .geometry.rotation import Quaternion
    from .geometry.vector import Vec3
    from .track.volume import OrientedBoxVolume
    if not callable(pose_at) or not isinstance(body,VehicleBodyGeometry):
        raise ValueError("finite patch clearance requires typed pose and body")
    if (type(max_subdivisions) is not int or not 0<=max_subdivisions<=20
        or any(type(v) not in (int,float) or not isfinite(v) for v in
               (start_s,end_s,gravity_mps2))
        or not 0<=start_s<=end_s or gravity_mps2<0.):
        raise ValueError("invalid time/acceleration/subdivision for bounded patches")
    if len(atlas)!=2 or len(atlas[0])!=len(atlas[1]):
        raise ValueError("bounded authored atlas required")
    obstacles=[]
    for patch in atlas[0]:
        pose=Quaternion.from_basis(patch.right,patch.normal,patch.forward)
        obstacles.append(OrientedBoxVolume(patch.origin,
            Vec3(patch.width/2.,1e-9,patch.length/2.),pose))
    if not obstacles:raise ValueError("no bounded road surfaces")
    samples=0
    def gap(state:VehicleState)->float:
        nonlocal samples
        samples+=1
        box=body.box_at(state.position,state.orientation)
        return min(obb_separation_lower_bound(box,patch) for patch in obstacles)
    def visit(left:float,right:float,budget:int)->None:
        if right<=left:
            if gap(pose_at(left))<=1e-9:
                raise SuspensionResolutionError("bounded chassis/road patch overlap")
            return
        a=pose_at(left)
        b=pose_at(right)
        middle=(left+right)/2.
        m=pose_at(middle)
        # A witnessed bounded road/vehicle intersection is physical; an
        # overhanging body corner outside the patch is NOT an intersection.
        for state in (a,m,b):
            if gap(state)<=1e-9:
                raise SuspensionResolutionError("bounded chassis witnessed physical road contact")
        angle=2.*acos(min(1.,max(0.,abs(
            a.orientation.w*b.orientation.w+a.orientation.x*b.orientation.x+
            a.orientation.y*b.orientation.y+a.orientation.z*b.orientation.z))))
        bound=.5*((b.position-a.position).magnitude+angle*body.motion_radius_m)
        bound+=gravity_mps2*(right-left)*(right-left)/8.
        if gap(m)>bound+1e-9:
            return
        if budget==0 or middle==left or middle==right:
            raise SuspensionResolutionError("bounded patch/body sweep unresolved at refinement limit")
        visit(left,middle,budget-1)
        visit(middle,right,budget-1)
    visit(start_s,end_s,max_subdivisions)
    return samples


def certify_first_bounded_road_contact(
    before:VehicleState,after:VehicleState,body:VehicleBodyGeometry,atlas:tuple,
    *,duration_s:float,resolution_s:float=1e-6,
    ballistic_gravity_mps2:float=0.,
):
    """First finite road/body witness on linear or exact-gravity/SLERP source.

    Returns (patch_id, interval_lower_s, interval_upper_s, number_of_checks).
    Unknown or initially overlapping geometry raises, never becomes a scored
    crash. The caller must separately certify the dynamic segment itself.
    """
    from .linear_first_body_contact import certify_first_linear_body_contact
    from .geometry.rotation import Quaternion
    from .geometry.vector import Vec3
    from .track.volume import OrientedBoxVolume
    if len(atlas)!=2 or not atlas[0]:
        raise ValueError("nonempty authored road atlas required")
    collisions=[]
    for patch in atlas[0]:
        obstacle=OrientedBoxVolume(patch.origin,
            Vec3(patch.width/2.,1e-9,patch.length/2.),
            Quaternion.from_basis(patch.right,patch.normal,patch.forward))
        hit=certify_first_linear_body_contact(before,after,body,obstacle,
               duration_s=duration_s,resolution_s=resolution_s,max_refinements=24,
               ballistic_gravity_mps2=ballistic_gravity_mps2)
        if hit is not None:
            collisions.append((hit.last_certified_clear_s,patch.id,
                               hit.first_witnessed_overlap_s,hit.gap_checks))
    if not collisions:return None
    first=min(collisions)
    if any(other[0] < first[2] and other!=first
           for other in collisions):
        raise ValueError("multiple road patches have unresolved first-contact order")
    return first[1],first[0],first[2],first[3]
