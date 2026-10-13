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
