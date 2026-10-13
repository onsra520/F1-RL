"""Prospective continuous rigid-body + four-point wheel support on ONE planar ramp.

This deliberately narrow implementation does not replace suspension,
tire contact mechanics, airborne/landing events, or multi-surface transitions.
It can certify a single planar road patch under linear translation and
shortest-arc SLERP, using conservative derivative bounds between samples.

No legacy vehicle geometry or wheel dimensions are inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite

from .body_collision import VehicleBodyGeometry, _slerp
from .geometry.vector import Vec3
from .track.patch import PlanarSurfacePatch
from .vehicle import VehicleState

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class WheelContactPoints:
    """Four explicitly supplied local tire-ground reference points."""
    front_left: Vec3
    front_right: Vec3
    rear_left: Vec3
    rear_right: Vec3

    def __post_init__(self) -> None:
        positions = (self.front_left, self.front_right, self.rear_left, self.rear_right)
        if any(not isinstance(point, Vec3) for point in positions):
            raise ValueError("wheel contact points must be Vec3")
        if len(set(positions)) != 4:
            raise ValueError("four unique wheel contact points are required")
        if not (self.front_left.z > self.rear_left.z
                and self.front_right.z > self.rear_right.z
                and self.front_left.x < self.front_right.x
                and self.rear_left.x < self.rear_right.x):
            raise ValueError("wheel reference points must have front/rear and left/right ordering")

    @property
    def positions(self) -> tuple[Vec3, ...]:
        return (self.front_left, self.front_right, self.rear_left, self.rear_right)


@dataclass(frozen=True, slots=True)
class PlanarRampVerdict:
    status: str
    checks: int
    reason: str

    def __post_init__(self) -> None:
        if self.status not in ("clear", "outside", "uncertain", "unsupported"):
            raise ValueError("invalid planar-ramp support status")
        if type(self.checks) is not int or self.checks < 0:
            raise ValueError("checks must be a non-negative integer")

    @property
    def valid(self) -> bool:
        return self.status == "clear"


def certify_single_planar_ramp(
    previous: VehicleState,
    candidate: VehicleState,
    body: VehicleBodyGeometry,
    wheels: WheelContactPoints,
    patch: PlanarSurfacePatch,
    *,
    plane_tolerance_m: float,
    max_subdivisions: int = 12,
) -> PlanarRampVerdict:
    """Certify swept body projection + all 4 tire points on one planar patch.

    Every 3D body corner is projected onto the road plane to check road
    limits; each tire-ground reference point additionally must lie within
    'plane_tolerance_m' of the patch plane.

    A status of 'clear' is conservative for the stated rigid interpolation,
    NOT a claim of true suspension forces, physical tire footprints,
    road friction or landing safety.
    """
    if (not isinstance(previous, VehicleState)
        or not isinstance(candidate, VehicleState)
        or not isinstance(body, VehicleBodyGeometry)
        or not isinstance(wheels, WheelContactPoints)
        or not isinstance(patch, PlanarSurfacePatch)):
        raise ValueError("invalid ramp support inputs")
    if (isinstance(plane_tolerance_m, bool)
        or not isinstance(plane_tolerance_m, (int, float))
        or not isfinite(plane_tolerance_m)
        or plane_tolerance_m <= 0.0):
        raise ValueError("plane_tolerance_m must be finite positive")
    if type(max_subdivisions) is not int or not 0 <= max_subdivisions <= 20:
        raise ValueError("max_subdivisions must be integer in [0,20]")
    if patch.normal.y <= 0.0:
        return PlanarRampVerdict("unsupported", 0, "support plane must face upward")
    if not previous.grounded or not candidate.grounded:
        return PlanarRampVerdict("unsupported", 0, "airborne/landing policy is not registered")

    q0, q1 = previous.orientation, candidate.orientation
    dot = abs(q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z)
    theta = 2.0*acos(max(0.0, min(1.0, dot)))
    delta = candidate.position-previous.position
    right, forward, normal = patch.right, patch.forward, patch.normal

    # Each projected scalar coordinate changes at most its translational
    # component plus (angle * maximum local point radius).
    body_radius = body.motion_radius_m
    wheel_radius = max(p.magnitude for p in wheels.positions)
    body_r_bound = abs(delta.dot(right)) + theta*body_radius
    body_f_bound = abs(delta.dot(forward)) + theta*body_radius
    wheel_r_bound = abs(delta.dot(right)) + theta*wheel_radius
    wheel_f_bound = abs(delta.dot(forward)) + theta*wheel_radius
    wheel_n_bound = abs(delta.dot(normal)) + theta*wheel_radius

    checks = 0

    def margins(t: float) -> tuple[tuple[float,float], ...]:
        nonlocal checks
        checks += 1
        position = previous.position + delta*t
        rotation = _slerp(q0, q1, t)
        box = body.box_at(position, rotation)
        b_right = rotation.rotate(Vec3(1.,0.,0.))
        b_up = rotation.rotate(Vec3(0.,1.,0.))
        b_forward = rotation.rotate(Vec3(0.,0.,1.))
        e = box.half_extents
        results: list[tuple[float,float]] = []

        # Eight vertices suffice because the patch is convex; if every
        # projected vertex is inside, the entire projected body is inside.
        for sx in (-1., 1.):
            for sy in (-1., 1.):
                for sz in (-1., 1.):
                    v = box.center + b_right*(sx*e.x) + b_up*(sy*e.y) + b_forward*(sz*e.z)
                    offset = v-patch.origin
                    results.append((
                        patch.width/2. - abs(offset.dot(right)),
                        body_r_bound,
                    ))
                    results.append((
                        patch.length/2. - abs(offset.dot(forward)),
                        body_f_bound,
                    ))

        for local in wheels.positions:
            world = position + rotation.rotate(local)
            offset = world-patch.origin
            results.extend((
                (patch.width/2. - abs(offset.dot(right)), wheel_r_bound),
                (patch.length/2. - abs(offset.dot(forward)), wheel_f_bound),
                (plane_tolerance_m - abs(offset.dot(normal)), wheel_n_bound),
            ))
        return tuple(results)

    def status_at(t: float) -> tuple[str,tuple[tuple[float,float],...]]:
        m = margins(t)
        if any(value < -_EPS for value,_ in m):
            return "outside", m
        return "supported", m

    first,_ = status_at(0.0)
    if first == "outside":
        return PlanarRampVerdict("outside", checks, "start body or tire contact outside ramp")
    last,_ = status_at(1.0)
    if last == "outside":
        return PlanarRampVerdict("outside", checks, "end body or tire contact outside ramp")

    def visit(lo:float, hi:float, depth:int)->str:
        mid=(lo+hi)/2.
        state,m = status_at(mid)
        if state=="outside":
            return "outside"
        if all(value >= bound*(hi-lo)/2. + _EPS for value,bound in m):
            return "clear"
        # A stationary, exactly on-boundary contact can still be certified
        # because there is zero continuous displacement.
        if all(bound == 0.0 and value >= -_EPS for value,bound in m):
            return "clear"
        if depth==0:
            return "uncertain"
        left=visit(lo,mid,depth-1)
        if left=="outside":
            return left
        right_status=visit(mid,hi,depth-1)
        if right_status=="outside":
            return right_status
        return "uncertain" if "uncertain" in (left,right_status) else "clear"

    verdict=visit(0.,1.,max_subdivisions)
    return PlanarRampVerdict(verdict,checks,{
        "clear":"whole-body projected footprint and four tire points supported throughout planar sweep",
        "outside":"body footprint or at least one tire contact outside planar road",
        "uncertain":"continuous planar wheel support not provable within subdivision budget",
    }[verdict])


@dataclass(frozen=True, slots=True)
class PlanarRampPolicy:
    """Explicit research-only single-patch support policy, never auto-selected."""
    wheels: WheelContactPoints
    plane_tolerance_m: float
    max_subdivisions: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.wheels, WheelContactPoints):
            raise ValueError("ramp policy requires WheelContactPoints")
        if (isinstance(self.plane_tolerance_m, bool)
            or not isinstance(self.plane_tolerance_m, (float, int))
            or not isfinite(self.plane_tolerance_m)
            or self.plane_tolerance_m <= 0.0):
            raise ValueError("ramp policy tolerance must be finite positive")
        if (type(self.max_subdivisions) is not int
            or not 0 <= self.max_subdivisions <= 20):
            raise ValueError("ramp policy subdivision budget must be an integer in [0,20]")
