"""Prospective WP7 body-aware 3D collision kernel (not legacy physics).

A vehicle body is an oriented 3D box, including its local centre offset.
The sweep interpolates position linearly and orientation by shortest-arc SLERP.
A 15-axis OBB SAT checks actual box intersection at sampled poses. Between
samples a conservative Hausdorff displacement bound proves separation, or
the interval is subdivided. Unresolved intervals are *blocked*, never clear.

This API is deliberately separate from the legacy sampled-centre simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite, sin

from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState


_AXES = (Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0))
_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class VehicleBodyGeometry:
    """Prospective rigid collision hull. Caller must explicitly choose dimensions.

    +X width, +Y height, +Z length. No dimensions are inferred from wheelbase.
    """
    width_m: float
    height_m: float
    length_m: float
    local_center: Vec3

    def __post_init__(self) -> None:
        for name in ("width_m", "height_m", "length_m"):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite positive")
        if not isinstance(self.local_center, Vec3):
            raise ValueError("local_center must be Vec3")

    @property
    def half_extents(self) -> Vec3:
        return Vec3(self.width_m / 2.0, self.height_m / 2.0, self.length_m / 2.0)

    @property
    def motion_radius_m(self) -> float:
        """Any hull point is within this radius of vehicle's state position."""
        e = self.half_extents
        o = self.local_center
        return Vec3(abs(o.x) + e.x, abs(o.y) + e.y, abs(o.z) + e.z).magnitude

    def box_at(self, position: Vec3, orientation: Quaternion) -> OrientedBoxVolume:
        if not isinstance(position, Vec3) or not isinstance(orientation, Quaternion):
            raise ValueError("body pose requires position Vec3 and orientation Quaternion")
        return OrientedBoxVolume(
            position + orientation.rotate(self.local_center),
            self.half_extents,
            orientation,
        )


@dataclass(frozen=True, slots=True)
class SweptCollisionResult:
    """contact=sampled overlap, uncertain=insufficient resolution (blocked)."""
    contact: bool
    uncertain: bool
    checks: int

    @property
    def blocked(self) -> bool:
        return self.contact or self.uncertain


def _box_axes(box: OrientedBoxVolume) -> tuple[Vec3, Vec3, Vec3]:
    return tuple(box.orientation.rotate(axis) for axis in _AXES)  # type: ignore[return-value]


def obb_separation_lower_bound(first: OrientedBoxVolume, second: OrientedBoxVolume) -> float:
    """Positive => provably separated; nonpositive => SAT overlap/contact.

    The maximum separating projection gap over the 15 OBB axes is a valid
    *lower bound* on Euclidean separation when positive.
    """
    if not isinstance(first, OrientedBoxVolume) or not isinstance(second, OrientedBoxVolume):
        raise ValueError("two OrientedBoxVolume objects required")
    a, b = _box_axes(first), _box_axes(second)
    axes = (*a, *b, *(x.cross(y) for x in a for y in b))
    ah = (first.half_extents.x, first.half_extents.y, first.half_extents.z)
    bh = (second.half_extents.x, second.half_extents.y, second.half_extents.z)
    delta = second.center - first.center
    gap = float("-inf")
    for raw_axis in axes:
        magnitude = raw_axis.magnitude
        if magnitude < 1e-12:
            continue
        axis = raw_axis / magnitude
        a_radius = sum(h * abs(axis.dot(unit)) for h, unit in zip(ah, a, strict=True))
        b_radius = sum(h * abs(axis.dot(unit)) for h, unit in zip(bh, b, strict=True))
        gap = max(gap, abs(delta.dot(axis)) - a_radius - b_radius)
    return gap


def _slerp(first: Quaternion, second: Quaternion, t: float) -> Quaternion:
    # Endpoints are the ACCEPTED source poses, even for tiny rotations.
    # Returning the first pose at t=1 silently invalidates witnessed
    # micro-step road/hard-stop contact brackets.
    if t == 0.0:
        return first
    if t == 1.0:
        return second
    dot = first.w * second.w + first.x * second.x + first.y * second.y + first.z * second.z
    sign = 1.0 if dot >= 0.0 else -1.0
    aligned = (second.w * sign, second.x * sign, second.y * sign, second.z * sign)
    dot = min(1.0, max(-1.0, dot * sign))
    if dot > 1.0 - 1e-12:
        # First-order normalized quaternion blend, not a frozen pose;
        # preserves angular motion throughout arbitrarily short steps.
        return Quaternion(
            first.w*(1.0-t)+aligned[0]*t,
            first.x*(1.0-t)+aligned[1]*t,
            first.y*(1.0-t)+aligned[2]*t,
            first.z*(1.0-t)+aligned[3]*t)
    arc = acos(dot)
    denominator = sin(arc)
    left = sin((1.0 - t) * arc) / denominator
    right = sin(t * arc) / denominator
    return Quaternion(
        first.w * left + aligned[0] * right,
        first.x * left + aligned[1] * right,
        first.y * left + aligned[2] * right,
        first.z * left + aligned[3] * right,
    )


def sweep_body_against_obstacle(
    previous: VehicleState,
    candidate: VehicleState,
    body: VehicleBodyGeometry,
    obstacle: OrientedBoxVolume,
    *,
    max_subdivisions: int = 12,
) -> SweptCollisionResult:
    """No false clearances within stated linear+SLERP motion assumptions.

    If an interval cannot be proven separated by conservative displacement
    bounds, report 'uncertain' and block it rather than permitting tunnelling.
    """
    if (
        not isinstance(previous, VehicleState) or not isinstance(candidate, VehicleState)
        or not isinstance(body, VehicleBodyGeometry)
        or not isinstance(obstacle, OrientedBoxVolume)
    ):
        raise ValueError("invalid body sweep inputs")
    if type(max_subdivisions) is not int or not 0 <= max_subdivisions <= 20:
        raise ValueError("max_subdivisions must be an integer in [0,20]")
    q0, q1 = previous.orientation, candidate.orientation
    dot = abs(q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z)
    theta = 2.0 * acos(max(0.0, min(1.0, dot)))
    translation = (candidate.position - previous.position).magnitude
    max_motion = translation + theta * body.motion_radius_m
    checks = 0

    def gap_at(t: float) -> float:
        nonlocal checks
        checks += 1
        position = previous.position + (candidate.position - previous.position) * t
        rotated = _slerp(q0, q1, t)
        return obb_separation_lower_bound(body.box_at(position, rotated), obstacle)

    # Also catch bodies already overlapping at the beginning of a step.
    if gap_at(0.0) <= _EPS or gap_at(1.0) <= _EPS:
        return SweptCollisionResult(True, False, checks)

    def visit(lo: float, hi: float, remaining: int) -> tuple[bool, bool]:
        middle = (lo + hi) / 2.0
        separation = gap_at(middle)
        if separation <= _EPS:
            return True, False
        # For any t in [lo,hi], every hull point differs from the pose at
        # 'middle' by <= (hi-lo)*(translation+theta*motion_radius)/2.
        # Hence obstacle distance cannot reach zero when gap exceeds bound.
        half_window_motion = (hi - lo) * max_motion / 2.0
        if separation > half_window_motion + _EPS:
            return False, False
        if remaining == 0:
            return False, True
        first_hit, first_unknown = visit(lo, middle, remaining - 1)
        if first_hit:
            return True, False
        last_hit, last_unknown = visit(middle, hi, remaining - 1)
        return last_hit, first_unknown or last_unknown

    contact, uncertain = visit(0.0, 1.0, max_subdivisions)
    # A witnessed overlap is sufficient evidence of contact; unknown
    # clearance elsewhere does not erase the witnessed collision.
    return SweptCollisionResult(contact, uncertain and not contact, checks)


def sweep_ballistic_body_against_obstacle(previous, candidate, body, obstacle, *,
                                           duration_s, gravity_mps2, max_subdivisions=12):
    """Bound a gravity parabola between endpoints with shortest-arc SLERP.

    Expand the obstacle along its local axes by the vertical chord sagitta.
    A clear expanded-obstacle sweep therefore bounds the whole body path.
    """
    if (type(duration_s) not in (int,float) or not isfinite(duration_s) or duration_s<=0.
            or type(gravity_mps2) not in (int,float) or not isfinite(gravity_mps2) or gravity_mps2<=0.
            or not isinstance(obstacle,OrientedBoxVolume)):
        raise ValueError('finite positive ballistic duration/gravity and obstacle required')
    sagitta=gravity_mps2*duration_s*duration_s/8.
    if not isfinite(sagitta):
        raise ValueError('ballistic obstacle envelope exceeds finite domain')
    ax,ay,az=_box_axes(obstacle)
    e=obstacle.half_extents
    expanded=OrientedBoxVolume(obstacle.center,
        Vec3(e.x+sagitta*abs(ax.y),e.y+sagitta*abs(ay.y),e.z+sagitta*abs(az.y)),obstacle.orientation)
    return sweep_body_against_obstacle(previous,candidate,body,expanded,max_subdivisions=max_subdivisions)
