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
from decimal import Decimal, localcontext
from math import acos, isfinite, sin, ulp

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


def _same_rotation(first: Quaternion, second: Quaternion) -> bool:
    return (
        (first.w == second.w and first.x == second.x
         and first.y == second.y and first.z == second.z)
        or (first.w == -second.w and first.x == -second.x
            and first.y == -second.y and first.z == -second.z)
    )


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
    same_rotation = _same_rotation(first.orientation, second.orientation)
    for index, raw_axis in enumerate(axes):
        magnitude = raw_axis.magnitude
        if magnitude < 1e-12:
            if index >= 6 and (magnitude != 0. or not same_rotation):
                return float(_precise_obb_separation(first, second))
            continue
        axis = raw_axis / magnitude
        a_radius = sum(h * abs(axis.dot(unit)) for h, unit in zip(ah, a, strict=True))
        b_radius = sum(h * abs(axis.dot(unit)) for h, unit in zip(bh, b, strict=True))
        gap = max(gap, abs(delta.dot(axis)) - a_radius - b_radius)
    return gap


def _precise_obb_separation(first: OrientedBoxVolume, second: OrientedBoxVolume) -> Decimal:
    with localcontext() as context:
        context.prec = 100
        decimal = Decimal.from_float

        def dot(left, right):
            return sum((a * b for a, b in zip(left, right)), Decimal(0))

        def cross(left, right):
            return (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )

        def axes(box):
            quaternion = tuple(decimal(value) for value in (
                box.orientation.w, box.orientation.x,
                box.orientation.y, box.orientation.z,
            ))
            norm = sum((value * value for value in quaternion), Decimal(0)).sqrt()
            w, x, y, z = (value / norm for value in quaternion)
            vector = (x, y, z)
            norm_squared = dot(vector, vector)
            result = []
            for basis in (
                (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
            ):
                value = tuple(decimal(component) for component in basis)
                projection = dot(vector, value)
                perpendicular = cross(vector, value)
                result.append(tuple(
                    2 * vector[i] * projection
                    + value[i] * (w * w - norm_squared)
                    + 2 * w * perpendicular[i]
                    for i in range(3)
                ))
            return tuple(result)

        first_axes, second_axes = axes(first), axes(second)
        first_half = tuple(decimal(value) for value in (
            first.half_extents.x, first.half_extents.y, first.half_extents.z,
        ))
        second_half = tuple(decimal(value) for value in (
            second.half_extents.x, second.half_extents.y, second.half_extents.z,
        ))
        offset = tuple(
            decimal(b) - decimal(a)
            for a, b in zip(
                (first.center.x, first.center.y, first.center.z),
                (second.center.x, second.center.y, second.center.z),
            )
        )
        gaps = []
        for raw_axis in (
            *first_axes,
            *second_axes,
            *(cross(a, b) for a in first_axes for b in second_axes),
        ):
            magnitude_squared = dot(raw_axis, raw_axis)
            if magnitude_squared == 0:
                continue
            magnitude = magnitude_squared.sqrt()
            axis = tuple(value / magnitude for value in raw_axis)
            first_radius = sum((
                extent * abs(dot(axis, unit))
                for extent, unit in zip(first_half, first_axes)
            ), Decimal(0))
            second_radius = sum((
                extent * abs(dot(axis, unit))
                for extent, unit in zip(second_half, second_axes)
            ), Decimal(0))
            gaps.append(abs(dot(offset, axis)) - first_radius - second_radius)
        return max(gaps)


def _obb_geometry_scale(first: OrientedBoxVolume, second: OrientedBoxVolume) -> float:
    return max(
        1., abs(first.center.x), abs(first.center.y), abs(first.center.z),
        abs(second.center.x), abs(second.center.y), abs(second.center.z),
        abs(first.half_extents.x), abs(first.half_extents.y), abs(first.half_extents.z),
        abs(second.half_extents.x), abs(second.half_extents.y), abs(second.half_extents.z),
    )


def _obb_roundoff_margin(
    first: OrientedBoxVolume, second: OrientedBoxVolume, threshold_m: float,
) -> float:
    return 256. * ulp(max(_obb_geometry_scale(first, second), abs(threshold_m)))


def _obb_separation_comparisons_from_gap(
    first: OrientedBoxVolume,
    second: OrientedBoxVolume,
    thresholds_m: tuple[float, ...],
    gap: float,
) -> tuple[bool, ...]:
    if any(type(threshold) not in (int, float)
           or not isfinite(threshold) or threshold < 0.
           for threshold in thresholds_m):
        raise ValueError("finite non-negative SAT threshold required")
    geometry_scale = _obb_geometry_scale(first, second)
    margins = tuple(256. * ulp(max(geometry_scale, abs(threshold)))
                    for threshold in thresholds_m)
    if not isfinite(gap) or any(
        abs(gap - threshold) <= margin
        for threshold, margin in zip(thresholds_m, margins, strict=True)
    ):
        precise_gap = _precise_obb_separation(first, second)
        return tuple(
            precise_gap <= Decimal.from_float(float(threshold))
            for threshold in thresholds_m
        )
    return tuple(gap <= threshold for threshold in thresholds_m)


def _has_finite_obb_geometry(box: object) -> bool:
    if (not isinstance(box, OrientedBoxVolume)
            or not isinstance(box.center, Vec3)
            or not isinstance(box.half_extents, Vec3)
            or not isinstance(box.orientation, Quaternion)):
        return False
    return all(
        value > 0. for value in (
            box.half_extents.x, box.half_extents.y, box.half_extents.z,
        )
    )


def obb_separation_at_most(
    first: OrientedBoxVolume, second: OrientedBoxVolume, threshold_m: float,
) -> bool:
    """Conservatively compare the OBB SAT gap to a physical distance threshold.

    The fast float kernel decides queries outside a 256-ULP uncertainty band.
    Decimal SAT recomputes only queries inside that band. Invalid geometry or
    threshold input raises so callers cannot score uncertainty as a collision.
    """
    if not _has_finite_obb_geometry(first) or not _has_finite_obb_geometry(second):
        raise ValueError("finite OBB geometry required for SAT comparison")
    if type(threshold_m) not in (int, float):
        raise ValueError("finite non-negative SAT threshold required")
    try:
        threshold = float(threshold_m)
    except (OverflowError, ValueError):
        raise ValueError("finite non-negative SAT threshold required") from None
    if not isfinite(threshold) or threshold < 0.:
        raise ValueError("finite non-negative SAT threshold required")

    gap = obb_separation_lower_bound(first, second)
    return _obb_separation_at_most_from_gap(first, second, threshold, gap)


def _obb_separation_at_most_from_gap(
    first: OrientedBoxVolume,
    second: OrientedBoxVolume,
    threshold_m: float,
    gap: float,
) -> bool:
    if (type(threshold_m) not in (int, float)
            or not isfinite(threshold_m) or threshold_m < 0.):
        raise ValueError("finite non-negative SAT threshold required")
    margin = 256. * ulp(max(_obb_geometry_scale(first, second), abs(threshold_m)))
    if not isfinite(gap) or abs(gap - threshold_m) <= margin:
        return _precise_obb_separation(first, second) <= Decimal.from_float(
            float(threshold_m),
        )
    return gap <= threshold_m


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

    def gap_at(t: float) -> tuple[OrientedBoxVolume, float]:
        nonlocal checks
        checks += 1
        position = previous.position + (candidate.position - previous.position) * t
        rotated = _slerp(q0, q1, t)
        box = body.box_at(position, rotated)
        return box, obb_separation_lower_bound(box, obstacle)

    # Also catch bodies already overlapping at the beginning of a step.
    first_box, first_gap = gap_at(0.0)
    if _obb_separation_at_most_from_gap(first_box, obstacle, _EPS, first_gap):
        return SweptCollisionResult(True, False, checks)
    last_box, last_gap = gap_at(1.0)
    if _obb_separation_at_most_from_gap(last_box, obstacle, _EPS, last_gap):
        return SweptCollisionResult(True, False, checks)

    def visit(lo: float, hi: float, remaining: int) -> tuple[bool, bool]:
        middle = (lo + hi) / 2.0
        middle_box, separation = gap_at(middle)
        # For any t in [lo,hi], every hull point differs from the pose at
        # 'middle' by <= (hi-lo)*(translation+theta*motion_radius)/2.
        # Hence obstacle distance cannot reach zero when gap exceeds bound.
        half_window_motion = (hi - lo) * max_motion / 2.0
        contact, may_contact = _obb_separation_comparisons_from_gap(
            middle_box, obstacle, (_EPS, half_window_motion + _EPS), separation,
        )
        if contact:
            return True, False
        if not may_contact:
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
