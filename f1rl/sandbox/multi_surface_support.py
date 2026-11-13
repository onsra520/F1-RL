"""Experimental independent four-wheel contact certification over planar road patches.

Version 1 is a conservative *pose/sweep verifier*, not a suspension force
solver. It does not change legacy tyre forces, integrate spring forces,
or certify jumps/landings. Unknown geometric sweeps are rejected.

The surface set must be a height-consistent 2.5D road atlas: projected
overlaps and touching seams must agree in elevation. Stacked decks and
discontinuous vertical steps are unsupported rather than silently linked.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import acos, hypot, isfinite

from .body_collision import VehicleBodyGeometry, _slerp
from .geometry.vector import Vec3
from .ramp_support import WheelContactPoints
from .road_footprint import _area, _clip, _hull, _inflate, _uncovered_area, body_xz_polygon
from .track.patch import PlanarSurfacePatch
from .vehicle import VehicleState

_EPS = 1e-9
_WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


@dataclass(frozen=True, slots=True)
class IndependentWheelContact:
    wheel: str
    patch_id: str
    ground_height_m: float
    deflection_m: float

    def __post_init__(self) -> None:
        if self.wheel not in _WHEEL_NAMES or not self.patch_id:
            raise ValueError("invalid independent wheel contact")
        if not isfinite(self.ground_height_m) or not isfinite(self.deflection_m):
            raise ValueError("contact geometry must be finite")


@dataclass(frozen=True, slots=True)
class MultiSurfaceSupportVerdict:
    status: str
    checks: int
    reason: str
    contacts: tuple[IndependentWheelContact, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("clear", "outside", "uncertain", "unsupported"):
            raise ValueError("invalid multi-surface verdict")
        if type(self.checks) is not int or self.checks < 0:
            raise ValueError("invalid contact check count")

    @property
    def valid(self) -> bool:
        return self.status == "clear"


@dataclass(frozen=True, slots=True)
class MultiSurfaceSuspensionPolicy:
    wheels: WheelContactPoints
    max_bump_m: float
    max_droop_m: float
    max_subdivisions: int = 12
    seam_height_tolerance_m: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.wheels, WheelContactPoints):
            raise ValueError("explicit four-wheel geometry required")
        for name in ("max_bump_m", "max_droop_m", "seam_height_tolerance_m"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite positive")
        if type(self.max_subdivisions) is not int or not 0 <= self.max_subdivisions <= 20:
            raise ValueError("max_subdivisions must be integer in [0,20]")


def _polygon(p: PlanarSurfacePatch) -> tuple[tuple[float, float], ...]:
    right = p.right
    return _hull(tuple(
        (p.origin.x + sx * right.x * p.width / 2 + sz * p.forward.x * p.length / 2,
         p.origin.z + sx * right.z * p.width / 2 + sz * p.forward.z * p.length / 2)
        for sx in (-1., 1.) for sz in (-1., 1.)
    ))


def _height(p: PlanarSurfacePatch, x: float, z: float) -> float:
    return p.origin.y - (
        p.normal.x * (x - p.origin.x) + p.normal.z * (z - p.origin.z)
    ) / p.normal.y


def _intersection(a, b):
    result = a
    for v0, v1 in zip(b, b[1:] + b[:1], strict=True):
        result = _clip(result, v0, v1, inside=True)
        if not result:
            break
    return result


def minimum_layer_clearance_m(body: VehicleBodyGeometry, margin_m: float = 0.2) -> float:
    if not isinstance(body, VehicleBodyGeometry):
        raise ValueError("registered vehicle body geometry required")
    if type(margin_m) not in (int, float) or not isfinite(margin_m) or margin_m <= 0.:
        raise ValueError("layer clearance margin must be finite and positive")
    half = body.half_extents
    return 2. * hypot(hypot(half.x, half.y), half.z) + margin_m


def has_separated_surface_layers(
    patches: tuple[PlanarSurfacePatch, ...], minimum_clearance_m: float,
    *, continuous_seam_tolerance_m: float = 0.01,
) -> bool:
    for p, q in combinations(patches, 2):
        overlap = _intersection(_polygon(p), _polygon(q))
        if not overlap:
            continue
        differences = tuple(_height(p, x, z) - _height(q, x, z)
                            for x, z in overlap)
        if all(abs(value) <= continuous_seam_tolerance_m for value in differences):
            continue
        if (all(value >= minimum_clearance_m for value in differences)
                or all(value <= -minimum_clearance_m for value in differences)):
            return True
    return False


def _atlas(
    patches: tuple[PlanarSurfacePatch, ...], tol: float,
    *, layered_clearance_m: float | None = None,
    continuous_seam_tolerance_m: float | None = None,
):
    if not isinstance(patches, tuple) or not patches:
        return None
    if (layered_clearance_m is not None and
            (type(layered_clearance_m) not in (int, float)
             or not isfinite(layered_clearance_m) or layered_clearance_m <= 0.)):
        return None
    if (continuous_seam_tolerance_m is not None and
            (type(continuous_seam_tolerance_m) not in (int, float)
             or not isfinite(continuous_seam_tolerance_m)
             or continuous_seam_tolerance_m < tol)):
        return None
    if any(not isinstance(p, PlanarSurfacePatch) or p.normal.y <= 1e-6 for p in patches):
        return None
    if len({p.id for p in patches}) != len(patches):
        return None
    ordered = tuple(sorted(patches, key=lambda p: p.id))
    shapes = tuple(_polygon(p) for p in ordered)
    if any(_area(poly) <= _EPS for poly in shapes):
        return None
    for i, p in enumerate(ordered):
        for q, other in zip(ordered[i + 1:], shapes[i + 1:], strict=True):
            overlap = _intersection(shapes[i], other)
            if not overlap:
                continue
            differences = tuple(_height(p, x, z) - _height(q, x, z)
                                for x, z in overlap)
            if all(abs(value) <= tol for value in differences):
                continue
            first_sequence = p.id.removeprefix("surface-")
            second_sequence = q.id.removeprefix("surface-")
            if (continuous_seam_tolerance_m is not None
                    and first_sequence.isdigit() and second_sequence.isdigit()
                    and abs(int(first_sequence) - int(second_sequence)) == 1
                    and all(abs(value) <= continuous_seam_tolerance_m for value in differences)):
                continue
            if layered_clearance_m is None:
                return None
            if not (all(value >= layered_clearance_m - tol for value in differences)
                    or all(value <= -layered_clearance_m + tol for value in differences)):
                return None
    return ordered, shapes


def _within_shape(x: float, z: float, shape) -> bool:
    # Convex CCW inclusive polygon membership.
    return all(
        (b[0] - a[0]) * (z - a[1]) - (b[1] - a[1]) * (x - a[0]) >= -_EPS
        for a, b in zip(shape, shape[1:] + shape[:1], strict=True)
    )


def _contacts(state, policy, atlas, *, free_extension=False):
    patches, polygons = atlas
    found = []
    for name, local in zip(_WHEEL_NAMES, policy.wheels.positions, strict=True):
        point = state.position + state.orientation.rotate(local)
        candidates = []
        for p, poly in zip(patches, polygons, strict=True):
            if not _within_shape(point.x, point.z, poly):
                continue
            h = _height(p, point.x, point.z)
            d = h - point.y
            if (free_extension or d >= -policy.max_droop_m - _EPS) and d <= policy.max_bump_m + _EPS:
                candidates.append((abs(d), p.id, h, d))
        if not candidates:
            return None
        _, chosen, height, deflection = min(candidates)
        found.append(IndependentWheelContact(name, chosen, height, deflection))
    return tuple(found)


def _body_corners(state, body):
    box = body.box_at(state.position, state.orientation)
    x = box.orientation.rotate(Vec3(1., 0., 0.))
    y = box.orientation.rotate(Vec3(0., 1., 0.))
    z = box.orientation.rotate(Vec3(0., 0., 1.))
    e = box.half_extents
    return tuple(box.center + x * (sx * e.x) + y * (sy * e.y) + z * (sz * e.z)
                 for sx in (-1., 1.) for sy in (-1., 1.) for sz in (-1., 1.))


def _bounded_patch_body_clearance(state, body, patch, polygon, inflation=0.):
    """Lower clearance of an inflated OBB clipped to the patch's vertical prism.

    Clip all six original faces. Every vertex of the bounded intersection lies
    on one of them: vertical prism planes alone have rank at most two. A linear
    signed-plane clearance therefore reaches its minimum in this vertex set.
    """
    box = body.box_at(state.position, state.orientation)
    scale = 1. + max(inflation, *(abs(v) for p in (box.center, box.half_extents, patch.origin)
                                for v in (p.x, p.y, p.z)),
                     *(abs(v) for p in polygon for v in p))
    if not isfinite(scale) or scale > 1e6 or not isfinite(inflation) or inflation < 0.:
        return None
    margin = 1e-9 * scale
    axes = tuple(box.orientation.rotate(v) for v in (Vec3(1.,0.,0.), Vec3(0.,1.,0.), Vec3(0.,0.,1.)))
    e = box.half_extents + Vec3(inflation+margin, inflation+margin, inflation+margin)
    corners = tuple(box.center + axes[0]*(sx*e.x) + axes[1]*(sy*e.y) + axes[2]*(sz*e.z)
                    for sx in (-1.,1.) for sy in (-1.,1.) for sz in (-1.,1.))
    vertices = []
    for indices in ((0,1,3,2), (4,6,7,5), (0,4,5,1), (2,3,7,6), (0,2,6,4), (1,5,7,3)):
        face = tuple(corners[i] for i in indices)
        for a,b in zip(polygon, polygon[1:]+polygon[:1], strict=True):
            if not face:
                break
            dx,dz = b[0]-a[0],b[1]-a[1]
            length = hypot(dx,dz)
            if length <= 1e-4 * scale:
                return None
            def signed(p):
                return (dx*(p.z-a[1])-dz*(p.x-a[0]))/length + margin
            clipped = []
            for first,last in zip(face, face[1:]+face[:1], strict=True):
                d0,d1 = signed(first),signed(last)
                if d0 >= 0.:
                    clipped.append(first)
                if (d0 >= 0.) != (d1 >= 0.):
                    clipped.append(first + (last-first)*(d0/(d0-d1)))
            face = tuple(clipped)
        vertices.extend(face)
    if not vertices:
        return None  # No numerical witness of a bounded intersection.
    clearance = min((p-patch.origin).dot(patch.normal) for p in vertices) - margin
    return clearance if isfinite(clearance) else None


def _square(x, z, r):
    return ((x-r, z-r), (x+r, z-r), (x+r, z+r), (x-r, z+r))


def certify_multisurface_support(
    previous: VehicleState,
    candidate: VehicleState,
    body: VehicleBodyGeometry,
    policy: MultiSurfaceSuspensionPolicy,
    surfaces: tuple[PlanarSurfacePatch, ...],
    *, body_clearance_version: str = 'infinite_patch_planes_v1',
    contact_domain: str = 'four_supported_v1', ballistic_sag_m: float = 0.,
    maximum_deflection_m: float | None = None, minimum_support_deflection_m: float | None = None,
    body_coverage_domain: str = 'full_footprint_v1',
) -> MultiSurfaceSupportVerdict:
    """Conservative continuous four-wheel / body containment for linear+SLERP sweep.

    CLEAR guarantees contact travel and projected whole-body road coverage
    throughout this *prescribed* motion, subject to 2.5D atlas assumptions.
    The explicit unilateral domain instead allows unloaded free extension;
    optional deflection bounds certify phase persistence, not spring forces.
    It does not calculate a physically correct chassis pose or spring forces.
    """
    if contact_domain not in ('four_supported_v1','unilateral_free_extension_v1'):
        raise ValueError('unregistered geometric contact domain')
    free_extension = contact_domain == 'unilateral_free_extension_v1'
    if body_coverage_domain not in ('full_footprint_v1', 'wheel_supported_overhang_v1'):
        raise ValueError('unregistered body coverage domain')
    if body_coverage_domain == 'wheel_supported_overhang_v1' and (
        not free_extension or body_clearance_version != 'bounded_patch_planes_v2'
    ):
        raise ValueError('wheel-supported overhang requires unilateral contact and bounded clearance')
    if type(ballistic_sag_m) not in (int,float) or not isfinite(ballistic_sag_m) or ballistic_sag_m < 0.:
        raise ValueError('finite nonnegative ballistic sag required')
    if ballistic_sag_m and not free_extension:
        raise ValueError('ballistic sag requires unilateral free-extension domain')
    if maximum_deflection_m is not None and (not free_extension
            or type(maximum_deflection_m) not in (int,float) or not isfinite(maximum_deflection_m)
            or maximum_deflection_m>policy.max_bump_m):
        raise ValueError('finite unilateral deflection enclosure required')
    if minimum_support_deflection_m is not None and (not free_extension
            or type(minimum_support_deflection_m) not in (int,float) or not isfinite(minimum_support_deflection_m)
            or maximum_deflection_m is not None):
        raise ValueError('finite exclusive supported-phase enclosure required')
    upper_deflection=policy.max_bump_m if maximum_deflection_m is None else maximum_deflection_m
    if body_clearance_version not in ('infinite_patch_planes_v1', 'bounded_patch_planes_v2'):
        raise ValueError('unregistered body/road clearance version')
    if not isinstance(previous, VehicleState) or not isinstance(candidate, VehicleState):
        raise ValueError("previous/candidate must be VehicleState")
    if not isinstance(body, VehicleBodyGeometry) or not isinstance(policy, MultiSurfaceSuspensionPolicy):
        raise ValueError("explicit body and suspension policy required")
    if not free_extension and (not previous.grounded or not candidate.grounded):
        return MultiSurfaceSupportVerdict("unsupported", 0, "takeoff/landing not registered")
    atlas = _atlas(surfaces, policy.seam_height_tolerance_m)
    if atlas is None:
        return MultiSurfaceSupportVerdict("unsupported", 0, "inconsistent or non-2.5D road atlas")
    patches, polygons = atlas
    q0, q1 = previous.orientation, candidate.orientation
    dot = abs(q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z)
    theta = 2 * acos(min(1., max(0., dot)))
    delta = candidate.position - previous.position
    wheel_bound = delta.magnitude + theta * max(p.magnitude for p in policy.wheels.positions)
    body_bound = delta.magnitude + theta * body.motion_radius_m
    checks = 0
    numerical_unresolved = False

    def state_at(t):
        return VehicleState(previous.position + delta*t, previous.velocity,
                            _slerp(q0, q1, t), previous.angular_velocity, True)

    def sample(t, r_w, r_b):
        nonlocal checks, numerical_unresolved
        checks += 1
        state = state_at(t)
        contacts = _contacts(state, policy, atlas, free_extension=free_extension)
        r_b += ballistic_sag_m
        if contacts is None:
            return False, False, ()
        if maximum_deflection_m is not None and any(c.deflection_m>maximum_deflection_m for c in contacts):
            return False, False, contacts
        if minimum_support_deflection_m is not None and not any(c.deflection_m>minimum_support_deflection_m for c in contacts):
            return False, False, contacts
        hull = body_xz_polygon(state, body)
        envelope = _inflate(hull, r_b)
        if (body_coverage_domain == 'full_footprint_v1'
                and _uncovered_area(envelope, polygons) > _EPS):
            return False, False, contacts

        corners = _body_corners(state, body)
        for patch, poly in zip(patches, polygons, strict=True):
            if not _intersection(envelope, poly):
                continue
            clearance = min((v-patch.origin).dot(patch.normal) for v in corners) - r_b
            if body_clearance_version == 'bounded_patch_planes_v2':
                clearance = _bounded_patch_body_clearance(state, body, patch, poly, r_b)
                if clearance is None:
                    numerical_unresolved = True
                    return True, False, contacts
            if clearance < -_EPS:
                return False, False, contacts

        compressed_witness=False
        for local in policy.wheels.positions:
            wheel_compressed=True
            w = state.position + state.orientation.rotate(local)
            swept = _square(w.x, w.z, r_w)
            if _uncovered_area(swept, polygons) > _EPS:
                return False, False, contacts
            for patch, poly in zip(patches, polygons, strict=True):
                if not _intersection(swept, poly):
                    continue
                d = _height(patch, w.x, w.z) - w.y
                slope = hypot(patch.normal.x, patch.normal.z) / patch.normal.y
                allowance = (1. + slope) * r_w
                if minimum_support_deflection_m is not None and d-allowance<=minimum_support_deflection_m:
                    wheel_compressed=False
                if (not free_extension and d < -policy.max_droop_m + allowance - _EPS
                        or d > upper_deflection - allowance + (0. if maximum_deflection_m is not None else _EPS)):
                    return True, False, contacts
            compressed_witness |= wheel_compressed
        if minimum_support_deflection_m is not None and not compressed_witness:
            return True, False, contacts
        return True, True, contacts

    # Endpoint failures are witnessed invalid poses. Midpoint uncertainty
    # never manufactures an OUTSIDE report or a successful rollout.
    for t in (0., 1.):
        present, _, contacts = sample(t, 0., 0.)
        if numerical_unresolved:
            return MultiSurfaceSupportVerdict('unsupported', checks, 'bounded patch clipping outside numerical domain', contacts)
        if not present:
            return MultiSurfaceSupportVerdict("outside", checks, "invalid endpoint body or wheel support", contacts)

    def visit(lo, hi, depth):
        mid = (lo + hi) / 2.
        radius = (hi - lo) / 2.
        present, proven, contacts = sample(mid, wheel_bound * radius, body_bound * radius)
        if numerical_unresolved:
            return 'uncertain', contacts
        if proven:
            return "clear", contacts
        # Need an actual witnessed invalid midpoint before declaring OUTSIDE.
        if not present:
            actual, _, _ = sample(mid, 0., 0.)
            if not actual:
                return "outside", contacts
        if depth == 0:
            return "uncertain", contacts
        left, left_contacts = visit(lo, mid, depth - 1)
        if left == "outside":
            return left, left_contacts
        right, right_contacts = visit(mid, hi, depth - 1)
        if right == "outside":
            return right, right_contacts
        if "uncertain" in (left, right):
            return "uncertain", contacts
        return "clear", contacts

    verdict, contacts = visit(0., 1., policy.max_subdivisions)
    reasons = {
        "clear": "continuous road union, independent wheel travel and body clearance certified",
        "outside": "witnessed unsupported body or wheel pose within motion",
        "uncertain": "continuous contact not certified within subdivision budget",
    }
    return MultiSurfaceSupportVerdict(verdict, checks, reasons[verdict], contacts)
