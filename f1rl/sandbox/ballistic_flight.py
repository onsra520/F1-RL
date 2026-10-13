"""Prospective WP7 flight kernel: explicit launch, exact ballistic arc, guarded landing.

Strict v1 domain: fixed airborne orientation, horizontal 2.5D support patches.
No automatic ground-to-air release under partial wheel contact, no slope
landing, no impulse/tire-force claims. Unproven collision => blocked rollout.
Landing consumes only its physical portion of dt; caller MUST integrate or
reject the reported remainder rather than silently changing simulation time.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, hypot, isfinite, sqrt

from .body_collision import VehicleBodyGeometry, sweep_body_against_obstacle
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .multi_surface_support import (
    MultiSurfaceSuspensionPolicy, _atlas, _within_shape, _body_corners,
    _intersection, certify_multisurface_support,
)
from .road_footprint import _hull
from .road_footprint import body_xz_polygon
from .track.patch import PlanarSurfacePatch
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState

_UP = Vec3(0., 1., 0.)
_EPS = 1e-9


class FlightPhysicsError(RuntimeError):
    """Invalid/unsupported prospective flight, never a scored collision event."""


@dataclass(frozen=True, slots=True)
class BallisticFlightPolicy:
    contact: MultiSurfaceSuspensionPolicy
    max_touchdown_vertical_speed_mps: float
    max_touchdown_tilt_rad: float
    collision_segments: int = 16
    touchdown_time_epsilon_s: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.contact, MultiSurfaceSuspensionPolicy):
            raise ValueError("flight requires explicit independent contact contract")
        for name in ("max_touchdown_vertical_speed_mps",
                     "max_touchdown_tilt_rad", "touchdown_time_epsilon_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not isfinite(value) or value <= 0.):
                raise ValueError(f"{name} must be finite positive")
        if self.max_touchdown_tilt_rad >= .75:
            raise ValueError("touchdown tilt must obey small-angle research domain")
        if type(self.collision_segments) is not int or not 2 <= self.collision_segments <= 256:
            raise ValueError("collision_segments must be integer in [2, 256]")


@dataclass(frozen=True, slots=True)
class FlightState:
    """'takeoff' requires explicit authorized release from a supported pose."""
    phase: str = "grounded"

    def __post_init__(self) -> None:
        if self.phase not in ("grounded", "takeoff", "airborne"):
            raise ValueError("flight phase must be grounded, takeoff or airborne")


@dataclass(frozen=True, slots=True)
class FlightTransition:
    kind: str
    time_s: float
    patch_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("takeoff", "landing"):
            raise ValueError("unknown flight transition")
        if not isfinite(self.time_s) or self.time_s < 0.:
            raise ValueError("flight event time must be nonnegative")


@dataclass(frozen=True, slots=True)
class FlightStep:
    vehicle: VehicleState
    flight: FlightState
    poses: tuple[VehicleState, ...]
    transitions: tuple[FlightTransition, ...]
    remaining_dt_s: float
    touchdown_speed_mps: float | None = None

    def __post_init__(self) -> None:
        if not self.poses or self.vehicle != self.poses[-1]:
            raise ValueError("flight poses must include final pose")
        if not isfinite(self.remaining_dt_s) or self.remaining_dt_s < 0.:
            raise ValueError("invalid remaining simulation time")


def authorize_takeoff(
    grounded: VehicleState, policy: BallisticFlightPolicy,
    body: VehicleBodyGeometry, surfaces: tuple[PlanarSurfacePatch, ...],
) -> VehicleState:
    """Explicit upward release only; a wheel dropping off an edge is NOT inferred."""
    if not isinstance(grounded, VehicleState) or not grounded.grounded:
        raise FlightPhysicsError("takeoff requires grounded previous pose")
    if grounded.velocity.y <= 0.:
        raise FlightPhysicsError("takeoff requires positive upward launch velocity")
    if grounded.angular_velocity.magnitude > _EPS:
        raise FlightPhysicsError("rotating airborne launch not yet supported")
    if not isinstance(policy, BallisticFlightPolicy) or not isinstance(body, VehicleBodyGeometry):
        raise ValueError("invalid flight launch configuration")
    if not certify_multisurface_support(grounded, grounded, body, policy.contact, surfaces).valid:
        raise FlightPhysicsError("takeoff has no certified four-wheel/body launch support")
    return VehicleState(
        grounded.position, grounded.velocity, grounded.orientation,
        grounded.angular_velocity, False,
    )


def _flight_pose(start: VehicleState, t: float, gravity: float) -> VehicleState:
    if t < 0. or not isfinite(t):
        raise ValueError("nonnegative finite time required")
    return VehicleState(
        start.position + start.velocity*t + Vec3(0., -.5*gravity*t*t, 0.),
        start.velocity + Vec3(0., -gravity*t, 0.),
        start.orientation, start.angular_velocity, False,
    )


def _descending_root(height: float, vertical_velocity: float, gravity: float) -> float | None:
    if height < -_EPS:
        return None
    disc = vertical_velocity**2 + 2.*gravity*max(0., height)
    return (vertical_velocity + sqrt(disc)) / gravity


def _flat_atlas(
    surfaces: tuple[PlanarSurfacePatch, ...], policy: BallisticFlightPolicy,
):
    atlas = _atlas(surfaces, policy.contact.seam_height_tolerance_m)
    if atlas is None:
        raise FlightPhysicsError("unsupported flight road atlas or stacked decks")
    patches, polygons = atlas
    if any(abs(p.normal.x) > _EPS or abs(p.normal.z) > _EPS
           or abs(p.normal.y - 1.) > _EPS for p in patches):
        raise FlightPhysicsError("sloped touchdown surfaces not certified in v1")
    return patches, polygons


def _landable_events(start, dt, gravity, policy, body, patches, polygons):
    """Exact quadratic down-crossings of four wheel reference points and body bottom."""
    candidates = []
    wheel_positions = tuple(start.orientation.rotate(w) for w in policy.contact.wheels.positions)
    min_body_offset = min(v.y - start.position.y for v in _body_corners(start, body))
    body_poly = body_xz_polygon(start, body)
    for p, poly in zip(patches, polygons, strict=True):
        # Before touchdown, the whole chassis may NOT have struck this patch.
        body_t = _descending_root(start.position.y + min_body_offset - p.origin.y,
                                  start.velocity.y, gravity)
        if body_t is None and _intersection(body_poly, poly):
            raise FlightPhysicsError("body starts penetrating eligible road level")
        if 0. < body_t <= dt + _EPS:
            body_xz = tuple((x + start.velocity.x*body_t,
                             z + start.velocity.z*body_t) for x, z in body_poly)
            if _intersection(body_xz, poly):
                candidates.append((body_t, 0, p.id, "body"))
        for i, local in enumerate(wheel_positions):
            t = _descending_root(start.position.y + local.y - p.origin.y,
                                 start.velocity.y, gravity)
            if t is None and _within_shape(
                    start.position.x + local.x, start.position.z + local.z, poly
            ):
                raise FlightPhysicsError("wheel begins under an eligible road level")
            if 0. < t <= dt + _EPS:
                wx = start.position.x + local.x + start.velocity.x*t
                wz = start.position.z + local.z + start.velocity.z*t
                if _within_shape(wx, wz, poly):
                    candidates.append((t, 1, p.id, str(i)))
    return sorted(candidates)


def _certify_obstacle_arc(
    start: VehicleState,
    end_s: float,
    gravity: float,
    body: VehicleBodyGeometry,
    obstacles: tuple[OrientedBoxVolume, ...],
    segments: int,
) -> tuple[VehicleState, ...]:
    """Conservative parabola envelope using y inflation g*h^2/8.

    Between chord endpoints, vertical sagitta of constant-acceleration
    ballistic motion is at most g*h²/8. Inflating obstacle OBBs vertically
    in WORLD coordinates is valid only for axis-aligned obstacle orientations.
    Non-axis-aligned boxes require a separate continuous solver.
    """
    if any(not isinstance(o, OrientedBoxVolume) for o in obstacles):
        raise ValueError("flight obstacle volumes must be OrientedBoxVolume")
    poses = tuple(_flight_pose(start, end_s*i/segments, gravity)
                  for i in range(segments+1))
    for i, (a, b) in enumerate(zip(poses, poses[1:])):
        h = end_s/segments
        sagitta = gravity*h*h/8.
        for obstacle in obstacles:
            q = obstacle.orientation
            if abs(abs(q.w) - 1.) > 1e-10:
                raise FlightPhysicsError("rotated flight obstacle unsupported by world-vertical envelope")
            inflated = OrientedBoxVolume(
                obstacle.center,
                Vec3(obstacle.half_extents.x,
                     obstacle.half_extents.y+sagitta,
                     obstacle.half_extents.z),
                obstacle.orientation,
            )
            sweep = sweep_body_against_obstacle(a, b, body, inflated)
            if sweep.contact or sweep.uncertain:
                raise FlightPhysicsError(
                    f"ballistic obstacle sweep blocked at segment {i}: "
                    f"{'contact' if sweep.contact else 'uncertain'}"
                )
    return poses


def advance_ballistic_flight(
    airborne: VehicleState, policy: BallisticFlightPolicy,
    body: VehicleBodyGeometry, surfaces: tuple[PlanarSurfacePatch, ...],
    obstacles: tuple[OrientedBoxVolume, ...], *,
    dt: float, gravity_mps2: float,
) -> FlightStep:
    """Advance exact gravity-only path, with certified early contact or rejection.

    LANDING result is an event-time state; caller must integrate remaining_dt_s
    using its registered grounded dynamics before claiming a full world step.
    """
    if (not isinstance(airborne, VehicleState) or airborne.grounded
            or not isinstance(policy, BallisticFlightPolicy)
            or not isinstance(body, VehicleBodyGeometry)):
        raise ValueError("airborne vehicle, body and flight policy required")
    if (isinstance(dt, bool) or not isinstance(dt, (int, float))
            or not isfinite(dt) or dt <= 0.
            or isinstance(gravity_mps2, bool)
            or not isinstance(gravity_mps2, (int, float))
            or not isfinite(gravity_mps2) or gravity_mps2 <= 0.):
        raise ValueError("positive finite dt and gravity required")
    if airborne.angular_velocity.magnitude > _EPS:
        raise FlightPhysicsError("rotating airborne body unsupported until continuous attitude solver")
    patches, polygons = _flat_atlas(surfaces, policy)
    # A zero-time descending wheel contact must never be ignored simply
    # because the later quadratic descending root is t=0.
    initial_footprint = body_xz_polygon(airborne, body)
    initial_bottom_y = min(v.y for v in _body_corners(airborne, body))
    wheel_offsets = tuple(airborne.orientation.rotate(w)
                          for w in policy.contact.wheels.positions)
    for patch, polygon in zip(patches, polygons, strict=True):
        if _intersection(initial_footprint, polygon):
            if initial_bottom_y < patch.origin.y - _EPS:
                raise FlightPhysicsError("airborne chassis begins penetrating road")
            if (initial_bottom_y <= patch.origin.y + _EPS
                    and airborne.velocity.y <= 0.):
                raise FlightPhysicsError("airborne chassis contacts road at start")
        for local in wheel_offsets:
            x, z = airborne.position.x + local.x, airborne.position.z + local.z
            if _within_shape(x, z, polygon):
                clearance = airborne.position.y + local.y - patch.origin.y
                if clearance < -_EPS or (clearance <= _EPS and airborne.velocity.y <= 0.):
                    raise FlightPhysicsError("airborne wheel begins in descending road contact")
    events = _landable_events(airborne, dt, gravity_mps2, policy, body, patches, polygons)
    first = events[0] if events else None
    end_time = min(dt, first[0]) if first else dt
    # Conservative lateral-entry guard. A vehicle that has fallen below the
    # road plane may not enter a road patch sideways without a wheel-first
    # certified touchdown, even when its vertical down-crossing was outside.
    endpoint = _flight_pose(airborne, end_time, gravity_mps2)
    final_poly = body_xz_polygon(endpoint, body)
    path_hull = _hull((*initial_footprint, *final_poly))
    body_floor = min(v.y for v in _body_corners(endpoint, body))
    for patch, polygon in zip(patches, polygons, strict=True):
        if body_floor < patch.origin.y - _EPS and _intersection(path_hull, polygon):
            raise FlightPhysicsError("unmodeled airborne lateral entry below road surface")
    poses = _certify_obstacle_arc(
        airborne, end_time, gravity_mps2, body, obstacles, policy.collision_segments,
    )
    if first is None:
        return FlightStep(poses[-1], FlightState("airborne"), poses, (), 0.)
    t, _, patch_id, collision_kind = first
    if collision_kind == "body":
        raise FlightPhysicsError("chassis contacts road before eligible wheel touchdown")
    impact = _flight_pose(airborne, t, gravity_mps2)
    down_speed = -impact.velocity.y
    if down_speed < 0. or down_speed > policy.max_touchdown_vertical_speed_mps + _EPS:
        raise FlightPhysicsError("unsafe touchdown vertical speed")
    up = impact.orientation.rotate(_UP)
    tilt = acos(max(-1., min(1., up.dot(_UP))))
    if tilt > policy.max_touchdown_tilt_rad + _EPS:
        raise FlightPhysicsError("touchdown tilt exceeds configured acceptance")
    # v1 grounded solver expects a pure-yaw chassis at zero pitch and roll.
    if tilt > 1e-8:
        raise FlightPhysicsError("nonzero landing tilt needs impulse/attitude solver")
    landing = VehicleState(
        impact.position,
        Vec3(impact.velocity.x, 0., impact.velocity.z),
        impact.orientation, Vec3(0., 0., 0.), True,
    )
    support = certify_multisurface_support(landing, landing, body, policy.contact, surfaces)
    if not support.valid:
        raise FlightPhysicsError(f"touchdown road/wheel/body validation {support.status}: {support.reason}")
    # Ensure t wasn't preceded by an unmodeled road/body impact.
    return FlightStep(
        landing, FlightState("grounded"),
        (*poses[:-1], landing), (FlightTransition("landing", t, patch_id),),
        max(0., dt-t), down_speed,
    )
