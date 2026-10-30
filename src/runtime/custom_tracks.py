from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from math import atan2, hypot, isfinite, sqrt

import numpy as np

from src.runtime.catalog import RuntimeCatalog
from src.simulation.ballistic_flight import FlightState
from src.simulation.entities.boost_pad import BoostPad
from src.simulation.entities.checkpoint import Checkpoint
from src.simulation.entities.finish_line import FinishLine
from src.simulation.entities.obstacle import Obstacle
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.independent_suspension_dynamics import SuspensionDynamicsState
from src.simulation.interactions.state import InteractionState
from src.simulation.multi_surface_support import _atlas
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.model import TrackWorld
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.track.state import TrackState
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.vehicle import VehicleState
from src.simulation.world import WorldState
from src.tracks.model import AuthoredNavigation

MAX_TRACK_BYTES = 1_000_000
MAX_CONTROL_POINTS = 512
MAX_PROPS = 200
MAX_ROUTE_LENGTH_M = 1_000.0
MAX_SURFACES = 512
SURFACE_SPACING_M = 4.0
ROAD_PREVIEW_SPACING_M = 1.0
MIN_CHECKPOINT_GAP_M = 6.0
SPAWN_DISTANCE_M = 2.6
CONVERSION_SCHEMA = "f1rl_custom_track_conversion_v1"


class TrackValidationError(ValueError):
    def __init__(self, code: str, message: str, *, details: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True, slots=True)
class ConvertedTrack:
    schema_version: str
    development_only: bool
    track_id: str
    track_sha256: str
    source_document: dict[str, object]
    factory: object
    road_samples: tuple[dict[str, float], ...]
    auto_bridge_count: int
    ppo_compatible: bool
    warnings: tuple[str, ...]


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise TrackValidationError("invalid_geometry", f"{name} must be a finite number")
    number = float(value)
    if not low <= number <= high:
        raise TrackValidationError("invalid_geometry", f"{name} must be within [{low}, {high}]")
    return number


def _smooth_step(a: float, b: float, value: float) -> float:
    t = max(0.0, min(1.0, (value - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def _bump(distance: float, center: float, plateau: float, outer: float) -> float:
    delta = abs(distance - center)
    if delta <= plateau:
        return 1.0
    if delta >= outer:
        return 0.0
    t = (delta - plateau) / (outer - plateau)
    return 1.0 - t * t * (3.0 - 2.0 * t)


class _CentripetalSpline:
    """Three.js CatmullRomCurve3 centripetal sampling for the editor's open route."""

    def __init__(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self._arc_lengths = [0.0]
        previous = self.point(0.0)
        for index in range(1, 201):
            current = self.point(index / 200.0)
            self._arc_lengths.append(self._arc_lengths[-1] + hypot(current[0] - previous[0], current[1] - previous[1]))
            previous = current
        self.length = self._arc_lengths[-1]

    def point(self, value: float) -> tuple[float, float]:
        t = max(0.0, min(1.0, value)) * (len(self.points) - 1)
        index = min(len(self.points) - 2, int(t))
        weight = t - index
        p1 = self.points[index]
        p2 = self.points[index + 1]
        p0 = self.points[index - 1] if index > 0 else (2.0 * p1[0] - p2[0], 2.0 * p1[1] - p2[1])
        p3 = self.points[index + 2] if index + 2 < len(self.points) else (2.0 * p2[0] - p1[0], 2.0 * p2[1] - p1[1])

        def distance_power(a: tuple[float, float], b: tuple[float, float]) -> float:
            squared = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            return max(sqrt(sqrt(squared)), 1e-4)

        t0 = 0.0
        t1 = t0 + distance_power(p0, p1)
        t2 = t1 + distance_power(p1, p2)
        t3 = t2 + distance_power(p2, p3)
        current = t1 + weight * (t2 - t1)

        def mix(a, b, wa, wb):
            return a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb

        a1 = mix(p0, p1, (t1 - current) / (t1 - t0), (current - t0) / (t1 - t0))
        a2 = mix(p1, p2, (t2 - current) / (t2 - t1), (current - t1) / (t2 - t1))
        a3 = mix(p2, p3, (t3 - current) / (t3 - t2), (current - t2) / (t3 - t2))
        b1 = mix(a1, a2, (t2 - current) / (t2 - t0), (current - t0) / (t2 - t0))
        b2 = mix(a2, a3, (t3 - current) / (t3 - t1), (current - t1) / (t3 - t1))
        return mix(b1, b2, (t2 - current) / (t2 - t1), (current - t1) / (t2 - t1))

    def point_at(self, fraction: float) -> tuple[float, float]:
        target = max(0.0, min(1.0, fraction)) * self.length
        index = min(200, bisect_left(self._arc_lengths, target))
        if index == 0:
            t = 0.0
        else:
            before = self._arc_lengths[index - 1]
            span = self._arc_lengths[index] - before
            part = 0.0 if span <= 1e-12 else (target - before) / span
            t = (index - 1 + part) / 200.0
        return self.point(t)


@dataclass(frozen=True, slots=True)
class _AutoBridge:
    under_m: float
    over_m: float
    rise: float
    plateau_m: float
    outer_m: float


def _segment_intersection(a, b, c, d):
    rx, rz = b[0] - a[0], b[1] - a[1]
    sx, sz = d[0] - c[0], d[1] - c[1]
    determinant = rx * sz - rz * sx
    if abs(determinant) < 1e-8:
        return None
    qx, qz = c[0] - a[0], c[1] - a[1]
    u = (qx * sz - qz * sx) / determinant
    v = (qx * rz - qz * rx) / determinant
    if u <= 0.0001 or u >= 0.9999 or v <= 0.0001 or v >= 0.9999:
        return None
    return u, v, (a[0] + u * rx, a[1] + u * rz)


def _auto_bridges(points: list[tuple[float, float]], distances: list[float], total: float) -> tuple[_AutoBridge, ...]:
    intersections = []
    for first in range(len(points) - 1):
        for second in range(first + 3, len(points) - 1):
            if distances[second] - distances[first + 1] < 16.0:
                continue
            hit = _segment_intersection(points[first], points[first + 1], points[second], points[second + 1])
            if hit is None:
                continue
            u, v, point = hit
            under = distances[first] + u * (distances[first + 1] - distances[first])
            over = distances[second] + v * (distances[second + 1] - distances[second])
            if any(hypot(existing[0] - point[0], existing[1] - point[1]) < 1.5 for existing in intersections):
                continue
            intersections.append((point[0], point[1], under, over))
    intersections.sort(key=lambda value: value[3])
    result: list[_AutoBridge] = []
    for _, _, under, over in intersections:
        remaining = min(over, total - over)
        if remaining < 10.0:
            raise TrackValidationError("unsupported_intersection", "route crossing is too close to the start or finish for a safe overpass")
        plateau = 6.0
        outer = min(42.0, remaining - 0.1)
        if outer <= plateau + 1.0:
            raise TrackValidationError("unsupported_intersection", "route crossing has insufficient room for a supported overpass")
        existing_over = sum(item.rise * _bump(over, item.over_m, item.plateau_m, item.outer_m) for item in result)
        existing_under = sum(item.rise * _bump(under, item.over_m, item.plateau_m, item.outer_m) for item in result)
        rise = max(0.0, 6.0 - (existing_over - existing_under))
        if rise >= 0.01:
            result.append(_AutoBridge(under, over, rise, plateau, outer))
    return tuple(result)


def _parse_document(raw: object) -> tuple[dict[str, object], list[tuple[float, float]], list[dict[str, object]]]:
    if not isinstance(raw, dict) or set(raw) != {"schema", "id", "name", "width", "points", "props"}:
        raise TrackValidationError("invalid_schema", "track must use the exact f1rl_racing_sandbox_draft_v1 fields")
    if raw.get("schema") != "f1rl_racing_sandbox_draft_v1":
        raise TrackValidationError("invalid_schema", "unsupported track schema")
    identity, name = raw.get("id"), raw.get("name")
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 100 or not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise TrackValidationError("invalid_schema", "track id and name must be non-empty strings of at most 100 characters")
    width = _number(raw.get("width"), "width", 4.0, 30.0)
    raw_points, raw_props = raw.get("points"), raw.get("props")
    if not isinstance(raw_points, list) or not 2 <= len(raw_points) <= MAX_CONTROL_POINTS or not isinstance(raw_props, list) or len(raw_props) > MAX_PROPS:
        raise TrackValidationError("resource_limit", "track must contain 2–512 points and at most 200 props")
    points = []
    for index, point in enumerate(raw_points):
        if not isinstance(point, dict) or set(point) != {"x", "z"}:
            raise TrackValidationError("invalid_geometry", f"point {index} must contain only x and z")
        x = _number(point["x"], f"points[{index}].x", -10_000.0, 10_000.0)
        z = _number(point["z"], f"points[{index}].z", -10_000.0, 10_000.0)
        if points and hypot(x - points[-1][0], z - points[-1][1]) < 0.1:
            raise TrackValidationError("invalid_geometry", f"adjacent points {index - 1} and {index} are too close")
        points.append((x, z))
    if len({(round(x, 3), round(z, 3)) for x, z in points}) < 2:
        raise TrackValidationError("invalid_geometry", "track route must have nonzero length")
    props = []
    identities = set()
    kinds = {"bridge", "ramp", "barrier", "cone", "booster", "checkpoint"}
    for index, item in enumerate(raw_props):
        if not isinstance(item, dict) or set(item) != {"id", "kind", "t", "offset", "rotation"}:
            raise TrackValidationError("invalid_props", f"prop {index} must contain exactly id, kind, t, offset, rotation")
        prop_id, kind = item["id"], item["kind"]
        if (not isinstance(prop_id, str) or not prop_id.strip() or len(prop_id) > 100
                or prop_id in identities or not isinstance(kind, str) or kind not in kinds):
            raise TrackValidationError("invalid_props", f"prop {index} has a duplicate id or unsupported kind")
        identities.add(prop_id)
        props.append({
            "id": prop_id,
            "kind": kind,
            "t": _number(item["t"], f"props[{index}].t", 0.0, 1.0),
            "offset": _number(item["offset"], f"props[{index}].offset", -30.0, 30.0),
            "rotation": _number(item["rotation"], f"props[{index}].rotation", -1000.0, 1000.0),
        })
    clean = {"schema": raw["schema"], "id": identity, "name": name, "width": width,
             "points": [{"x": x, "z": z} for x, z in points], "props": props}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_TRACK_BYTES:
        raise TrackValidationError("resource_limit", "track document exceeds the 1 MB limit")
    return clean, points, props


def _track_elevation(t: float, props: list[dict[str, object]], bridges: tuple[_AutoBridge, ...], total: float) -> float:
    height = 0.0
    for prop in props:
        delta = abs(t - float(prop["t"]))
        if prop["kind"] == "bridge":
            height = max(height, 5.0 * (1.0 - _smooth_step(0.055, 0.125, delta)))
        elif prop["kind"] == "ramp":
            height = max(height, 2.8 * (1.0 - _smooth_step(0.0, 0.075, delta)))
    distance = t * total
    return height + sum(item.rise * _bump(distance, item.over_m, item.plateau_m, item.outer_m) for item in bridges)


def _route_point(points: list[tuple[float, float]], distances: list[float], distance: float) -> tuple[float, float, float]:
    index = min(len(points) - 2, max(0, bisect_left(distances, distance) - 1))
    span = distances[index + 1] - distances[index]
    fraction = 0.0 if span <= 1e-12 else (distance - distances[index]) / span
    first, second = points[index], points[index + 1]
    return first[0] + (second[0] - first[0]) * fraction, first[1] + (second[1] - first[1]) * fraction, index + fraction


def _pose_at(points, distances, distance, props, bridges, total):
    x, z, segment = _route_point(points, distances, distance)
    index = min(len(points) - 2, int(segment))
    first, second = points[index], points[index + 1]
    dx, dz = second[0] - first[0], second[1] - first[1]
    horizontal = hypot(dx, dz)
    if horizontal <= 1e-9:
        raise TrackValidationError("invalid_geometry", "vertical centerline segments are not supported")
    t = distance / total
    y = _track_elevation(t, props, bridges, total)
    heading = atan2(dx, dz)
    return x, y, z, heading, -dz / horizontal, dx / horizontal


def _volume(center: Vec3, half: Vec3, heading: float) -> OrientedBoxVolume:
    return OrientedBoxVolume(center, half, Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), heading))


def _spaces_match(first, second) -> bool:
    return (first.shape == second.shape and first.dtype == second.dtype
            and np.array_equal(first.low, second.low) and np.array_equal(first.high, second.high))


def convert_custom_track(raw: object, catalog: RuntimeCatalog, *, reference_track_id: str = "straight-sprint-v1") -> ConvertedTrack:
    """Validate and convert one unregistered editor draft into a Gym factory."""
    clean, control_points, props = _parse_document(raw)
    spline = _CentripetalSpline(control_points)
    if spline.length <= 20.0:
        raise TrackValidationError("invalid_geometry", "playable route must be longer than 20 metres")
    curve_points = [spline.point_at(index / 400.0) for index in range(401)]
    curve_distances = [0.0]
    for first, second in pairwise(curve_points):
        curve_distances.append(curve_distances[-1] + hypot(second[0] - first[0], second[1] - first[1]))
    total = curve_distances[-1]
    if total > MAX_ROUTE_LENGTH_M:
        raise TrackValidationError("resource_limit", f"route length exceeds {MAX_ROUTE_LENGTH_M:g} metres")
    bridges = _auto_bridges(curve_points, curve_distances, total)
    if bridges:
        raise TrackValidationError(
            "unsupported_intersection",
            "automatic overpasses create stacked road decks, which the registered spatial physics contract does not support",
        )

    checkpoint_props = sorted((item for item in props if item["kind"] == "checkpoint"), key=lambda item: float(item["t"]))
    checkpoint_distances = [float(item["t"]) * total for item in checkpoint_props]
    if any(distance <= SPAWN_DISTANCE_M + 3.0 or total - distance <= 5.0 for distance in checkpoint_distances):
        raise TrackValidationError("checkpoint_order", "checkpoints must leave room after spawn and before finish")
    if any(second - first < MIN_CHECKPOINT_GAP_M for first, second in pairwise(checkpoint_distances)):
        raise TrackValidationError("checkpoint_order", "ordered checkpoints must be at least 6 metres apart")
    for item in props:
        if item["kind"] == "checkpoint" and (abs(item["offset"]) > 1e-9 or abs(item["rotation"]) > 1e-9):
            raise TrackValidationError("unsupported_checkpoint_transform", "checkpoint gates are centered and aligned to the route")
        if item["kind"] == "booster" and abs(item["rotation"]) > 1e-9:
            raise TrackValidationError("unsupported_booster_transform", "boost pads follow route direction and cannot be rotated independently")

    # Preview and navigation both use the same authoritative, arc-length road samples.
    preview_distances = [float(index) for index in np.arange(0.0, total, ROAD_PREVIEW_SPACING_M)] + [total]
    road_samples = []
    for distance in preview_distances:
        x, y, z, heading, nx, nz = _pose_at(curve_points, curve_distances, distance, props, bridges, total)
        road_samples.append({"x": x, "y": y, "z": z, "t": distance / total, "heading": heading, "nx": nx, "nz": nz})

    surface_distances = [SPAWN_DISTANCE_M]
    surface_distances.extend(float(distance) for distance in np.arange(SURFACE_SPACING_M + SPAWN_DISTANCE_M, total, SURFACE_SPACING_M))
    surface_distances.extend(distance for distance in checkpoint_distances if distance > SPAWN_DISTANCE_M)
    surface_distances.append(total)
    surface_distances = sorted(surface_distances)
    deduplicated = []
    for distance in surface_distances:
        if not deduplicated or distance - deduplicated[-1] >= 0.25:
            deduplicated.append(distance)
        elif distance == total:
            deduplicated[-1] = total
    if len(deduplicated) - 1 > MAX_SURFACES:
        raise TrackValidationError("resource_limit", f"converted road exceeds the {MAX_SURFACES} surface limit")

    route_points: list[Vec3] = []
    surface_ids: list[str] = []
    surfaces: list[PlanarSurfacePatch] = []
    for index, distance in enumerate(deduplicated):
        x, y, z, _, _, _ = _pose_at(curve_points, curve_distances, distance, props, bridges, total)
        route_points.append(Vec3(x, y, z))
        if index == 0:
            continue
        previous = route_points[index - 1]
        current = route_points[index]
        delta = current - previous
        length = delta.magnitude
        if length <= 1e-4:
            raise TrackValidationError("invalid_geometry", "converted navigation contains a zero-length segment")
        forward = delta / length
        horizontal = hypot(delta.x, delta.z)
        lateral = Vec3(delta.z / horizontal, 0.0, -delta.x / horizontal)
        normal = forward.cross(lateral).normalized()
        if normal.y < 0.7:
            raise TrackValidationError("unsupported_slope", "road slope exceeds the registered physics contact limit")
        surface_id = f"surface-{index - 1:04d}"
        patch = PlanarSurfacePatch(
            surface_id,
            (previous + current) * 0.5,
            forward,
            normal,
            float(clean["width"]),
            length,
        )
        surfaces.append(patch)
        surface_ids.append(surface_id)

    checkpoint_point_indices = tuple(
        min(range(len(deduplicated)), key=lambda index: abs(deduplicated[index] - distance))
        for distance in checkpoint_distances
    )
    if any(deduplicated[index] - distance > 0.2 for index, distance in zip(checkpoint_point_indices, checkpoint_distances, strict=True)):
        raise TrackValidationError("checkpoint_order", "checkpoint could not be aligned to a route point")
    navigation = AuthoredNavigation(tuple(route_points), tuple(surface_ids), checkpoint_point_indices)

    checkpoints = []
    for order, (item, index) in enumerate(zip(checkpoint_props, checkpoint_point_indices, strict=True)):
        _, _, _, heading, _, _ = _pose_at(curve_points, curve_distances, checkpoint_distances[order], props, bridges, total)
        checkpoints.append(Checkpoint(
            f"checkpoint-{order:03d}",
            _volume(route_points[index], Vec3(float(clean["width"]) / 2.0 + 1.0, 3.0, 1.0), heading),
            order,
        ))
    finish_pose = _pose_at(curve_points, curve_distances, total, props, bridges, total)
    finish = FinishLine("finish", _volume(route_points[-1], Vec3(float(clean["width"]) / 2.0 + 1.0, 3.0, 1.0), finish_pose[3]))
    obstacles = []
    boost_pads = []
    for index, item in enumerate(props):
        kind = item["kind"]
        if kind in ("bridge", "ramp", "checkpoint"):
            continue
        distance = float(item["t"]) * total
        x, y, z, heading, nx, nz = _pose_at(curve_points, curve_distances, distance, props, bridges, total)
        x += nx * float(item["offset"])
        z += nz * float(item["offset"])
        heading += float(item["rotation"])
        identity = f"{kind}-{index:03d}"
        if kind == "barrier":
            obstacles.append(Obstacle(identity, _volume(Vec3(x, y + 0.93, z), Vec3(2.0, 0.75, 0.35), heading)))
        elif kind == "cone":
            obstacles.append(Obstacle(identity, _volume(Vec3(x, y + 0.66, z), Vec3(0.65, 0.60, 0.65), heading)))
        elif kind == "booster":
            boost_pads.append(BoostPad(identity, _volume(Vec3(x, y + 0.25, z), Vec3(2.4, 0.2, 3.1), heading), 3.0))

    first, second = route_points[0], route_points[1]
    first_forward = (second - first).normalized()
    if abs(first_forward.y) > 1e-9:
        raise TrackValidationError("unsupported_spawn", "spawn support must be a horizontal road segment")
    start_support = PlanarSurfacePatch(
        "surface-spawn-support",
        first - first_forward * (SPAWN_DISTANCE_M / 2.0),
        first_forward,
        Vec3(0.0, 1.0, 0.0),
        float(clean["width"]),
        SPAWN_DISTANCE_M,
    )
    last_forward = (route_points[-1] - route_points[-2]).normalized()
    finish_support = PlanarSurfacePatch(
        "surface-finish-support",
        route_points[-1] + last_forward * 2.5,
        last_forward,
        surfaces[-1].normal,
        float(clean["width"]),
        5.0,
    )
    road_surfaces = (start_support, *surfaces, finish_support)
    if _atlas(tuple(road_surfaces), 1e-6) is None:
        raise TrackValidationError(
            "unsupported_surface_intersection",
            "road surfaces overlap with inconsistent heights; split elevated sections away from turns and intersections",
        )
    world = TrackWorld(
        surfaces=tuple(road_surfaces),
        obstacles=tuple(obstacles),
        boost_pads=tuple(boost_pads),
        checkpoints=tuple(checkpoints),
        finish_lines=(finish,),
    )
    base = catalog.track_factory(reference_track_id)
    start = route_points[0]
    start_pose = _pose_at(curve_points, curve_distances, SPAWN_DISTANCE_M, props, bridges, total)
    first_contact = world.surface_at(start, 0.0)
    if first_contact is None or first_contact.normal != Vec3(0.0, 1.0, 0.0):
        raise TrackValidationError("unsupported_spawn", "spawn must sit on a horizontal supported road segment")
    orientation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), start_pose[3])
    vehicle_position = start + Vec3(0.0, base.vehicle_parameters.ride_height, 0.0)
    initial = WorldState(
        vehicle=VehicleState(vehicle_position, Vec3(0.0, 0.0, 0.0), orientation, Vec3(0.0, 0.0, 0.0), True),
        track=TrackState(frozenset()),
        interactions=InteractionState(frozenset(), frozenset()),
        score=ScoreState(0),
        suspension=SuspensionDynamicsState(vehicle_position.y),
        flight=FlightState("grounded"),
    )
    factory = replace(
        base,
        track=world,
        initial_state=initial,
        navigation=navigation,
        offtrack_config=None,
        reset_config=None,
        reset_track_id=None,
    )
    try:
        custom_env = factory.create()
        reference_env = base.create()
        try:
            if not _spaces_match(custom_env.action_space, reference_env.action_space) or not _spaces_match(custom_env.observation_space, reference_env.observation_space):
                ppo_compatible = False
            else:
                ppo_compatible = True
            observation, _ = custom_env.reset(seed=0)
            if not np.isfinite(observation).all() or custom_env.runtime_snapshot()["tick"] != 0:
                raise TrackValidationError("invalid_spawn", "custom Gym environment did not reset to a finite initial state")
        finally:
            custom_env.close()
            reference_env.close()
    except TrackValidationError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise TrackValidationError("incompatible_physics", f"custom track cannot use the registered development physics contract: {error}") from error

    content = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    track_sha256 = sha256(CONVERSION_SCHEMA.encode("ascii") + b"\0" + content).hexdigest()
    track_id = f"sandbox-{track_sha256[:16]}"
    clean["id"] = track_id
    warnings = ("Development-only custom track; results are not scientific evidence.",)
    return ConvertedTrack(
        CONVERSION_SCHEMA,
        True,
        track_id,
        track_sha256,
        clean,
        factory,
        tuple(road_samples),
        len(bridges),
        ppo_compatible,
        warnings,
    )
