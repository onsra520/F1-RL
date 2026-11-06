from __future__ import annotations

import hashlib
import json
from math import isfinite

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.volume import OrientedBoxVolume

from .model import AuthoredNavigation, SpawnPose, TrackAsset


def _number(value: float) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError("canonical numbers must be finite")
    return 0.0 if number == 0.0 else number


def _vec3(value: Vec3) -> list[float]:
    return [_number(value.x), _number(value.y), _number(value.z)]


def _quaternion(value: Quaternion) -> list[float]:
    components = [_number(value.w), _number(value.x), _number(value.y), _number(value.z)]
    for component in components:
        if component == 0.0:
            continue
        if component < 0.0:
            components = [_number(-item) for item in components]
        break
    return components


def _volume(value: OrientedBoxVolume) -> dict[str, object]:
    return {
        "center": _vec3(value.center),
        "half_extents": _vec3(value.half_extents),
        "orientation": _quaternion(value.orientation),
    }


def _world_payload(world: TrackWorld) -> dict[str, object]:
    return {
        "surfaces": [
            {
                "id": item.id,
                "origin": _vec3(item.origin),
                "forward": _vec3(item.forward),
                "normal": _vec3(item.normal),
                "width": _number(item.width),
                "length": _number(item.length),
            }
            for item in world.surfaces
        ],
        "obstacles": [
            {"id": item.id, "volume": _volume(item.volume)}
            for item in world.obstacles
        ],
        "boost_pads": [
            {
                "id": item.id,
                "volume": _volume(item.volume),
                "acceleration_bonus": _number(item.acceleration_bonus),
            }
            for item in world.boost_pads
        ],
        "slow_zones": [
            {
                "id": item.id,
                "volume": _volume(item.volume),
                "drag_multiplier": _number(item.drag_multiplier),
            }
            for item in world.slow_zones
        ],
        "checkpoints": [
            {"id": item.id, "volume": _volume(item.volume), "order": item.order}
            for item in world.checkpoints
        ],
        "finish_lines": [
            {"id": item.id, "volume": _volume(item.volume)}
            for item in world.finish_lines
        ],
        "score_pickups": [
            {"id": item.id, "volume": _volume(item.volume), "points": item.points}
            for item in world.score_pickups
        ],
    }


def _spawn_payload(spawn: SpawnPose) -> dict[str, object]:
    return {
        "position": _vec3(spawn.position),
        "orientation": _quaternion(spawn.orientation),
    }


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def fingerprint_track_world(world: TrackWorld) -> str:
    """Canonical geometry identity, independent of checkout line endings."""
    if not isinstance(world, TrackWorld):
        raise ValueError("world must be TrackWorld")
    return _sha256({"hash_schema": "f1rl-track-world-v1", "world": _world_payload(world)})


def _build_track_asset(
    *,
    schema_version: int,
    track_id: str,
    spawn: SpawnPose,
    world: TrackWorld,
    navigation: AuthoredNavigation | None = None,
) -> TrackAsset:
    if not isinstance(spawn, SpawnPose):
        raise ValueError("spawn must be SpawnPose")
    if not isinstance(world, TrackWorld):
        raise ValueError("world must be TrackWorld")

    canonical_world = _world_payload(world)
    world_sha256 = fingerprint_track_world(world)
    content_payload: dict[str, object] = {
        "hash_schema": "f1rl-track-content-v2" if schema_version == 2 else "f1rl-track-content-v1",
        "spawn": _spawn_payload(spawn),
        "world": canonical_world,
    }
    if schema_version == 2:
        if not isinstance(navigation, AuthoredNavigation):
            raise ValueError("schema version 2 requires navigation")
        content_payload["navigation"] = {
            "points": [_vec3(point) for point in navigation.points],
            "surface_ids": list(navigation.surface_ids),
            "checkpoint_point_indices": list(navigation.checkpoint_point_indices),
        }
    content_sha256 = _sha256(content_payload)
    return TrackAsset(schema_version, track_id, spawn, world, world_sha256, content_sha256, navigation)
