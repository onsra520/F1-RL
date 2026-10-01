from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any, Callable

from f1rl.sandbox.entities.boost_pad import BoostPad
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.entities.score_pickup import ScorePickup
from f1rl.sandbox.entities.slow_zone import SlowZone
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.volume import OrientedBoxVolume

from .canonical import _build_track_asset
from .model import SpawnPose, TrackAsset

_ROOT_KEYS = {
    "schema_version",
    "track_id",
    "spawn",
    "surfaces",
    "obstacles",
    "boost_pads",
    "slow_zones",
    "checkpoints",
    "finish_lines",
    "score_pickups",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {token}")


def _read_strict_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise ValueError(f"{name} missing required keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {', '.join(sorted(unknown))}")


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _vec3(value: Any, name: str) -> Vec3:
    items = _array(value, name)
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    return Vec3(*(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(items)))


def _quaternion(value: Any, name: str) -> Quaternion:
    items = _array(value, name)
    if len(items) != 4:
        raise ValueError(f"{name} must contain exactly 4 numbers")
    try:
        return Quaternion(*(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(items)))
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _volume(value: Any, name: str) -> OrientedBoxVolume:
    data = _mapping(value, name)
    _exact_keys(data, {"center", "half_extents", "orientation"}, name)
    try:
        return OrientedBoxVolume(
            center=_vec3(data["center"], f"{name}.center"),
            half_extents=_vec3(data["half_extents"], f"{name}.half_extents"),
            orientation=_quaternion(data["orientation"], f"{name}.orientation"),
        )
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _item(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    data = _mapping(value, name)
    _exact_keys(data, expected, name)
    return data


def _build_many(values: Any, name: str, builder: Callable[[dict[str, Any], str], Any]) -> tuple[Any, ...]:
    items = _array(values, name)
    built = []
    for index, value in enumerate(items):
        item_name = f"{name}[{index}]"
        try:
            built.append(builder(_mapping(value, item_name), item_name))
        except ValueError as exc:
            if str(exc).startswith(f"{item_name}:"):
                raise
            raise ValueError(f"{item_name}: {exc}") from exc
    return tuple(built)


def _surface(data: dict[str, Any], name: str) -> PlanarSurfacePatch:
    _exact_keys(data, {"id", "origin", "forward", "normal", "width", "length"}, name)
    return PlanarSurfacePatch(
        id=_text(data["id"], f"{name}.id"),
        origin=_vec3(data["origin"], f"{name}.origin"),
        forward=_vec3(data["forward"], f"{name}.forward"),
        normal=_vec3(data["normal"], f"{name}.normal"),
        width=_finite_float(data["width"], f"{name}.width"),
        length=_finite_float(data["length"], f"{name}.length"),
    )


def _obstacle(data: dict[str, Any], name: str) -> Obstacle:
    _exact_keys(data, {"id", "volume"}, name)
    return Obstacle(_text(data["id"], f"{name}.id"), _volume(data["volume"], f"{name}.volume"))


def _boost_pad(data: dict[str, Any], name: str) -> BoostPad:
    _exact_keys(data, {"id", "volume", "acceleration_bonus"}, name)
    return BoostPad(
        _text(data["id"], f"{name}.id"),
        _volume(data["volume"], f"{name}.volume"),
        _finite_float(data["acceleration_bonus"], f"{name}.acceleration_bonus"),
    )


def _slow_zone(data: dict[str, Any], name: str) -> SlowZone:
    _exact_keys(data, {"id", "volume", "drag_multiplier"}, name)
    return SlowZone(
        _text(data["id"], f"{name}.id"),
        _volume(data["volume"], f"{name}.volume"),
        _finite_float(data["drag_multiplier"], f"{name}.drag_multiplier"),
    )


def _checkpoint(data: dict[str, Any], name: str) -> Checkpoint:
    _exact_keys(data, {"id", "volume", "order"}, name)
    return Checkpoint(
        _text(data["id"], f"{name}.id"),
        _volume(data["volume"], f"{name}.volume"),
        _strict_int(data["order"], f"{name}.order"),
    )


def _finish_line(data: dict[str, Any], name: str) -> FinishLine:
    _exact_keys(data, {"id", "volume"}, name)
    return FinishLine(_text(data["id"], f"{name}.id"), _volume(data["volume"], f"{name}.volume"))


def _score_pickup(data: dict[str, Any], name: str) -> ScorePickup:
    _exact_keys(data, {"id", "volume", "points"}, name)
    return ScorePickup(
        _text(data["id"], f"{name}.id"),
        _volume(data["volume"], f"{name}.volume"),
        _strict_int(data["points"], f"{name}.points"),
    )


def load_track_asset(path: str | Path) -> TrackAsset:
    source = Path(path)
    root = _mapping(_read_strict_json(source), "track")
    _exact_keys(root, _ROOT_KEYS, "track")

    schema_version = _strict_int(root["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be integer 1")
    track_id = _text(root["track_id"], "track_id")

    spawn_data = _item(root["spawn"], {"position", "orientation"}, "spawn")
    spawn = SpawnPose(
        _vec3(spawn_data["position"], "spawn.position"),
        _quaternion(spawn_data["orientation"], "spawn.orientation"),
    )

    world = TrackWorld(
        surfaces=_build_many(root["surfaces"], "surfaces", _surface),
        obstacles=_build_many(root["obstacles"], "obstacles", _obstacle),
        boost_pads=_build_many(root["boost_pads"], "boost_pads", _boost_pad),
        slow_zones=_build_many(root["slow_zones"], "slow_zones", _slow_zone),
        checkpoints=_build_many(root["checkpoints"], "checkpoints", _checkpoint),
        finish_lines=_build_many(root["finish_lines"], "finish_lines", _finish_line),
        score_pickups=_build_many(root["score_pickups"], "score_pickups", _score_pickup),
    )
    return _build_track_asset(
        schema_version=schema_version,
        track_id=track_id,
        spawn=spawn,
        world=world,
    )
