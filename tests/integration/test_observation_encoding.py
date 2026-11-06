from __future__ import annotations

import ast
from copy import deepcopy
from math import pi
from pathlib import Path

import pytest

from f1rl.learning.observation import NavigationTarget, ObservationConfig, ObservationEncoder
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.perception.rangefinder import RangefinderSpec
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleState


def _obstacle(obstacle_id: str, center: Vec3, orientation: Quaternion | None = None) -> Obstacle:
    return Obstacle(
        obstacle_id,
        OrientedBoxVolume(center, Vec3(1.0, 1.0, 1.0), orientation or Quaternion.identity()),
    )


def _vehicle(*, velocity: Vec3, orientation: Quaternion | None = None, grounded: bool = True) -> VehicleState:
    return VehicleState(
        Vec3(0.0, 0.0, 0.0), velocity, orientation or Quaternion.identity(),
        Vec3(0.0, 0.0, 0.0), grounded,
    )


def _suite() -> SensorSuite:
    return SensorSuite((
        RangefinderSpec("z-forward", Vec3(0.0, 0.0, 1.0), 10.0),
        RangefinderSpec("a-right", Vec3(1.0, 0.0, 0.0), 8.0),
    ))


def test_complete_observation_matches_feature_layout() -> None:
    track = TrackWorld(obstacles=(
        _obstacle("forward", Vec3(0.0, 0.0, 6.0)),
        _obstacle("right", Vec3(3.0, 0.0, 0.0)),
    ))
    encoder = ObservationEncoder(ObservationConfig(_suite(), 10.0, 10.0))
    observation = encoder.encode(
        track,
        _vehicle(velocity=Vec3(5.0, -2.0, 10.0)),
        NavigationTarget(Vec3(3.0, 0.0, 4.0)),
    )

    assert encoder.feature_names == (
        "rangefinder:z-forward", "rangefinder:a-right",
        "velocity_local_x", "velocity_local_y", "velocity_local_z",
        "target_direction_local_x", "target_direction_local_y", "target_direction_local_z",
        "target_distance", "target_available", "grounded",
    )
    assert observation == pytest.approx((0.5, 0.25, 0.5, -0.2, 1.0, 0.6, 0.0, 0.8, 0.5, 1.0, 1.0))


def test_encoding_is_deterministic_and_does_not_mutate_inputs() -> None:
    suite = _suite()
    config = ObservationConfig(suite, 10.0, 20.0)
    encoder = ObservationEncoder(config)
    track = TrackWorld(obstacles=(_obstacle("forward", Vec3(0.0, 0.0, 6.0)),))
    vehicle = _vehicle(velocity=Vec3(2.0, 0.0, 5.0), grounded=False)
    target = NavigationTarget(Vec3(3.0, 0.0, 4.0))
    before = deepcopy((track, vehicle, suite, target, config, encoder))

    first = encoder.encode(track, vehicle, target)
    second = encoder.encode(track, vehicle, target)
    equivalent = ObservationEncoder(
        ObservationConfig(deepcopy(suite), 10.0, 20.0)
    ).encode(deepcopy(track), deepcopy(vehicle), deepcopy(target))

    assert first == second == equivalent
    assert (track, vehicle, suite, target, config, encoder) == before


def test_consistent_rigid_yaw_rotation_preserves_observation() -> None:
    yaw = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 3.0)
    suite = _suite()
    encoder = ObservationEncoder(ObservationConfig(suite, 10.0, 20.0))
    centers = (Vec3(0.0, 0.0, 6.0), Vec3(3.0, 0.0, 0.0))
    track_a = TrackWorld(obstacles=(
        _obstacle("forward", centers[0]), _obstacle("right", centers[1]),
    ))
    track_b = TrackWorld(obstacles=(
        _obstacle("forward", yaw.rotate(centers[0]), yaw),
        _obstacle("right", yaw.rotate(centers[1]), yaw),
    ))
    velocity = Vec3(2.0, -1.0, 5.0)
    target = Vec3(3.0, 0.0, 8.0)

    observation_a = encoder.encode(
        track_a, _vehicle(velocity=velocity), NavigationTarget(target)
    )
    observation_b = encoder.encode(
        track_b,
        _vehicle(velocity=yaw.rotate(velocity), orientation=yaw),
        NavigationTarget(yaw.rotate(target)),
    )

    assert observation_b == pytest.approx(observation_a, abs=1e-12)


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_observation_module_has_no_forbidden_dependencies() -> None:
    path = Path("f1rl/learning/observation.py")
    modules = _imported_modules(path)
    forbidden_prefixes = (
        "f1rl.runtime", "f1rl.api", "f1rl.drivers", "runtime", "api", "drivers",
        "gymnasium", "numpy", "torch", "stable_baselines3", "fastapi", "websocket", "websockets",
    )
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in modules for prefix in forbidden_prefixes
    )

    source = path.read_text(encoding="utf-8").lower()
    assert not any(token in source for token in ("three.js", "react", "ppo"))


def test_sandbox_does_not_import_learning() -> None:
    offenders: list[str] = []
    for path in Path("f1rl/sandbox").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "f1rl.learning" or alias.name.startswith("f1rl.learning.") for alias in node.names):
                    offenders.append(str(path))
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if module == "f1rl.learning" or module.startswith("f1rl.learning."):
                    offenders.append(str(path))
                if node.level > 0 and (module == "learning" or module.startswith("learning.")):
                    offenders.append(str(path))
    assert offenders == []
