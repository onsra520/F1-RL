from dataclasses import fields
from math import pi
from pathlib import Path

import pytest

from src.sandbox.entities.obstacle import Obstacle
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.perception.rangefinder import RangefinderReading, RangefinderSpec
from src.sandbox.perception.sensor_suite import SensorSuite
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState


def _obstacle(id_: str, center: Vec3) -> Obstacle:
    return Obstacle(
        id_, OrientedBoxVolume(center, Vec3(1.0, 1.0, 1.0), Quaternion.identity())
    )


def _rotated_vehicle() -> VehicleState:
    return VehicleState(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 2.0),
        Vec3(0.0, 0.0, 0.0),
        True,
    )


def _suite() -> SensorSuite:
    return SensorSuite((
        RangefinderSpec("z-forward", Vec3(0.0, 0.0, 1.0), 20.0),
        RangefinderSpec("a-right", Vec3(1.0, 0.0, 0.0), 20.0),
        RangefinderSpec("m-left", Vec3(-1.0, 0.0, 0.0), 20.0),
    ))


def _track() -> TrackWorld:
    return TrackWorld(obstacles=(
        _obstacle("east", Vec3(6.0, 0.0, 0.0)),
        _obstacle("south", Vec3(0.0, 0.0, -4.0)),
        _obstacle("north", Vec3(0.0, 0.0, 8.0)),
    ))


def test_rotated_vehicle_multi_sensor_flow_preserves_declared_order() -> None:
    readings = _suite().sense(_track(), _rotated_vehicle())

    assert tuple(reading.sensor_id for reading in readings) == (
        "z-forward", "a-right", "m-left"
    )
    assert tuple(reading.hit_id for reading in readings) == ("east", "south", "north")
    assert tuple(reading.distance for reading in readings) == pytest.approx((5.0, 3.0, 7.0))


def test_equivalent_suites_produce_exactly_equal_repeated_readings() -> None:
    left = _suite()
    right = _suite()
    track = _track()
    vehicle = _rotated_vehicle()

    assert left.sense(track, vehicle) == left.sense(track, vehicle) == right.sense(track, vehicle)


def test_readings_are_raw_physical_measurements_only() -> None:
    readings = _suite().sense(_track(), _rotated_vehicle())
    assert readings[0].distance == pytest.approx(5.0)
    assert readings[0].distance > 1.0
    assert tuple(field.name for field in fields(RangefinderReading)) == (
        "sensor_id", "distance", "hit_id"
    )


def test_perception_source_has_no_forbidden_dependencies() -> None:
    package_dir = Path(__file__).parents[2] / "src" / "sandbox" / "perception"
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in package_dir.glob("*.py"))
    forbidden = (
        "src.learning",
        "src.runtime",
        "src.api",
        "src.drivers",
        "gymnasium",
        "ppo",
        "fastapi",
        "websocket",
        "react",
        "three.js",
        "import random",
        "from random",
        "import time",
        "from time",
        "threading",
        "asyncio",
        "socket",
    )
    assert all(token not in source for token in forbidden)
