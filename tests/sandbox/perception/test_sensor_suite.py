from dataclasses import FrozenInstanceError

import pytest

from src.sandbox.entities.obstacle import Obstacle
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.perception.rangefinder import RangefinderSpec, sense_rangefinder
from src.sandbox.perception.sensor_suite import SensorSuite
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState


def _vehicle() -> VehicleState:
    return VehicleState(
        Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0),
        Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True,
    )


def _obstacle(id_: str, center: Vec3) -> Obstacle:
    return Obstacle(
        id_, OrientedBoxVolume(center, Vec3(1.0, 1.0, 1.0), Quaternion.identity())
    )


def _specs() -> tuple[RangefinderSpec, ...]:
    return (
        RangefinderSpec("z-forward", Vec3(0.0, 0.0, 1.0), 10.0),
        RangefinderSpec("a-right", Vec3(1.0, 0.0, 0.0), 10.0),
        RangefinderSpec("m-left", Vec3(-1.0, 0.0, 0.0), 10.0),
    )


def test_sensor_suite_validates_tuple_duplicates_and_preserves_order() -> None:
    specs = _specs()
    suite = SensorSuite(specs)
    assert suite.rangefinders is specs
    assert SensorSuite(()) == SensorSuite(())
    with pytest.raises(ValueError):
        SensorSuite(list(specs))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SensorSuite((specs[0], specs[0]))
    with pytest.raises(FrozenInstanceError):
        suite.rangefinders = ()  # type: ignore[misc]


def test_sense_returns_readings_in_declared_not_lexical_order() -> None:
    specs = _specs()
    suite = SensorSuite(specs)
    track = TrackWorld(obstacles=(
        _obstacle("front-wall", Vec3(0.0, 0.0, 5.0)),
        _obstacle("right-wall", Vec3(5.0, 0.0, 0.0)),
        _obstacle("left-wall", Vec3(-5.0, 0.0, 0.0)),
    ))
    vehicle = _vehicle()

    readings = suite.sense(track, vehicle)

    assert tuple(reading.sensor_id for reading in readings) == (
        "z-forward", "a-right", "m-left"
    )
    assert readings == tuple(sense_rangefinder(spec, track, vehicle) for spec in specs)


def test_repeated_sensing_is_equal_and_does_not_mutate_inputs() -> None:
    specs = _specs()
    suite = SensorSuite(specs)
    track = TrackWorld(obstacles=(_obstacle("front-wall", Vec3(0.0, 0.0, 5.0)),))
    vehicle = _vehicle()
    suite_before = suite
    track_before = track
    vehicle_before = vehicle
    specs_before = specs

    first = suite.sense(track, vehicle)
    second = suite.sense(track, vehicle)

    assert first == second
    assert suite == suite_before
    assert track == track_before
    assert vehicle == vehicle_before
    assert specs == specs_before
