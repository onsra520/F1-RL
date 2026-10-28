from dataclasses import FrozenInstanceError
from math import pi

import pytest

from src.simulation.entities.boost_pad import BoostPad
from src.simulation.entities.checkpoint import Checkpoint
from src.simulation.entities.finish_line import FinishLine
from src.simulation.entities.obstacle import Obstacle
from src.simulation.entities.score_pickup import ScorePickup
from src.simulation.entities.slow_zone import SlowZone
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.perception.rangefinder import RangefinderReading, RangefinderSpec, sense_rangefinder
from src.simulation.track.model import TrackWorld
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.vehicle import VehicleState


def _volume(center: Vec3) -> OrientedBoxVolume:
    return OrientedBoxVolume(center, Vec3(1.0, 1.0, 1.0), Quaternion.identity())


def _vehicle(*, orientation: Quaternion | None = None) -> VehicleState:
    return VehicleState(
        Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0),
        orientation or Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True,
    )


def _spec() -> RangefinderSpec:
    return RangefinderSpec("front", Vec3(0.0, 0.0, 1.0), 10.0)


def test_rangefinder_spec_is_frozen_and_validates_inputs() -> None:
    spec = _spec()
    with pytest.raises(FrozenInstanceError):
        spec.id = "changed"  # type: ignore[misc]

    for sensor_id in ("", "   "):
        with pytest.raises(ValueError):
            RangefinderSpec(sensor_id, Vec3(0.0, 0.0, 1.0), 10.0)
    for direction in (Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 2.0)):
        with pytest.raises(ValueError):
            RangefinderSpec("front", direction, 10.0)
    for max_distance in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            RangefinderSpec("front", Vec3(0.0, 0.0, 1.0), max_distance)  # type: ignore[arg-type]


def test_rangefinder_reading_is_frozen_raw_value_object() -> None:
    reading = RangefinderReading("front", 3.25, "wall")
    assert reading == RangefinderReading("front", 3.25, "wall")
    with pytest.raises(FrozenInstanceError):
        reading.distance = 4.0  # type: ignore[misc]


def test_no_hit_reports_configured_max_distance() -> None:
    reading = sense_rangefinder(_spec(), TrackWorld(), _vehicle())
    assert reading == RangefinderReading("front", 10.0, None)


def test_nearest_obstacle_wins_over_lexically_earlier_far_obstacle() -> None:
    track = TrackWorld(obstacles=(
        Obstacle("a-far", _volume(Vec3(0.0, 0.0, 8.0))),
        Obstacle("z-near", _volume(Vec3(0.0, 0.0, 4.0))),
    ))
    assert sense_rangefinder(_spec(), track, _vehicle()) == RangefinderReading(
        "front", pytest.approx(3.0), "z-near"
    )


def test_equal_distance_obstacles_resolve_by_lexical_id() -> None:
    shared = _volume(Vec3(0.0, 0.0, 4.0))
    track = TrackWorld(obstacles=(Obstacle("z-wall", shared), Obstacle("a-wall", shared)))
    reading = sense_rangefinder(_spec(), track, _vehicle())
    assert reading.distance == pytest.approx(3.0)
    assert reading.hit_id == "a-wall"


def test_origin_inside_obstacle_reports_zero_distance() -> None:
    track = TrackWorld(obstacles=(Obstacle("inside", _volume(Vec3(0.0, 0.0, 0.0))),))
    assert sense_rangefinder(_spec(), track, _vehicle()) == RangefinderReading(
        "front", 0.0, "inside"
    )


def test_local_forward_sensor_rotates_with_vehicle_orientation() -> None:
    orientation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 2.0)
    track = TrackWorld(obstacles=(Obstacle("east", _volume(Vec3(5.0, 0.0, 0.0))),))
    reading = sense_rangefinder(_spec(), track, _vehicle(orientation=orientation))
    assert reading.distance == pytest.approx(4.0)
    assert reading.hit_id == "east"


def test_non_solid_track_entities_are_not_rangefinder_hits() -> None:
    shared = _volume(Vec3(0.0, 0.0, 4.0))
    track = TrackWorld(
        boost_pads=(BoostPad("boost", shared, 2.0),),
        slow_zones=(SlowZone("slow", shared, 2.0),),
        checkpoints=(Checkpoint("checkpoint", shared, 0),),
        finish_lines=(FinishLine("finish", shared),),
        score_pickups=(ScorePickup("pickup", shared, 10),),
    )
    assert sense_rangefinder(_spec(), track, _vehicle()) == RangefinderReading(
        "front", 10.0, None
    )
