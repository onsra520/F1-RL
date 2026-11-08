from math import pi

import pytest

from f1rl.learning.observation import ObservationConfig, ObservationEncoder
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.perception.rangefinder import RangefinderSpec
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleState


def _obstacle(obstacle_id: str, center: Vec3, half_extents: Vec3 = Vec3(1.0, 1.0, 1.0)) -> Obstacle:
    return Obstacle(obstacle_id, OrientedBoxVolume(center, half_extents, Quaternion.identity()))


def _vehicle(*, velocity: Vec3 = Vec3(0.0, 0.0, 0.0), orientation: Quaternion | None = None) -> VehicleState:
    return VehicleState(
        position=Vec3(0.0, 0.0, 0.0),
        velocity=velocity,
        orientation=orientation or Quaternion.identity(),
        angular_velocity=Vec3(0.0, 0.0, 0.0),
        grounded=False,
    )


def test_rangefinders_use_each_specs_max_distance_in_declared_order() -> None:
    suite = SensorSuite((
        RangefinderSpec("z-forward", Vec3(0.0, 0.0, 1.0), 8.0),
        RangefinderSpec("a-right", Vec3(1.0, 0.0, 0.0), 12.0),
        RangefinderSpec("m-left", Vec3(-1.0, 0.0, 0.0), 10.0),
    ))
    track = TrackWorld(obstacles=(
        _obstacle("forward", Vec3(0.0, 0.0, 5.0)),
        _obstacle("right", Vec3(4.0, 0.0, 0.0)),
    ))
    encoder = ObservationEncoder(ObservationConfig(suite, 50.0, 100.0))

    observation = encoder.encode(track, _vehicle(), None)

    assert observation[:3] == pytest.approx((0.5, 0.25, 1.0))
    assert observation[3:6] == pytest.approx((0.0, 0.0, 0.0))


def test_zero_distance_obstacle_encodes_zero() -> None:
    suite = SensorSuite((RangefinderSpec("front", Vec3(0.0, 0.0, 1.0), 10.0),))
    track = TrackWorld(obstacles=(_obstacle("inside", Vec3(0.0, 0.0, 0.0)),))
    encoder = ObservationEncoder(ObservationConfig(suite, 50.0, 100.0))

    assert encoder.encode(track, _vehicle(), None)[0] == 0.0


def test_velocity_is_projected_into_rotated_vehicle_local_axes() -> None:
    orientation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 2.0)
    vehicle = _vehicle(velocity=Vec3(20.0, -5.0, -10.0), orientation=orientation)
    encoder = ObservationEncoder(ObservationConfig(SensorSuite(()), 20.0, 100.0))

    observation = encoder.encode(TrackWorld(), vehicle, None)

    assert observation[:3] == pytest.approx((0.5, -0.25, 1.0), abs=1e-12)


def test_velocity_components_clamp_independently_with_signs_preserved() -> None:
    vehicle = _vehicle(velocity=Vec3(-100.0, 60.0, 125.0))
    encoder = ObservationEncoder(ObservationConfig(SensorSuite(()), 50.0, 100.0))

    observation = encoder.encode(TrackWorld(), vehicle, None)

    assert observation[:3] == (-1.0, 1.0, 1.0)
