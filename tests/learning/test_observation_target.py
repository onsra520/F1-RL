from math import isfinite, pi

import pytest

from src.learning.observation import NavigationTarget, ObservationConfig, ObservationEncoder
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.perception.rangefinder import RangefinderSpec
from src.sandbox.perception.sensor_suite import SensorSuite
from src.sandbox.track.model import TrackWorld
from src.sandbox.vehicle import VehicleState


def _vehicle(
    *,
    position: Vec3 = Vec3(0.0, 0.0, 0.0),
    velocity: Vec3 = Vec3(0.0, 0.0, 0.0),
    orientation: Quaternion | None = None,
    grounded: bool = False,
) -> VehicleState:
    return VehicleState(
        position=position,
        velocity=velocity,
        orientation=orientation or Quaternion.identity(),
        angular_velocity=Vec3(0.0, 0.0, 0.0),
        grounded=grounded,
    )


def _encoder(*, sensors: SensorSuite | None = None, velocity_scale: float = 50.0, target_scale: float = 100.0) -> ObservationEncoder:
    return ObservationEncoder(ObservationConfig(sensors or SensorSuite(()), velocity_scale, target_scale))


def test_target_direction_uses_vehicle_local_axes() -> None:
    encoder = _encoder()
    identity = encoder.encode(
        TrackWorld(),
        _vehicle(),
        NavigationTarget(Vec3(10.0, 0.0, 0.0)),
    )
    assert identity[3:6] == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)

    yawed = encoder.encode(
        TrackWorld(),
        _vehicle(orientation=Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 2.0)),
        NavigationTarget(Vec3(10.0, 0.0, 0.0)),
    )
    assert yawed[3:6] == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_target_at_vehicle_position_remains_available() -> None:
    vehicle = _vehicle(position=Vec3(4.0, 2.0, -3.0))
    observation = _encoder().encode(
        TrackWorld(), vehicle, NavigationTarget(vehicle.position)
    )
    assert observation[3:6] == (0.0, 0.0, 0.0)
    assert observation[6] == 0.0
    assert observation[7] == 1.0


def test_target_distance_normalizes_and_clamps() -> None:
    encoder = _encoder(target_scale=100.0)
    near = encoder.encode(
        TrackWorld(), _vehicle(), NavigationTarget(Vec3(0.0, 0.0, 50.0))
    )
    far = encoder.encode(
        TrackWorld(), _vehicle(), NavigationTarget(Vec3(0.0, 0.0, 250.0))
    )
    assert near[6] == pytest.approx(0.5)
    assert far[6] == 1.0
    assert near[7] == far[7] == 1.0


def test_absent_target_uses_zero_direction_distance_and_availability() -> None:
    observation = _encoder().encode(TrackWorld(), _vehicle(), None)
    assert observation[3:6] == (0.0, 0.0, 0.0)
    assert observation[6] == 0.0
    assert observation[7] == 0.0


@pytest.mark.parametrize("grounded,expected", [(False, 0.0), (True, 1.0)])
def test_grounded_flag_is_exact_binary_value(grounded: bool, expected: float) -> None:
    observation = _encoder().encode(TrackWorld(), _vehicle(grounded=grounded), None)
    assert observation[8] == expected


def test_complete_observation_is_finite_bounded_and_matches_size() -> None:
    suite = SensorSuite((RangefinderSpec("front", Vec3(0.0, 0.0, 1.0), 10.0),))
    encoder = _encoder(sensors=suite, velocity_scale=10.0, target_scale=20.0)
    observation = encoder.encode(
        TrackWorld(),
        _vehicle(velocity=Vec3(-1000.0, 500.0, 250.0), grounded=True),
        NavigationTarget(Vec3(1000.0, 1000.0, 1000.0)),
    )

    assert len(observation) == encoder.size == 10
    assert all(isfinite(value) and -1.0 <= value <= 1.0 for value in observation)
    assert 0.0 <= observation[0] <= 1.0
    assert 0.0 <= observation[7] <= 1.0
    assert observation[8] in (0.0, 1.0)
    assert observation[9] in (0.0, 1.0)
