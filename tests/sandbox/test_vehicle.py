from dataclasses import FrozenInstanceError
from math import pi

import pytest

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState


def valid_parameters(**overrides: float) -> VehicleParameters:
    values = {
        "wheelbase": 2.5,
        "ride_height": 0.3,
        "max_steering_angle": 0.6,
        "max_acceleration": 12.0,
        "max_braking": 20.0,
        "rolling_drag": 0.1,
        "air_drag": 0.02,
        "max_ground_speed": 90.0,
        "gravity": 9.81,
    }
    values.update(overrides)
    return VehicleParameters(**values)


def test_vehicle_state_speed_is_derived_from_velocity() -> None:
    state = VehicleState(Vec3(1.0, 2.0, 3.0), Vec3(3.0, 4.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True)
    assert state.speed == pytest.approx(5.0)


def test_vehicle_domain_values_are_frozen() -> None:
    state = VehicleState(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True)
    control = VehicleControl(0.0, 0.0, 0.0)

    with pytest.raises(FrozenInstanceError):
        state.grounded = False
    with pytest.raises(FrozenInstanceError):
        control.throttle = 1.0


def test_vehicle_control_validates_normalized_ranges() -> None:
    VehicleControl(-1.0, 0.0, 1.0)
    VehicleControl(1.0, 1.0, 0.0)

    for values in [(-1.01, 0.0, 0.0), (1.01, 0.0, 0.0), (0.0, -0.01, 0.0), (0.0, 1.01, 0.0), (0.0, 0.0, -0.01), (0.0, 0.0, 1.01)]:
        with pytest.raises(ValueError):
            VehicleControl(*values)


def test_vehicle_parameters_validate_positive_dimensions_and_gravity() -> None:
    for field in ("wheelbase", "ride_height", "gravity"):
        with pytest.raises(ValueError):
            valid_parameters(**{field: 0.0})
        with pytest.raises(ValueError):
            valid_parameters(**{field: -1.0})


def test_vehicle_parameters_reject_invalid_limits() -> None:
    for field in ("max_acceleration", "max_braking", "rolling_drag", "air_drag", "max_ground_speed"):
        with pytest.raises(ValueError):
            valid_parameters(**{field: -0.01})

    with pytest.raises(ValueError):
        valid_parameters(max_steering_angle=-0.01)
    with pytest.raises(ValueError):
        valid_parameters(max_steering_angle=pi / 2.0)


def test_vehicle_parameters_reject_non_finite_values() -> None:
    with pytest.raises(ValueError):
        valid_parameters(max_acceleration=float("nan"))
    with pytest.raises(ValueError):
        valid_parameters(gravity=float("inf"))
