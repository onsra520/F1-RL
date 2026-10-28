from math import inf, nan

import pytest

from src.sandbox.dynamics.air_dynamics import compute_air_motion
from src.sandbox.dynamics.ground_dynamics import compute_ground_motion
from src.sandbox.dynamics.modifiers import MotionModifiers
from src.sandbox.dynamics.vehicle_dynamics import compute_motion
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.surface import SurfaceContact
from src.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState


def state(velocity: Vec3 = Vec3(0.0, 0.0, 10.0)) -> VehicleState:
    return VehicleState(Vec3(0.0, 0.3, 0.0), velocity, Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True)


def params(*, rolling_drag: float = 0.0) -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 0.0, 20.0, rolling_drag, 0.02, 50.0, 9.81)


def flat() -> SurfaceContact:
    return SurfaceContact(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))


def test_motion_modifier_validation() -> None:
    for value in (nan, inf):
        with pytest.raises(ValueError):
            MotionModifiers(longitudinal_acceleration_bonus=value)
    with pytest.raises(ValueError):
        MotionModifiers(longitudinal_acceleration_bonus=-0.1)
    with pytest.raises(ValueError):
        MotionModifiers(rolling_drag_multiplier=0.99)


def test_boost_adds_grounded_longitudinal_acceleration() -> None:
    motion = compute_ground_motion(state(), VehicleControl(0.0, 0.0, 0.0), params(), flat(), MotionModifiers(5.0, 1.0))
    assert motion.linear_acceleration == Vec3(0.0, 0.0, 5.0)


def test_slow_zone_multiplies_rolling_drag() -> None:
    baseline = compute_ground_motion(state(), VehicleControl(0.0, 0.0, 0.0), params(rolling_drag=0.5), flat())
    slowed = compute_ground_motion(state(), VehicleControl(0.0, 0.0, 0.0), params(rolling_drag=0.5), flat(), MotionModifiers(0.0, 3.0))
    assert baseline.linear_acceleration.z == -5.0
    assert slowed.linear_acceleration.z == -15.0


def test_airborne_motion_ignores_ground_modifiers() -> None:
    airborne = VehicleState(state().position, Vec3(1.0, 2.0, 3.0), state().orientation, Vec3(0.0, 0.1, 0.0), False)
    control = VehicleControl(1.0, 1.0, 1.0)
    baseline = compute_air_motion(airborne, control, params())
    modified = compute_air_motion(airborne, control, params(), MotionModifiers(50.0, 9.0))
    assert modified == baseline


def test_compute_motion_omitted_modifier_preserves_behavior() -> None:
    control = VehicleControl(0.2, 0.3, 0.0)
    without = compute_motion(state(), control, params(rolling_drag=0.1), flat())
    explicit_default = compute_motion(state(), control, params(rolling_drag=0.1), flat(), MotionModifiers())
    assert without == explicit_default
