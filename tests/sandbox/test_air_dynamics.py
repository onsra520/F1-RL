import pytest

from src.simulation.dynamics.air_dynamics import compute_air_motion
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleControl, VehicleParameters, VehicleState


def parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 12.0, 20.0, 0.1, 0.02, 50.0, 9.81)


def airborne_state() -> VehicleState:
    return VehicleState(
        Vec3(2.0, 5.0, 3.0),
        Vec3(4.0, 6.0, 8.0),
        Quaternion.identity(),
        Vec3(0.1, 0.2, 0.3),
        False,
    )


def test_air_motion_applies_gravity_and_linear_drag() -> None:
    motion = compute_air_motion(airborne_state(), VehicleControl(0.0, 0.0, 0.0), parameters())
    assert motion.linear_acceleration.x == pytest.approx(-0.08)
    assert motion.linear_acceleration.y == pytest.approx(-9.93)
    assert motion.linear_acceleration.z == pytest.approx(-0.16)


def test_air_motion_preserves_angular_velocity() -> None:
    motion = compute_air_motion(airborne_state(), VehicleControl(0.0, 0.0, 0.0), parameters())
    assert motion.angular_velocity == Vec3(0.1, 0.2, 0.3)


def test_airborne_controls_have_no_direct_effect() -> None:
    state = airborne_state()
    idle = compute_air_motion(state, VehicleControl(0.0, 0.0, 0.0), parameters())
    aggressive = compute_air_motion(state, VehicleControl(1.0, 1.0, 1.0), parameters())
    assert aggressive == idle


def test_air_motion_is_deterministic() -> None:
    args = (airborne_state(), VehicleControl(-0.4, 0.7, 0.2), parameters())
    assert compute_air_motion(*args) == compute_air_motion(*args)
