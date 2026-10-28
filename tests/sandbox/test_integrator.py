from math import pi, sqrt

import pytest

from src.sandbox.dynamics.integrator import integrate
from src.sandbox.dynamics.vehicle_dynamics import MotionDerivatives
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.surface import SurfaceContact
from src.sandbox.vehicle import VehicleParameters, VehicleState


def parameters(max_ground_speed: float = 50.0) -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 12.0, 20.0, 0.1, 0.02, max_ground_speed, 9.81)


def state(position: Vec3, velocity: Vec3, grounded: bool, angular_velocity: Vec3 | None = None) -> VehicleState:
    return VehicleState(
        position,
        velocity,
        Quaternion.identity(),
        angular_velocity or Vec3(0.0, 0.0, 0.0),
        grounded,
    )


def test_airborne_semi_implicit_euler_advances_velocity_then_position() -> None:
    initial = state(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 2.0, 3.0), False)
    derivatives = MotionDerivatives(Vec3(0.0, -10.0, 0.0), Vec3(0.0, 0.0, 0.0))

    result = integrate(initial, derivatives, parameters(), None, 0.5)

    assert result.velocity == Vec3(1.0, -3.0, 3.0)
    assert result.position == Vec3(0.5, -1.5, 1.5)
    assert result.grounded is False


def test_orientation_advances_from_world_angular_velocity() -> None:
    initial = state(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), False)
    derivatives = MotionDerivatives(Vec3(0.0, 0.0, 0.0), Vec3(0.0, pi, 0.0))

    result = integrate(initial, derivatives, parameters(), None, 0.5)
    forward = result.orientation.rotate(Vec3(0.0, 0.0, 1.0))

    assert forward.x == pytest.approx(1.0, abs=1e-9)
    assert forward.y == pytest.approx(0.0, abs=1e-9)
    assert forward.z == pytest.approx(0.0, abs=1e-9)


def test_integrator_requires_positive_dt() -> None:
    initial = state(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), False)
    derivatives = MotionDerivatives(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    with pytest.raises(ValueError):
        integrate(initial, derivatives, parameters(), None, 0.0)
    with pytest.raises(ValueError):
        integrate(initial, derivatives, parameters(), None, -0.1)


def test_ground_constraint_removes_lateral_slip() -> None:
    contact = SurfaceContact(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    initial = state(Vec3(0.0, 0.3, 0.0), Vec3(2.0, -3.0, 4.0), True)
    derivatives = MotionDerivatives(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    result = integrate(initial, derivatives, parameters(), contact, 1.0)

    assert result.velocity == Vec3(0.0, 0.0, 4.0)
    assert result.position == Vec3(0.0, 0.3, 4.0)
    assert result.grounded is True


def test_contact_loss_preserves_ramp_launch_velocity() -> None:
    ramp_velocity = Vec3(0.0, 5.0, 10.0)
    initial = state(Vec3(0.0, 1.0, 0.0), ramp_velocity, True)
    derivatives = MotionDerivatives(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    result = integrate(initial, derivatives, parameters(), None, 0.25)

    assert result.velocity == ramp_velocity
    assert result.position == Vec3(0.0, 2.25, 2.5)
    assert result.grounded is False


def test_landing_removes_normal_and_lateral_ground_velocity() -> None:
    contact = SurfaceContact(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    initial = state(Vec3(0.0, 1.0, 0.0), Vec3(2.0, -5.0, 4.0), False)
    derivatives = MotionDerivatives(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    result = integrate(initial, derivatives, parameters(), contact, 0.1)

    assert result.velocity == Vec3(0.0, 0.0, 4.0)
    assert result.position.x == pytest.approx(0.0)
    assert result.position.y == pytest.approx(0.3)
    assert result.position.z == pytest.approx(0.4)
    assert result.grounded is True


def test_ground_speed_is_capped_along_vehicle_heading() -> None:
    contact = SurfaceContact(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    initial = state(Vec3(0.0, 0.3, 0.0), Vec3(30.0, 0.0, 40.0), True)
    derivatives = MotionDerivatives(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    result = integrate(initial, derivatives, parameters(max_ground_speed=25.0), contact, 0.1)

    assert result.speed == pytest.approx(25.0)
    assert result.velocity == Vec3(0.0, 0.0, 25.0)


def test_ground_orientation_aligns_up_axis_to_surface_normal() -> None:
    normal = Vec3(0.0, sqrt(3.0) / 2.0, -0.5)
    contact = SurfaceContact(Vec3(0.0, 0.0, 0.0), normal)
    initial = state(Vec3(0.0, 0.3, 0.0), Vec3(0.0, 0.0, 5.0), True)
    derivatives = MotionDerivatives(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))

    result = integrate(initial, derivatives, parameters(), contact, 0.1)
    up = result.orientation.rotate(Vec3(0.0, 1.0, 0.0))

    assert up.x == pytest.approx(normal.x, abs=1e-9)
    assert up.y == pytest.approx(normal.y, abs=1e-9)
    assert up.z == pytest.approx(normal.z, abs=1e-9)
