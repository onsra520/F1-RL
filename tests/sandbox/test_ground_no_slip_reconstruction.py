"""C1 recovered ground-no-slip contract: no lateral drift, preserved air motion."""
from math import cos, pi, sin

import pytest

from f1rl.sandbox.dynamics.integrator import integrate
from f1rl.sandbox.dynamics.vehicle_dynamics import MotionDerivatives
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.surface import SurfaceContact
from f1rl.sandbox.vehicle import VehicleParameters, VehicleState


ZERO = Vec3(0.0, 0.0, 0.0)
FORWARD = Vec3(0.0, 0.0, 1.0)


def _params(speed_limit: float = 50.0) -> VehicleParameters:
    return VehicleParameters(
        wheelbase=2.5,
        ride_height=0.3,
        max_steering_angle=0.6,
        max_acceleration=12.0,
        max_braking=20.0,
        rolling_drag=0.1,
        air_drag=0.02,
        max_ground_speed=speed_limit,
        gravity=9.81,
    )


def _state(velocity: Vec3, *, grounded: bool = True) -> VehicleState:
    return VehicleState(
        position=Vec3(0.0, 0.3, 0.0),
        velocity=velocity,
        orientation=Quaternion.identity(),
        angular_velocity=ZERO,
        grounded=grounded,
    )


def _flat() -> SurfaceContact:
    return SurfaceContact(point=ZERO, normal=Vec3(0.0, 1.0, 0.0))


def _integrate(velocity: Vec3, angular_velocity: Vec3 = ZERO, *, contact=None, speed_limit=50.0):
    return integrate(
        _state(velocity),
        MotionDerivatives(ZERO, angular_velocity),
        _params(speed_limit),
        _flat() if contact is None else contact,
        0.1,
    )


def test_ground_transition_erases_lateral_slip_instead_of_preserving_it() -> None:
    result = _integrate(Vec3(3.0, 0.0, 7.0))

    assert result.velocity.x == pytest.approx(0.0, abs=1e-9)
    assert result.velocity.y == pytest.approx(0.0, abs=1e-9)
    assert result.velocity.z == pytest.approx(7.0, abs=1e-9)
    assert result.position.x == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("turn", [-0.8, 0.8])
def test_ground_steering_rotates_velocity_to_new_heading_without_slip(turn: float) -> None:
    result = _integrate(Vec3(0.0, 0.0, 12.0), Vec3(0.0, turn, 0.0))
    forward = result.orientation.rotate(FORWARD).normalized()
    right = result.orientation.rotate(Vec3(1.0, 0.0, 0.0)).normalized()

    assert result.velocity.dot(right) == pytest.approx(0.0, abs=1e-9)
    assert result.velocity.dot(forward) == pytest.approx(12.0, abs=1e-9)
    assert result.speed == pytest.approx(12.0, abs=1e-9)
    assert result.velocity.x * turn > 0.0


def test_ground_no_slip_on_ramp_preserves_surface_tangency() -> None:
    normal = Vec3(0.0, cos(pi / 6), -sin(pi / 6))
    contact = SurfaceContact(point=ZERO, normal=normal)
    result = _integrate(Vec3(3.0, 0.0, 8.0), contact=contact)

    forward = result.orientation.rotate(FORWARD).normalized()
    right = result.orientation.rotate(Vec3(1.0, 0.0, 0.0)).normalized()
    assert result.velocity.dot(right) == pytest.approx(0.0, abs=1e-9)
    assert result.velocity.dot(normal) == pytest.approx(0.0, abs=1e-9)
    assert result.velocity.dot(forward) > 0.0


def test_ground_no_reverse_motion_and_speed_cap() -> None:
    assert _integrate(Vec3(0.0, 0.0, -10.0)).speed == pytest.approx(0.0)
    assert _integrate(Vec3(0.0, 0.0, 40.0), speed_limit=25.0).speed == pytest.approx(25.0)


def test_airborne_motion_remains_ballistic_and_does_not_project_lateral_velocity() -> None:
    result = integrate(
        _state(Vec3(3.0, 4.0, 5.0), grounded=False),
        MotionDerivatives(ZERO, ZERO),
        _params(),
        None,
        0.1,
    )
    assert result.velocity == Vec3(3.0, 4.0, 5.0)
    assert result.grounded is False
