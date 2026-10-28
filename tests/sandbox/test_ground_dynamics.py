from math import cos, pi, sin

import pytest

from src.simulation.dynamics.ground_dynamics import compute_ground_motion
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.surface import SurfaceContact
from src.simulation.vehicle import VehicleControl, VehicleParameters, VehicleState


def parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 12.0, 20.0, 0.1, 0.02, 50.0, 9.81)


def state(velocity: Vec3) -> VehicleState:
    return VehicleState(Vec3(0.0, 0.3, 0.0), velocity, Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True)


def flat_contact() -> SurfaceContact:
    return SurfaceContact(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))


def test_flat_ground_throttle_accelerates_forward() -> None:
    motion = compute_ground_motion(state(Vec3(0.0, 0.0, 10.0)), VehicleControl(0.0, 1.0, 0.0), parameters(), flat_contact())
    assert motion.linear_acceleration == Vec3(0.0, 0.0, 11.0)


def test_braking_and_drag_decelerate_without_accelerating_backward() -> None:
    moving = state(Vec3(0.0, 0.0, 10.0))
    motion = compute_ground_motion(moving, VehicleControl(0.0, 0.0, 1.0), parameters(), flat_contact())
    assert motion.linear_acceleration.z < 0.0


def test_steering_creates_yaw_about_surface_normal() -> None:
    motion = compute_ground_motion(state(Vec3(0.0, 0.0, 10.0)), VehicleControl(1.0, 0.0, 0.0), parameters(), flat_contact())
    assert motion.angular_velocity.y > 0.0
    assert motion.angular_velocity.x == pytest.approx(0.0)
    assert motion.angular_velocity.z == pytest.approx(0.0)


def test_ground_speed_cap_suppresses_positive_drive_at_limit() -> None:
    motion = compute_ground_motion(state(Vec3(0.0, 0.0, 50.0)), VehicleControl(0.0, 1.0, 0.0), parameters(), flat_contact())
    assert motion.linear_acceleration.z <= 0.0


def test_slope_projects_forward_motion_up_ramp_and_gravity_down_ramp() -> None:
    angle = pi / 6.0
    normal = Vec3(0.0, cos(angle), -sin(angle))
    contact = SurfaceContact(Vec3(0.0, 0.0, 0.0), normal)
    motion = compute_ground_motion(state(Vec3(0.0, 0.0, 0.0)), VehicleControl(0.0, 1.0, 0.0), parameters(), contact)
    assert motion.linear_acceleration.y > 0.0

    gravity_only = compute_ground_motion(state(Vec3(0.0, 0.0, 0.0)), VehicleControl(0.0, 0.0, 0.0), parameters(), contact)
    ramp_forward = Vec3(0.0, sin(angle), cos(angle))
    assert gravity_only.linear_acceleration.dot(ramp_forward) < 0.0


def test_ground_motion_is_deterministic() -> None:
    args = (state(Vec3(0.0, 0.0, 12.0)), VehicleControl(0.25, 0.4, 0.1), parameters(), flat_contact())
    assert compute_ground_motion(*args) == compute_ground_motion(*args)


def test_ground_motion_handles_forward_parallel_to_surface_normal() -> None:
    vertical = Quaternion.from_axis_angle(Vec3(1.0, 0.0, 0.0), -pi / 2.0)
    unusual_state = VehicleState(
        Vec3(0.0, 0.3, 0.0),
        Vec3(0.0, 0.0, 0.0),
        vertical,
        Vec3(0.0, 0.0, 0.0),
        True,
    )

    motion = compute_ground_motion(unusual_state, VehicleControl(0.0, 1.0, 0.0), parameters(), flat_contact())

    assert motion.linear_acceleration.magnitude > 0.0
