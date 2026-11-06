from f1rl.sandbox.dynamics.air_dynamics import compute_air_motion
from f1rl.sandbox.dynamics.ground_dynamics import compute_ground_motion
from f1rl.sandbox.dynamics.vehicle_dynamics import compute_motion
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.surface import SurfaceContact
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState


def parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 12.0, 20.0, 0.1, 0.02, 50.0, 9.81)


def state(grounded: bool) -> VehicleState:
    return VehicleState(Vec3(0.0, 0.3, 0.0), Vec3(0.0, 0.0, 10.0), Quaternion.identity(), Vec3(0.1, 0.2, 0.3), grounded)


def test_contact_selects_ground_dynamics() -> None:
    control = VehicleControl(0.2, 0.5, 0.0)
    contact = SurfaceContact(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    expected = compute_ground_motion(state(True), control, parameters(), contact)
    actual = compute_motion(state(True), control, parameters(), contact)

    assert actual.linear_acceleration == expected.linear_acceleration
    assert actual.angular_velocity == expected.angular_velocity


def test_missing_contact_selects_air_dynamics() -> None:
    control = VehicleControl(1.0, 1.0, 1.0)
    expected = compute_air_motion(state(False), control, parameters())
    actual = compute_motion(state(False), control, parameters(), None)

    assert actual.linear_acceleration == expected.linear_acceleration
    assert actual.angular_velocity == expected.angular_velocity
