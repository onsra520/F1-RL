from math import cos, pi, sin

from src.simulation.dynamics.integrator import integrate
from src.simulation.dynamics.vehicle_dynamics import compute_motion
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.surface import SurfaceContact
from src.simulation.vehicle import VehicleControl, VehicleParameters, VehicleState


def parameters() -> VehicleParameters:
    return VehicleParameters(
        wheelbase=2.5,
        ride_height=0.3,
        max_steering_angle=0.6,
        max_acceleration=12.0,
        max_braking=20.0,
        rolling_drag=0.0,
        air_drag=0.02,
        max_ground_speed=50.0,
        gravity=9.81,
    )


def ramp_contact(angle: float) -> SurfaceContact:
    return SurfaceContact(
        point=Vec3(0.0, 0.0, 0.0),
        normal=Vec3(0.0, cos(angle), -sin(angle)),
    )


def simulate_jump() -> tuple[VehicleState, VehicleState, VehicleState]:
    params = parameters()
    control = VehicleControl(0.0, 0.0, 0.0)
    ramp = ramp_contact(pi / 6.0)
    state = VehicleState(
        position=ramp.normal * params.ride_height,
        velocity=Vec3(0.0, 0.0, 12.0),
        orientation=Quaternion.identity(),
        angular_velocity=Vec3(0.0, 0.0, 0.0),
        grounded=True,
    )

    ramp_motion = compute_motion(state, control, params, ramp)
    launched = integrate(state, ramp_motion, params, ramp, 0.1)

    airborne_motion = compute_motion(launched, control, params, None)
    airborne = integrate(launched, airborne_motion, params, None, 0.1)

    falling = airborne
    for _ in range(20):
        motion = compute_motion(falling, control, params, None)
        falling = integrate(falling, motion, params, None, 0.1)
        if falling.velocity.y < 0.0:
            break

    flat = SurfaceContact(Vec3(0.0, 0.0, falling.position.z), Vec3(0.0, 1.0, 0.0))
    tangential_before = falling.velocity.reject_from(flat.normal)
    landing_motion = compute_motion(falling, control, params, flat)
    landed = integrate(falling, landing_motion, params, flat, 0.05)
    assert landed.velocity == tangential_before
    return launched, airborne, landed


def test_ramp_airtime_and_landing_transition() -> None:
    launched, airborne, landed = simulate_jump()

    assert launched.velocity.y > 0.0
    assert airborne.grounded is False
    assert airborne.velocity.y < launched.velocity.y
    assert landed.grounded is True
    assert landed.velocity.y == 0.0


def test_repeated_simulation_is_exactly_deterministic() -> None:
    assert simulate_jump() == simulate_jump()
