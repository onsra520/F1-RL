"""Synthetic lifecycle references, not measured vehicle validation."""
from dataclasses import replace
from math import pi, sqrt

import numpy as np
import pytest

from src.runtime.physics import PhysicsConfig, PhysicsTransitionError
from src.sandbox.ballistic_flight import BallisticFlightPolicy, FlightState
from src.sandbox.contact_manifold import ContactManifoldPolicy
from src.sandbox.independent_suspension_dynamics import (
    SuspensionDynamicsState, angular_velocity_for_suspension,
)
from src.sandbox.partial_wheel_release import AutomaticTakeoffPolicy
from tests.integration.test_wp7_physics_integration import (
    BODY, CONTACT, SUSPENSION, world, factory, simulation, Vec3, Quaternion, VehicleControl,
)


def lifecycle_config(**changes):
    suspension = replace(SUSPENSION, angular_rate_mapping="yaw_rx_rz_v1")
    manifold = ContactManifoldPolicy(1600., max_ground_yaw_radps=1e-8)
    config = PhysicsConfig(
        "wp7_lifecycle_v1", BODY, suspension,
        BallisticFlightPolicy(CONTACT, 6., .12, manifold=manifold),
        AutomaticTakeoffPolicy(.6),
    )
    return replace(config, **changes)


def test_exact_rate_mapping_has_world_y_component_from_roll_at_pitched_attitude():
    state = SuspensionDynamicsState(.5, pitch_x_rad=pi/6, roll_rate_radps=2.)
    omega = angular_velocity_for_suspension(Quaternion.identity(), state, 0., "yaw_rx_rz_v1")
    assert omega.x == 0.
    assert omega.y == pytest.approx(-1.)
    assert omega.z == pytest.approx(sqrt(3.))


def test_launch_flight_landing_recovery_and_driving_share_one_gym_lifecycle():
    initial = world(speed=0., vertical_speed=2., flight=FlightState("takeoff"))
    candidate = factory(lifecycle_config(), initial)
    candidate = replace(candidate, vehicle_parameters=replace(candidate.vehicle_parameters, max_acceleration=4.))
    env = candidate.create()
    try:
        env.reset()
        transitions = []
        solvers = set()
        for tick in range(1, 11):
            info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
            audit = info["physics"]["transition"]
            assert audit["end_time_s"] == pytest.approx(tick*.05)
            solvers.update(audit["solvers"])
            transitions.extend(e["kind"] for e in audit["events"])
        assert transitions == ["takeoff", "landing"]
        assert info["telemetry"]["grounded"]
        assert info["telemetry"]["vehicle_position"][1] == pytest.approx(.5, abs=1e-7)
        assert "contact_manifold_pgs_v1" in solvers
        assert "ballistic_constant_omega_v1" in solvers
        driven = env.step(np.array([0., .5], dtype=np.float32))[-1]
        assert driven["telemetry"]["vehicle_velocity"][2] > 0.
        assert driven["physics"]["transition"]["final_phase"] == "grounded"
    finally:
        env.close()


def test_lifecycle_rejects_incompatible_attitude_before_state_is_overwritten():
    initial = world(flight=FlightState("grounded"))
    initial = replace(initial, suspension=replace(initial.suspension, pitch_x_rad=.02))
    with pytest.raises(ValueError, match="attitude"):
        simulation(lifecycle_config(), initial)


def test_unowned_grounded_yaw_momentum_cannot_be_discarded():
    initial = world(speed=0., flight=FlightState("grounded"))
    initial = replace(initial, vehicle=replace(initial.vehicle, angular_velocity=Vec3(0.,1.,0.)))
    with pytest.raises(ValueError, match="rates disagree"):
        simulation(lifecycle_config(), initial)


def test_registered_kinematic_steering_yaw_is_reconstructed_from_accepted_path():
    sim = simulation(lifecycle_config(), world(flight=FlightState("grounded")))
    for _ in range(3):
        out = sim.step(VehicleControl(.1, 0., 0.))
        assert out.world.vehicle.angular_velocity.y > 0.
        assert out.physics_transition.final_phase == "grounded"


@pytest.mark.parametrize("change", ["mapping", "manifold", "yaw", "release"])
def test_lifecycle_cannot_combine_unregistered_kernels(change):
    config = lifecycle_config()
    changes = {
        "mapping": {"independent_suspension_policy": SUSPENSION},
        "manifold": {"ballistic_flight_policy": replace(config.ballistic_flight_policy, manifold=None)},
        "yaw": {"ballistic_flight_policy": replace(config.ballistic_flight_policy,
                 manifold=ContactManifoldPolicy(1600.))},
        "release": {"automatic_takeoff_policy": None},
    }
    with pytest.raises(ValueError, match="lifecycle"):
        replace(config, **changes[change])


def test_missing_landing_event_cannot_publish_a_grounded_state():
    from src.sandbox.world import step_world
    from tests.integration.test_wp7_physics_integration import PARAMETERS, ROAD, TrackWorld
    initial = world(height=.85, vertical_speed=-1., grounded=False, flight=FlightState("airborne"))
    config = lifecycle_config()
    result = step_world(TrackWorld(surfaces=(ROAD,)), initial, VehicleControl(0., 0., 0.),
                        PARAMETERS, .25, 1., **config.world_options())
    assert result.state.vehicle.grounded
    with pytest.raises(PhysicsTransitionError, match="lacks ordered"):
        config.validate_transition(replace(result, events=()), previous=initial,
            parameters=PARAMETERS, dt=.25, start_time_s=0., end_time_s=.25)


def test_partial_support_release_has_actual_solver_and_tick_local_time_in_gym():
    from src.runtime.simulation import SimulationConfig
    from tests.integration.test_wp7_physics_integration import ROAD
    candidate = factory(lifecycle_config(), world(speed=40., flight=FlightState("grounded")))
    env = replace(candidate, track=replace(candidate.track, surfaces=(replace(ROAD, length=4.),)),
                  simulation_config=SimulationConfig(.1, 1.)).create()
    try:
        env.reset()
        info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
        audit = info["physics"]["transition"]
        assert audit["solvers"] == ("partial_wheel_support_v1", "ballistic_constant_omega_v1")
        assert audit["events"][0]["time_s"] == info["physics"]["partial_support"]["takeoff_at_s"]
        assert audit["final_phase"] == "airborne"
    finally:
        env.close()


def test_coplanar_multi_patch_landing_is_auditable_through_gym():
    from src.runtime.simulation import SimulationConfig
    from tests.integration.test_wp7_physics_integration import ROAD
    config = lifecycle_config()
    config = replace(config, ballistic_flight_policy=replace(config.ballistic_flight_policy,
        manifold=replace(config.ballistic_flight_policy.manifold, coplanar_flat_contacts=True)))
    candidate = factory(config, world(height=.85, vertical_speed=-1., grounded=False,
                                      flight=FlightState("airborne")))
    patches = (replace(ROAD, id="left", origin=Vec3(-5., 0., 0.), width=10.),
               replace(ROAD, id="right", origin=Vec3(5., 0., 0.), width=10.))
    env = replace(candidate, track=replace(candidate.track, surfaces=patches),
                  simulation_config=SimulationConfig(.25, 1.)).create()
    try:
        env.reset()
        info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
        contacts = info["physics"]["flight"]["manifold_audit"]["contacts"]
        assert len(contacts) == 4
        assert {c["patch_id"] for c in contacts} == {"left", "right"}
        assert info["telemetry"]["grounded"]
    finally:
        env.close()
