"""Road-driven reduced suspension forces, including the actual TRAIN ramp."""
from dataclasses import replace
from math import atan2, cos, tan
from pathlib import Path

import pytest

from src.validation.wp7_protocol import load_registration, prepare_scenarios
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.independent_suspension_dynamics import (
    SuspensionDynamicsState, SuspensionResolutionError, angular_velocity_for_suspension,
    integrate_independent_suspension, suspension_orientation,
)


def ramp_translation():
    root = Path(__file__).resolve().parents[2]
    factory = next(s.factory for s in prepare_scenarios(root, load_registration(root), seed_index=0).scenarios
                   if s.track_id == 'ramp-v1')
    ramp = next(p for p in factory.track.surfaces if p.id == 'road-incline')
    angle = atan2(ramp.forward.y, ramp.forward.z)
    height = ramp.origin.y + (80. - ramp.origin.z) * tan(angle) + .5 / cos(angle)
    dynamic = SuspensionDynamicsState(height, 3. * tan(angle), -angle)
    vehicle = replace(factory.initial_state.vehicle, position=Vec3(0., height, 80.),
                      velocity=Vec3(0., dynamic.heave_velocity_mps, 3.),
                      orientation=suspension_orientation(Quaternion.identity(), dynamic))
    policy = replace(factory.physics_config.independent_suspension_policy,
                     max_abs_tilt_rad=.2, force_model='road_relative_vertical_v2')
    return factory, vehicle, dynamic, policy


def test_actual_ramp_constant_translation_does_not_create_damping_or_pitch():
    factory, vehicle, dynamic, policy = ramp_translation()
    dt = .04
    result = integrate_independent_suspension(
        vehicle, replace(vehicle, position=vehicle.position + vehicle.velocity * dt),
        dynamic, policy, factory.physics_config.body_geometry, factory.track.surfaces,
        dt=dt, gravity_mps2=factory.vehicle_parameters.gravity,
    )
    for frame in result.force_samples:
        for wheel in frame:
            assert wheel.vertical_wheel_velocity_mps == pytest.approx(0., abs=1e-10)
            assert wheel.upward_force_n == pytest.approx(policy.mass_kg * 9.81 / 4., abs=1e-7)
    assert (result.vehicle.velocity - vehicle.velocity).magnitude < 1e-10
    assert result.dynamics.pitch_x_rad == pytest.approx(dynamic.pitch_x_rad, abs=1e-12)
    assert result.dynamics.pitch_rate_radps == pytest.approx(0., abs=1e-12)
    assert result.vehicle.position.y == pytest.approx(vehicle.position.y + vehicle.velocity.y * dt, abs=1e-12)


def test_actual_ramp_simulation_preserves_steady_road_driven_state():
    from math import sin
    from src.core.simulation import Simulation
    from src.simulation.vehicle import VehicleControl

    factory, vehicle, dynamic, policy = ramp_translation()
    physics = replace(factory.physics_config, independent_suspension_policy=policy,
                      ground_motion_version='legacy_euler_v1')
    initial = replace(factory.initial_state, vehicle=vehicle, suspension=dynamic)
    simulation = Simulation(factory.track, initial, factory.vehicle_parameters,
                            factory.simulation_config, physics_config=physics)
    parameters = factory.vehicle_parameters
    throttle = (parameters.rolling_drag * vehicle.speed + parameters.gravity * sin(-dynamic.pitch_x_rad)) / parameters.max_acceleration
    for _ in range(10):
        snapshot = simulation.step(VehicleControl(0., throttle, 0.))
        assert (snapshot.world.vehicle.velocity - vehicle.velocity).magnitude < 1e-8
        assert abs(snapshot.world.suspension.pitch_rate_radps) < 1e-8
        assert snapshot.physics_transition is not None
    assert snapshot.world.vehicle.position.z > 82.


def test_finite_angle_damping_and_generalized_torques_match_geometric_gap_derivatives():
    factory, vehicle, _, policy = ramp_translation()
    road = next(p for p in factory.track.surfaces if p.id == 'road-incline')
    policy = replace(policy, integration_substeps=1)
    # Independent central differences of the physical wheel/road gap, rather
    # than the implementation's cross-product Jacobian.
    dynamic = SuspensionDynamicsState(vehicle.position.y, .3, -.12, .08, .015, -.04)
    yaw = Quaternion.from_axis_angle(Vec3(0., 1., 0.), .2)
    vehicle = replace(vehicle, velocity=Vec3(.1, .3, 2.),
        orientation=suspension_orientation(yaw, dynamic),
        angular_velocity=angular_velocity_for_suspension(yaw, dynamic, .06, 'yaw_rx_rz_v1'))

    def gap(local, t=0., pitch_delta=0., roll_delta=0.):
        d = replace(dynamic, pitch_x_rad=dynamic.pitch_x_rad + t*dynamic.pitch_rate_radps + pitch_delta,
                    roll_z_rad=dynamic.roll_z_rad + t*dynamic.roll_rate_radps + roll_delta)
        q = suspension_orientation(Quaternion.from_axis_angle(Vec3(0., 1., 0.), .2 + .06*t), d)
        p = vehicle.position + vehicle.velocity*t + q.rotate(local)
        return (p-road.origin).dot(road.normal)/road.normal.y

    epsilon, dt = 1e-6, 1e-6
    expected_rates = [(gap(local, epsilon)-gap(local, -epsilon))/(2*epsilon) for local in policy.contact.wheels.positions]
    result = integrate_independent_suspension(vehicle,
        replace(vehicle, position=vehicle.position+vehicle.velocity*dt), dynamic, policy,
        factory.physics_config.body_geometry, factory.track.surfaces, dt=dt, gravity_mps2=9.81)
    pitch_torque = -policy.angular_damping_nmsprad*dynamic.pitch_rate_radps
    roll_torque = -policy.angular_damping_nmsprad*dynamic.roll_rate_radps
    for local, expected, sample in zip(policy.contact.wheels.positions, expected_rates, result.force_samples[0], strict=True):
        assert sample.vertical_wheel_velocity_mps == pytest.approx(expected, abs=1e-8)
        expected_load = max(0., policy.mass_kg*9.81/4. - policy.spring_stiffness_npm*gap(local) - policy.damper_nspm*expected)
        assert sample.upward_force_n == pytest.approx(expected_load, abs=2e-5)
        pitch_torque += sample.upward_force_n*(gap(local, pitch_delta=epsilon)-gap(local, pitch_delta=-epsilon))/(2*epsilon)
        roll_torque += sample.upward_force_n*(gap(local, roll_delta=epsilon)-gap(local, roll_delta=-epsilon))/(2*epsilon)
    assert result.dynamics.pitch_rate_radps == pytest.approx(dynamic.pitch_rate_radps+dt*pitch_torque/policy.pitch_inertia_kg_m2, abs=1e-12)
    assert result.dynamics.roll_rate_radps == pytest.approx(dynamic.roll_rate_radps+dt*roll_torque/policy.roll_inertia_kg_m2, abs=1e-12)


def test_legacy_physics_document_pin_and_new_model_identity():
    from tests.integration.test_wp7_lifecycle_race import lifecycle_race_factory
    factory, _, _, policy = ramp_translation()
    assert lifecycle_race_factory(launch=False).physics_config.sha256 == '758c5c42d696e0ea02f23412668ef85dd7614c60fa5caeb39c72d010eaa4ca4f'
    changed = replace(factory.physics_config, independent_suspension_policy=policy)
    assert changed.document()['schema_version'] == 3
    assert changed.document()['independent_suspension_policy']['force_model'] == 'road_relative_vertical_v2'
    assert changed.sha256 != factory.physics_config.sha256


def test_new_force_model_cannot_silently_enter_legacy_partial_support():
    from src.simulation.partial_wheel_release import AutomaticTakeoffPolicy, integrate_partial_support
    factory, vehicle, dynamic, policy = ramp_translation()
    with pytest.raises(SuspensionResolutionError, match='force model'):
        integrate_partial_support(vehicle, vehicle, dynamic, policy, factory.physics_config.body_geometry,
                                  factory.track.surfaces, AutomaticTakeoffPolicy(.5, max_patch_tilt_rad=.2),
                                  dt=.01, gravity_mps2=9.81)


def test_new_force_model_cannot_silently_enter_registered_flight():
    from tests.integration.test_wp7_lifecycle_race import lifecycle_race_factory
    physics = lifecycle_race_factory(launch=False).physics_config
    policy = replace(physics.independent_suspension_policy, force_model='road_relative_vertical_v2')
    with pytest.raises(ValueError, match='force model'):
        replace(physics, independent_suspension_policy=policy)


@pytest.mark.parametrize('change', [{'force_model': 'unknown'}, {'angular_rate_mapping': 'small_angle_v1'}])
def test_new_force_contract_requires_known_model_and_exact_rate_mapping(change):
    _, _, _, policy = ramp_translation()
    with pytest.raises(ValueError, match='force model'):
        replace(policy, **change)


@pytest.mark.parametrize('change', ['heave', 'angular'])
def test_gap_rate_cannot_use_inconsistent_stored_velocities(change):
    factory, vehicle, dynamic, policy = ramp_translation()
    vehicle = replace(vehicle, **({'velocity': Vec3(0., 0., 3.)} if change == 'heave'
                                 else {'angular_velocity': Vec3(.1, 0., 0.)}))
    with pytest.raises(SuspensionResolutionError, match='consistent chassis/suspension rates'):
        integrate_independent_suspension(vehicle, vehicle, dynamic, policy,
            factory.physics_config.body_geometry, factory.track.surfaces, dt=.01, gravity_mps2=9.81)


def test_reduced_road_profile_does_not_silently_enable_steering_dynamics():
    from src.core.simulation import Simulation
    from src.simulation.vehicle import VehicleControl
    factory, vehicle, dynamic, policy = ramp_translation()
    physics = replace(factory.physics_config, independent_suspension_policy=policy,
                      ground_motion_version='legacy_euler_v1')
    simulation = Simulation(factory.track, replace(factory.initial_state, vehicle=vehicle, suspension=dynamic),
                            factory.vehicle_parameters, factory.simulation_config, physics_config=physics)
    prior = simulation.snapshot
    with pytest.raises(ValueError, match='zero steering'):
        simulation.step(VehicleControl(.1, .2, 0.))
    assert simulation.snapshot is prior


@pytest.mark.parametrize('entrypoint', ['ballistic', 'manifold', 'world'])
def test_direct_lifecycle_entrypoints_reject_the_unimplemented_force_regime(entrypoint):
    from tests.integration.test_wp7_lifecycle_race import lifecycle_race_factory
    from src.simulation.ballistic_flight import advance_ballistic_flight, FlightPhysicsError
    from src.simulation.manifold_flight import advance_manifold_flight
    from src.simulation.multi_surface_support import _atlas
    from src.simulation.world import step_world
    from src.simulation.vehicle import VehicleControl
    factory = lifecycle_race_factory(launch=False)
    physics = factory.physics_config
    policy = replace(physics.independent_suspension_policy, force_model='road_relative_vertical_v2')
    airborne = replace(factory.initial_state.vehicle, grounded=False)
    with pytest.raises((ValueError, FlightPhysicsError), match='force model'):
        if entrypoint == 'ballistic':
            advance_ballistic_flight(airborne, physics.ballistic_flight_policy, physics.body_geometry,
                factory.track.surfaces, (), dt=.1, gravity_mps2=9.81, landing_dynamics=policy)
        elif entrypoint == 'manifold':
            advance_manifold_flight(airborne, physics.ballistic_flight_policy, physics.body_geometry,
                _atlas(factory.track.surfaces, policy.contact.seam_height_tolerance_m), (), .1, 9.81, policy)
        else:
            step_world(factory.track, factory.initial_state, VehicleControl(0., 0., 0.),
                factory.vehicle_parameters, .1, .5, body_geometry=physics.body_geometry,
                independent_suspension_policy=policy, ballistic_flight_policy=physics.ballistic_flight_policy)
