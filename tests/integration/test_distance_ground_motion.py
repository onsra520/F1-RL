"""Research distance/yaw coherence; historical C2 stepping stays versioned."""
from dataclasses import replace
from math import atan2, cos, sin, tan

import numpy as np
import pytest

from f1rl.learning.action import ActionContract
from tests.integration.test_wp7_lifecycle_race import lifecycle_race_factory


def test_distance_version_integrates_stop_distance_and_steering_together():
    base = lifecycle_race_factory(launch=False,finish=20.)
    speed = .5
    candidate = replace(base,
        physics_config=replace(base.physics_config,ground_motion_version='arc_distance_v1'),
        initial_state=replace(base.initial_state,vehicle=replace(base.initial_state.vehicle,
            velocity=replace(base.initial_state.vehicle.velocity,z=speed))),
        action_contract=ActionContract('steering-throttle-brake-v2'))
    env = candidate.create()
    try:
        env.reset()
        start = env._simulation.snapshot.world.vehicle
        info = env.step(np.array([1.,0.,1.],dtype=np.float32))[-1]
        end = env._simulation.snapshot.world.vehicle
        p = candidate.vehicle_parameters
        distance = speed**2/(2*p.max_braking)
        curvature = tan(p.max_steering_angle)/p.wheelbase
        yaw = curvature*distance
        forward = end.orientation.rotate(type(end.position)(0.,0.,1.))
        assert atan2(forward.x,forward.z) == pytest.approx(yaw,abs=1e-9)
        assert end.position.x-start.position.x == pytest.approx((1-cos(yaw))/curvature,abs=1e-9)
        assert end.position.z-start.position.z == pytest.approx(sin(yaw)/curvature,abs=1e-9)
        assert end.speed == pytest.approx(0.,abs=1e-9)
        assert info['physics']['transition']['final_phase'] == 'grounded'
        poses = env._simulation.snapshot.partial_support_telemetry.poses
        for pose in poses[2:]:
            assert pose.position == end.position
            assert pose.velocity.magnitude == pytest.approx(0.,abs=1e-9)
            assert pose.angular_velocity.magnitude == pytest.approx(0.,abs=1e-9)
    finally:
        env.close()


def test_acceleration_from_rest_steers_over_the_integrated_distance():
    base = lifecycle_race_factory(launch=False,finish=20.)
    base = replace(base,physics_config=replace(base.physics_config,ground_motion_version='arc_distance_v1'),
        initial_state=replace(base.initial_state,vehicle=replace(base.initial_state.vehicle,
            velocity=replace(base.initial_state.vehicle.velocity,z=0.))),
        vehicle_parameters=replace(base.vehicle_parameters,max_acceleration=4.))
    env = base.create()
    try:
        env.reset()
        info = env.step(np.array([1.,1.],dtype=np.float32))[-1]
        end = env._simulation.snapshot.world.vehicle
        curvature = tan(base.vehicle_parameters.max_steering_angle)/base.vehicle_parameters.wheelbase
        forward = end.orientation.rotate(type(end.position)(0.,0.,1.))
        assert atan2(forward.x,forward.z) == pytest.approx(curvature*.02)
        assert end.velocity.magnitude == pytest.approx(.4)
        assert end.angular_velocity.y == pytest.approx(curvature*.4)
        assert info['physics']['transition']['final_phase']=='grounded'
    finally:
        env.close()


def test_legacy_physics_document_and_default_world_options_stay_identical():
    from dataclasses import asdict
    physics = lifecycle_race_factory(launch=False).physics_config
    expected = asdict(physics)
    expected.pop('ground_motion_version')
    expected.pop('spawn_reference')
    expected.pop('longitudinal_contact_policy')
    expected['independent_suspension_policy'].pop('force_model')
    assert physics.document() == dict(schema_version=1,validation_scope='software_research_only',
        longitudinal_model='legacy_prescribed_v1',**expected)
    assert 'ground_motion_version' not in physics.world_options()
    changed = replace(physics,ground_motion_version='arc_distance_v1')
    assert changed.document()['schema_version']==2
    assert changed.sha256!=physics.sha256


def test_research_brake_contract_cannot_select_legacy_euler_steering():
    base = lifecycle_race_factory(launch=False)
    with pytest.raises(ValueError,match='braking.*distance'):
        replace(base,action_contract=ActionContract('steering-throttle-brake-v2'))


def test_current_registered_scenarios_explicitly_bind_distance_ground_motion():
    from pathlib import Path
    from f1rl.research.wp7_execution import build_policy_execution
    binding = build_policy_execution(Path(__file__).resolve().parents[2],'multitrack-seed0')
    assert {s.factory.physics_config.ground_motion_version for s in binding.scenarios.scenarios} == {'arc_distance_v1'}


@pytest.mark.parametrize('direct',[False,True])
@pytest.mark.parametrize('body_only',[False,True])
def test_every_research_braking_boundary_requires_distance_registration(direct,body_only):
    from dataclasses import fields
    from f1rl.learning.gym_environment import F1RLEnv
    base = lifecycle_race_factory(launch=False)
    kwargs = {field.name:getattr(base,field.name) for field in fields(base) if field.init}
    if body_only:
        kwargs.update(physics_config=None,race_config=None,body_geometry=base.physics_config.body_geometry,
                      initial_state=replace(base.initial_state,suspension=None,flight=None))
    kwargs['action_contract'] = ActionContract('steering-throttle-brake-v2')
    with pytest.raises(ValueError,match='braking.*distance'):
        (F1RLEnv if direct else type(base))(**kwargs)


def test_accelerating_partial_support_checks_each_actual_substep_distance():
    from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    base = lifecycle_race_factory(launch=False,finish=20.)
    physics = replace(base.physics_config,ground_motion_version='arc_distance_v1',
        automatic_takeoff_policy=replace(base.physics_config.automatic_takeoff_policy,max_substep_xz_m=.003))
    candidate = replace(base,physics_config=physics,
        initial_state=replace(base.initial_state,vehicle=replace(base.initial_state.vehicle,
            velocity=replace(base.initial_state.vehicle.velocity,z=0.))),
        vehicle_parameters=replace(base.vehicle_parameters,max_acceleration=4.))
    env = candidate.create()
    try:
        env.reset()
        before = env._simulation.snapshot
        # Whole-tick average is .02/8=.0025; the final accelerating chord is .0046875.
        with pytest.raises(SuspensionResolutionError,match='substep XZ'):
            env.step(np.array([0.,1.],dtype=np.float32))
        assert env._simulation.snapshot is before
    finally:
        env.close()


def test_distance_step_without_automatic_release_uses_same_stop_law():
    from f1rl.learning.wp7_race import WP7RaceConfig
    base = lifecycle_race_factory(launch=False,finish=20.)
    physics = replace(base.physics_config,mode='wp7_suspension_v1',ballistic_flight_policy=None,
        automatic_takeoff_policy=None,ground_motion_version='arc_distance_v1')
    candidate = replace(base,physics_config=physics,race_config=WP7RaceConfig(),
        initial_state=replace(base.initial_state,flight=None,vehicle=replace(base.initial_state.vehicle,
            velocity=replace(base.initial_state.vehicle.velocity,z=.5))),
        action_contract=ActionContract('steering-throttle-brake-v2'))
    env = candidate.create()
    try:
        env.reset()
        info = env.step(np.array([1.,0.,1.],dtype=np.float32))[-1]
        poses = env._simulation.snapshot.suspension_telemetry.poses
        assert poses[-1].velocity.magnitude==pytest.approx(0.,abs=1e-12)
        assert poses[-1].angular_velocity.magnitude==pytest.approx(0.,abs=1e-12)
        assert all(p.position==poses[-1].position for p in poses[2:])
        assert 'arc_distance_ground_v1' in info['physics']['transition']['solvers']
    finally:
        env.close()


def test_transition_audit_rejects_forged_zero_distance_yaw():
    from f1rl.runtime.physics import PhysicsTransitionError
    from f1rl.sandbox.geometry.rotation import Quaternion
    from f1rl.sandbox.geometry.vector import Vec3
    from f1rl.sandbox.vehicle import VehicleControl
    from f1rl.sandbox.world import step_world
    base = lifecycle_race_factory(launch=False,finish=20.)
    physics = replace(base.physics_config,ground_motion_version='arc_distance_v1')
    initial = base.initial_state
    result = step_world(base.track,initial,VehicleControl(0.,0.,0.),base.vehicle_parameters,.1,.75,
                        **physics.world_options())
    forged = replace(result,state=replace(result.state,vehicle=replace(result.state.vehicle,
        position=initial.vehicle.position,orientation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),.01))))
    with pytest.raises(PhysicsTransitionError,match='zero-distance'):
        physics.validate_transition(forged,previous=initial,parameters=base.vehicle_parameters,
                                    dt=.1,start_time_s=0.,end_time_s=.1)


def test_distance_registration_rejects_legacy_suspension_rate_mapping():
    physics = lifecycle_race_factory(launch=False).physics_config
    with pytest.raises(ValueError,match='angular-rate mapping'):
        replace(physics,ground_motion_version='arc_distance_v1',
                independent_suspension_policy=replace(physics.independent_suspension_policy,
                                                      angular_rate_mapping='small_angle_v1'))


def test_braking_with_rear_wheel_support_and_no_center_contact():
    from f1rl.sandbox.ballistic_flight import FlightState
    from tests.integration.test_pv5_lifecycle import lifecycle_config
    from tests.integration.test_wp7_physics_integration import world, ROAD, TrackWorld, Vec3, simulation, VehicleControl
    initial = world(speed=.5,flight=FlightState('grounded'))
    initial = replace(initial,vehicle=replace(initial.vehicle,position=Vec3(0.,.5,2.1)))
    sim = simulation(replace(lifecycle_config(),ground_motion_version='arc_distance_v1'),initial,
        track=TrackWorld(surfaces=(replace(ROAD,length=4.),)),dt=.05)
    assert sim.snapshot.query.contact is None
    out = sim.step(VehicleControl(0.,0.,1.))
    assert out.world.vehicle.position.z-2.1 == pytest.approx(.00625)
    assert out.world.vehicle.velocity.z == 0.
    assert out.partial_support_telemetry.frames[0].supported_wheels==2
    assert sim.step(VehicleControl(0.,0.,1.)).world.vehicle.position.z == out.world.vehicle.position.z


def test_missing_center_and_wheel_support_never_falls_back_to_air_motion():
    from f1rl.sandbox.ballistic_flight import FlightState
    from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
    from tests.integration.test_pv5_lifecycle import lifecycle_config
    from tests.integration.test_wp7_physics_integration import world, ROAD, TrackWorld, Vec3, simulation, VehicleControl
    initial = world(speed=.5,flight=FlightState('grounded'))
    initial = replace(initial,vehicle=replace(initial.vehicle,position=Vec3(0.,.5,4.)))
    sim = simulation(replace(lifecycle_config(),ground_motion_version='arc_distance_v1'),initial,
        track=TrackWorld(surfaces=(replace(ROAD,length=4.),)),dt=.05)
    before = sim.snapshot
    with pytest.raises(SuspensionResolutionError,match='horizontal wheel support'):
        sim.step(VehicleControl(0.,0.,1.))
    assert sim.snapshot is before


@pytest.mark.parametrize('mode',['legacy_v1','wp7_ballistic_v1','wp7_grounded_yaw_v1'])
def test_distance_version_cannot_combine_unregistered_physics_modes(mode):
    with pytest.raises(ValueError,match='ground motion version/mode'):
        replace(lifecycle_race_factory(launch=False).physics_config,mode=mode,
                ground_motion_version='arc_distance_v1')


@pytest.mark.parametrize('displacement',[(.1,0.,0.),(0.,0.,-.1)])
def test_transition_audit_rejects_sideways_or_backward_arc_chord(displacement):
    from f1rl.runtime.physics import PhysicsTransitionError
    from f1rl.sandbox.geometry.vector import Vec3
    from f1rl.sandbox.vehicle import VehicleControl
    from f1rl.sandbox.world import step_world
    base = lifecycle_race_factory(launch=False,finish=20.)
    physics = replace(base.physics_config,ground_motion_version='arc_distance_v1')
    initial = base.initial_state
    result = step_world(base.track,initial,VehicleControl(0.,0.,0.),base.vehicle_parameters,.1,.75,
                        **physics.world_options())
    position = initial.vehicle.position+Vec3(*displacement)
    forged = replace(result,state=replace(result.state,vehicle=replace(result.state.vehicle,position=position)),
        center_segments=(*result.center_segments[:-1],replace(result.center_segments[-1],end=position)))
    with pytest.raises(PhysicsTransitionError,match='forward arc'):
        physics.validate_transition(forged,previous=initial,parameters=base.vehicle_parameters,
                                    dt=.1,start_time_s=0.,end_time_s=.1)
