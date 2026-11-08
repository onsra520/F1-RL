"""Rotating compliant flight uses world-H dynamics and ballistic/SLERP CCD."""
from dataclasses import replace
import pytest
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.vehicle import VehicleControl
from f1rl.sandbox.spatial_contact_dynamics import integrate_spatial_lifecycle
from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
from tests.sandbox.test_spatial_atlas_lifecycle import sloped_landing


def rotating_air():
    f,_,s,policy=sloped_landing()
    v=replace(f.initial_state.vehicle,position=Vec3(0.,5.,4.),velocity=Vec3(0.,0.,2.),
        angular_velocity=Vec3(.1,.2,.15),grounded=False)
    return f,v,s,policy


def momentum(v,s,policy):
    q=v.orientation
    w=Quaternion(q.w,-q.x,-q.y,-q.z).rotate(v.angular_velocity)
    return q.rotate(Vec3(s.pitch_inertia_kg_m2*w.x,policy.yaw_inertia_kg_m2*w.y,s.roll_inertia_kg_m2*w.z))


def test_variable_omega_free_flight_preserves_world_momentum_and_passive_energy():
    f,v,s,policy=rotating_air()
    args=(v,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,VehicleControl(.8,.7,.6),policy)
    with pytest.raises(SuspensionResolutionError,match='fixed airborne attitude'):
        integrate_spatial_lifecycle(*args,dt=.1,phase_domain='atlas_gradient_v2')
    r=integrate_spatial_lifecycle(*args,dt=.1,phase_domain='atlas_rotating_v3')
    end=r.suspension.vehicle
    assert not end.grounded and end.orientation!=v.orientation
    assert (end.angular_velocity-v.angular_velocity).magnitude>1e-5
    assert (momentum(end,s,policy)-momentum(v,s,policy)).magnitude<1e-9
    assert end.velocity.y==pytest.approx(-f.vehicle_parameters.gravity*.1,abs=1e-12)
    assert end.position.y==pytest.approx(5.-.5*f.vehicle_parameters.gravity*.1**2,abs=1e-12)
    assert r.audit.actuator_work_j==r.audit.dissipation_j==0.
    assert abs(end.angular_velocity.dot(momentum(end,s,policy))-v.angular_velocity.dot(momentum(v,s,policy)))<1e-7
    assert r.audit.version=='spatial_rotating_compliant_lifecycle_rk4_v3'
    assert all(abs(a.energy_residual_j)<=a.energy_tolerance_j for a in r.audit.substeps)


def test_ballistic_slerp_body_bound_blocks_hidden_overhead_contact():
    from f1rl.sandbox.body_collision import sweep_body_against_obstacle,sweep_ballistic_body_against_obstacle
    from f1rl.sandbox.track.volume import OrientedBoxVolume
    f,v,s,policy=rotating_air()
    v=replace(v,velocity=Vec3(0.,.4905,10.),angular_velocity=Vec3(0.,.2,0.))
    r=integrate_spatial_lifecycle(v,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,
        VehicleControl(0.,0.,0.),policy,dt=.1,phase_domain='atlas_rotating_v3')
    end=r.suspension.vehicle
    obstacle=OrientedBoxVolume(Vec3(0.,5.9112625,4.5),Vec3(.05,.0005,.05),Quaternion.identity())
    body=f.physics_config.body_geometry
    assert not sweep_body_against_obstacle(v,end,body,obstacle).blocked
    assert sweep_ballistic_body_against_obstacle(v,end,body,obstacle,duration_s=.1,gravity_mps2=9.81).blocked


def test_public_rotating_ballistic_world_blocks_hidden_overhead_obstacle():
    from f1rl.sandbox.spatial_contact_dynamics import _suspension_state
    from f1rl.sandbox.world import step_world
    from f1rl.sandbox.entities.obstacle import Obstacle
    from f1rl.sandbox.track.volume import OrientedBoxVolume
    from f1rl.sandbox.ballistic_flight import FlightPhysicsError
    from tests.integration.test_spatial_compliant_gym import airborne_step_fixture
    f=airborne_step_fixture()
    v=replace(f.initial_state.vehicle,angular_velocity=Vec3(0.,.2,0.))
    state=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v))
    config=replace(f.physics_config,ground_motion_version='spatial_rotating_compliant_lifecycle_v3')
    obstacle=Obstacle('overhead',OrientedBoxVolume(Vec3(0.,5.9112625,4.5),Vec3(.05,.0005,.05),Quaternion.identity()))
    with pytest.raises(FlightPhysicsError,match='ballistic.*obstacle'):
        step_world(replace(f.track,obstacles=(obstacle,)),state,VehicleControl(0.,0.,0.),f.vehicle_parameters,
            .1,1.,**config.world_options())


def test_rotating_takeoff_airborne_gates_landing_recovery_and_finish_share_one_gym_episode():
    import numpy as np
    from f1rl.sandbox.spatial_contact_dynamics import _suspension_state
    from f1rl.learning.wp7_race import WP7RaceConfig
    from tests.integration.test_spatial_compliant_gym import moving_race_factory
    f=moving_race_factory()
    v=replace(f.initial_state.vehicle,angular_velocity=Vec3(0.,.2,0.))
    f=replace(f,physics_config=replace(f.physics_config,ground_motion_version='spatial_rotating_compliant_lifecycle_v3'),
        initial_state=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v)),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_owner_v2',hard_violation='terminate_valid_failure'))
    env=f.create()
    phases=[]
    airborne_rotation=False
    try:
        env.reset()
        for _ in range(70):
            observation,_,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
            assert env.observation_space.contains(observation) and not truncated
            phases.extend(e['kind'] for e in info['physics']['transition']['events'])
            if info['physics']['transition']['final_phase']=='airborne':
                airborne_rotation=True
            assert 'spatial_rotating_compliant_lifecycle_rk4_v3' in info['physics']['transition']['solvers']
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='successful_finish'
        assert phases==['takeoff','landing'] and airborne_rotation
        assert info['physics']['transition']['final_phase']=='grounded'
        assert env.requires_scientific_admission
    finally:
        env.close()


def test_rotating_touchdown_on_actual_train_ramp_uses_gradient_phase_proof():
    f,v,s,policy=sloped_landing()
    v=replace(v,angular_velocity=Vec3(0.,.2,0.))
    r=integrate_spatial_lifecycle(v,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,
        VehicleControl(0.,0.,0.),policy,dt=.03,phase_domain='atlas_rotating_v3')
    assert r.suspension.vehicle.grounded
    assert [b.kind for b in r.audit.phase_brackets]==['landing']
    assert all(abs(a.energy_residual_j)<=a.energy_tolerance_j for a in r.audit.substeps)


def test_airborne_ground_actions_and_modifiers_do_not_change_torque_free_state():
    from f1rl.sandbox.dynamics.modifiers import MotionModifiers
    f,v,s,policy=rotating_air()
    args=(v,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters)
    zero=integrate_spatial_lifecycle(*args,VehicleControl(0.,0.,0.),policy,dt=.1,phase_domain='atlas_rotating_v3')
    acted=integrate_spatial_lifecycle(*args,VehicleControl(.8,.7,.6),policy,dt=.1,phase_domain='atlas_rotating_v3',
        modifiers=MotionModifiers(longitudinal_acceleration_bonus=10.,rolling_drag_multiplier=3.))
    assert acted==zero


def test_rotation_leaf_refines_before_a_possible_aliased_full_turn():
    from math import acos,pi
    f,v,s,policy=rotating_air()
    s=replace(s,integration_substeps=1)
    v=replace(v,angular_velocity=Vec3(0.,8.,0.))
    parameters=replace(f.vehicle_parameters,gravity=.01)
    args=(v,s,f.physics_config.body_geometry,f.track.surfaces,parameters,VehicleControl(0.,0.,0.))
    r=integrate_spatial_lifecycle(*args,policy,dt=.4,phase_domain='atlas_rotating_v3')
    assert len(r.audit.substeps)>=2
    for first,last in zip(r.suspension.poses,r.suspension.poses[1:]):
        q,p=first.orientation,last.orientation
        dot=abs(q.w*p.w+q.x*p.x+q.y*p.y+q.z*p.z)
        assert 2.*acos(min(1.,dot))<pi
    with pytest.raises(SuspensionResolutionError,match='full turn'):
        integrate_spatial_lifecycle(*args,replace(policy,traction=replace(policy.traction,max_refinements=0)),
            dt=.4,phase_domain='atlas_rotating_v3')


def test_rotating_air_runtime_rejects_claimed_actuator_work():
    from f1rl.runtime.physics import PhysicsTransitionError
    from f1rl.sandbox.spatial_contact_dynamics import _suspension_state
    from f1rl.sandbox.world import step_world
    from tests.integration.test_spatial_compliant_gym import airborne_step_fixture
    f=airborne_step_fixture()
    v=replace(f.initial_state.vehicle,angular_velocity=Vec3(.1,.2,.15))
    state=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v))
    config=replace(f.physics_config,ground_motion_version='spatial_rotating_compliant_lifecycle_v3')
    assert config.document()['schema_version']==8
    r=step_world(f.track,state,VehicleControl(0.,0.,0.),f.vehicle_parameters,.1,1.,**config.world_options())
    config.validate_transition(r,previous=state,parameters=f.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)
    audit=r.spatial_ground_telemetry
    forged=replace(r,spatial_ground_telemetry=replace(audit,substeps=(replace(audit.substeps[0],actuator_work_j=1.),*audit.substeps[1:])))
    with pytest.raises(PhysicsTransitionError,match='ballistic law'):
        config.validate_transition(forged,previous=state,parameters=f.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)


def test_seeded_world_h_integrator_matches_independent_body_euler_free_rotation():
    from random import Random
    from f1rl.sandbox.torque_free_rotation import (AngularRigidBodyState,TorqueFreeRotationPolicy,advance_torque_free_rotation)
    rng=Random(1027)
    f,base,s,policy=rotating_air()
    angular=TorqueFreeRotationPolicy(s.pitch_inertia_kg_m2,policy.yaw_inertia_kg_m2,s.roll_inertia_kg_m2,
        integration_steps=64,max_angular_speed_radps=policy.max_angular_speed_radps)
    for _ in range(8):
        axis=Vec3(*(rng.uniform(-1.,1.) for _ in range(3))).normalized()
        v=replace(base,orientation=Quaternion.from_axis_angle(axis,.03),
            angular_velocity=Vec3(*(rng.uniform(-.2,.2) for _ in range(3))))
        reference,_=advance_torque_free_rotation(AngularRigidBodyState(v.orientation,v.angular_velocity),angular,.1)
        r=integrate_spatial_lifecycle(v,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,
            VehicleControl(0.,0.,0.),policy,dt=.1,phase_domain='atlas_rotating_v3')
        end=r.suspension.vehicle
        assert (end.angular_velocity-reference.world_angular_velocity_radps).magnitude<1e-9
        q,p=end.orientation,reference.orientation
        assert abs(q.w*p.w+q.x*p.x+q.y*p.y+q.z*p.z)>1.-1e-12
