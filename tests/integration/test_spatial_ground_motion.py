"""Actual runtime/Gym seams for explicit supported spatial contact."""
from dataclasses import replace

import numpy as np
import pytest

from src.simulation.spatial_contact_dynamics import SpatialContactPolicy
from src.simulation.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from src.simulation.vehicle import VehicleControl
from src.core.simulation import Simulation
from src.rl.action import ActionContract
from tests.sandbox.test_road_relative_suspension import ramp_translation


def spatial_factory():
    f,_,_,s=ramp_translation()
    physics=replace(f.physics_config,independent_suspension_policy=s,
        ground_motion_version='spatial_contact_v1',
        spatial_contact_policy=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.))
    return replace(f,physics_config=physics,reset_config=None,reset_track_id=None,
                   action_contract=ActionContract('steering-throttle-brake-v2'))


def test_actual_simulation_publishes_spatial_impulse_and_complete_time_evidence():
    f=spatial_factory()
    sim=Simulation(f.track,f.initial_state,f.vehicle_parameters,f.simulation_config,physics_config=f.physics_config)
    r=sim.step(VehicleControl(.4,.5,0.))
    audit=r.spatial_ground_telemetry
    assert audit.version=='spatial_contact_world_momentum_rk4_v1'
    assert r.coupled_ground_telemetry is None
    assert r.world.vehicle.velocity.x>0. and r.world.vehicle.angular_velocity.y>0.
    assert 'spatial_contact_world_momentum_rk4_v1' in r.physics_transition.solvers
    assert r.center_segments[0].start_time_s==0.
    assert r.center_segments[-1].end_time_s==pytest.approx(.1)
    assert len(audit.substeps)==len(r.center_segments)
    assert f.physics_config.document()['schema_version']==5
    assert sim.reset().spatial_ground_telemetry is None


def test_actual_gym_steers_brakes_and_keeps_direct_ppo_guard_closed():
    from src.rl.algorithms.ppo import PPOConfig,PPOModel
    from src.validation.wp7_admission import ScientificAdmissionError
    env=spatial_factory().create()
    try:
        observation,_=env.reset(seed=17)
        for _ in range(5):
            observation,_,terminated,truncated,info=env.step(np.array([.2,.5,0.],dtype=np.float32))
            assert env.observation_space.contains(observation)
            assert not terminated and not truncated
        observation,_,_,_,info=env.step(np.array([0.,0.,1.],dtype=np.float32))
        assert info['physics']['spatial_ground']['actuator_work_j']<0.
        assert env.observation_space.contains(observation)
        assert env.requires_scientific_admission
        with pytest.raises(ScientificAdmissionError):
            PPOModel.create(env,PPOConfig(n_steps=2,batch_size=2))
    finally:
        env.close()


@pytest.mark.parametrize('changes',[
    {'spatial_contact_policy':None},{'ground_motion_version':'arc_distance_v1'},
    {'mode':'wp7_lifecycle_v1'}, {'longitudinal_contact_policy':LongitudinalContactPolicy(.8,100000.)},
])
def test_spatial_registration_rejects_fallback_or_mixed_solver(changes):
    with pytest.raises(ValueError):
        replace(spatial_factory().physics_config,**changes)


def test_transition_rejects_missing_spatial_force_audit():
    from src.simulation.world import step_world
    from src.core.physics import PhysicsTransitionError
    f=spatial_factory()
    r=step_world(f.track,f.initial_state,VehicleControl(.4,.5,0.),f.vehicle_parameters,
        .1,f.simulation_config.max_surface_distance,**f.physics_config.world_options())
    with pytest.raises(PhysicsTransitionError,match='force/time audit'):
        f.physics_config.validate_transition(replace(r,spatial_ground_telemetry=None),previous=f.initial_state,
            parameters=f.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)


def test_transition_recomputes_spatial_impulse_instead_of_trusting_zero_residual():
    from src.simulation.world import step_world
    from src.simulation.geometry.vector import Vec3
    from src.core.physics import PhysicsTransitionError
    f=spatial_factory()
    r=step_world(f.track,f.initial_state,VehicleControl(.4,.5,0.),f.vehicle_parameters,
        .1,f.simulation_config.max_surface_distance,**f.physics_config.world_options())
    audit=r.spatial_ground_telemetry
    for field in ('force_impulse_ns','torque_impulse_nms'):
        first=replace(audit.substeps[0],**{field:Vec3(100.,100.,100.)})
        changed=replace(r,spatial_ground_telemetry=replace(audit,substeps=(first,*audit.substeps[1:])))
        with pytest.raises(PhysicsTransitionError,match='spatial impulse'):
            f.physics_config.validate_transition(changed,previous=f.initial_state,
                parameters=f.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)


@pytest.mark.parametrize('change',['infinite_ledger','negative_loss','infinite_summary','forged_summary'])
def test_transition_rejects_invalid_or_self_asserted_spatial_work_ledger(change):
    from src.simulation.world import step_world
    from src.core.physics import PhysicsTransitionError
    f=spatial_factory()
    r=step_world(f.track,f.initial_state,VehicleControl(.4,.5,0.),f.vehicle_parameters,
        .1,f.simulation_config.max_surface_distance,**f.physics_config.world_options())
    audit=r.spatial_ground_telemetry
    if change in ('infinite_ledger','negative_loss'):
        values=(dict(final_energy_j=float('inf'),energy_residual_j=float('inf'),energy_tolerance_j=float('inf'))
                if change=='infinite_ledger' else dict(dissipation_j=-1.))
        audit=replace(audit,substeps=(replace(audit.substeps[0],**values),*audit.substeps[1:]))
    else:
        audit=replace(audit,actuator_work_j=float('inf') if change=='infinite_summary' else 100000.)
    with pytest.raises(PhysicsTransitionError,match='spatial'):
        f.physics_config.validate_transition(replace(r,spatial_ground_telemetry=audit),previous=f.initial_state,
            parameters=f.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)
