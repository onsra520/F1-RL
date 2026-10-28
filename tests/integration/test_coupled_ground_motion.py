"""Opt-in coupled dynamics through the actual runtime and Gym boundaries."""
from dataclasses import replace

import pytest
import numpy as np

from src.core.simulation import Simulation
from src.simulation.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from src.simulation.vehicle import VehicleControl
from src.rl.action import ActionContract
from tests.sandbox.test_road_relative_suspension import ramp_translation


def coupled_factory():
    factory, _, _, suspension = ramp_translation()
    physics = replace(factory.physics_config, independent_suspension_policy=suspension,
        ground_motion_version='coupled_longitudinal_v1',
        longitudinal_contact_policy=LongitudinalContactPolicy(.8,100000.))
    return replace(factory,physics_config=physics,reset_config=None,reset_track_id=None)


def test_simulation_publishes_coupled_force_audit_and_actual_substep_times():
    factory = coupled_factory()
    simulation = Simulation(factory.track,factory.initial_state,factory.vehicle_parameters,
                            factory.simulation_config,physics_config=factory.physics_config)
    snapshot = simulation.step(VehicleControl(0.,.5,0.))
    audit = snapshot.coupled_ground_telemetry
    assert audit.version == 'coupled_longitudinal_rk4_work_v1'
    assert audit.substeps
    assert snapshot.world.vehicle.velocity.z > 0.
    assert snapshot.world.vehicle.angular_velocity.x < 0.
    assert 'coupled_longitudinal_rk4_work_v1' in snapshot.physics_transition.solvers
    assert snapshot.center_segments[0].start_time_s == 0.
    assert snapshot.center_segments[-1].end_time_s == pytest.approx(factory.simulation_config.fixed_dt)
    assert len(snapshot.center_segments) == len(audit.substeps)
    for segment,leaf in zip(snapshot.center_segments,audit.substeps,strict=True):
        assert segment.start_time_s == leaf.start_time_s
        assert segment.end_time_s == leaf.end_time_s
        assert abs(leaf.energy_residual_j)<=leaf.energy_tolerance_j
    document = factory.physics_config.document()
    assert document['schema_version'] == 4
    assert document['longitudinal_model'] == 'coupled_contact_force_rk4_v1'
    assert document['longitudinal_contact_policy']['traction_mu'] == .8
    assert simulation.reset().coupled_ground_telemetry is None


def test_gym_three_action_contact_dynamics_brakes_and_keeps_admission_closed():
    factory = replace(coupled_factory(),action_contract=ActionContract('steering-throttle-brake-v2'))
    env = factory.create()
    try:
        observation,info = env.reset(seed=17)
        assert env.observation_space.contains(observation)
        assert env.requires_scientific_admission
        assert info['physics']['sha256'] == factory.physics_config.sha256
        for _ in range(4):
            observation,_,terminated,truncated,info = env.step(np.array([0.,.5,0.],dtype=np.float32))
            assert not terminated and not truncated
            assert env.observation_space.contains(observation)
        before = info['telemetry']['vehicle_velocity'][2]
        observation,_,_,_,info = env.step(np.array([0.,0.,1.],dtype=np.float32))
        assert info['telemetry']['vehicle_velocity'][2] < before
        assert info['physics']['coupled_ground']['actuator_work_j'] < 0.
        assert env.observation_space.contains(observation)
        with pytest.raises(ValueError,match='zero steering'):
            env.step(np.array([.1,0.,0.],dtype=np.float32))
        with pytest.raises(RuntimeError,match='reset'):
            env.step(np.zeros(3,dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize('changes', [
    {'longitudinal_contact_policy':None}, {'ground_motion_version':'arc_distance_v1'},
    {'mode':'wp7_lifecycle_v1'}, {'mode':'legacy_v1'},
])
def test_coupled_configuration_cannot_silently_fallback_or_enable_flight(changes):
    with pytest.raises(ValueError):
        replace(coupled_factory().physics_config,**changes)


def test_transition_rejects_missing_force_audit_or_shifted_time_evidence():
    from src.core.physics import PhysicsTransitionError
    from src.simulation.world import step_world
    factory = coupled_factory()
    result = step_world(factory.track,factory.initial_state,VehicleControl(0.,.5,0.),factory.vehicle_parameters,
        factory.simulation_config.fixed_dt,factory.simulation_config.max_surface_distance,**factory.physics_config.world_options())
    for invalid in (replace(result,coupled_ground_telemetry=None),
                    replace(result,center_segments=result.center_segments[1:])):
        with pytest.raises(PhysicsTransitionError,match='force/time audit'):
            factory.physics_config.validate_transition(invalid,previous=factory.initial_state,
                parameters=factory.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)


def test_coupled_proposal_cannot_allocate_trainable_ppo():
    from src.rl.algorithms.ppo import PPOConfig, PPOModel
    from src.validation.wp7_admission import ScientificAdmissionError
    env = coupled_factory().create()
    try:
        with pytest.raises(ScientificAdmissionError):
            PPOModel.create(env,PPOConfig(n_steps=2,batch_size=2))
    finally:
        env.close()


def test_coupled_transition_also_rejects_infinite_work_ledger():
    from src.core.physics import PhysicsTransitionError
    from src.simulation.world import step_world
    f=coupled_factory()
    r=step_world(f.track,f.initial_state,VehicleControl(0.,.5,0.),f.vehicle_parameters,
        .1,f.simulation_config.max_surface_distance,**f.physics_config.world_options())
    audit=r.coupled_ground_telemetry
    leaf=replace(audit.substeps[0],final_energy_j=float('inf'),energy_residual_j=float('inf'),energy_tolerance_j=float('inf'))
    changed=replace(r,coupled_ground_telemetry=replace(audit,substeps=(leaf,*audit.substeps[1:])))
    with pytest.raises(PhysicsTransitionError,match='force/time audit'):
        f.physics_config.validate_transition(changed,previous=f.initial_state,
            parameters=f.vehicle_parameters,dt=.1,start_time_s=0.,end_time_s=.1)
