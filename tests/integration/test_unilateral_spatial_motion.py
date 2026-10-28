"""Opt-in unilateral law through actual PhysicsConfig, Simulation and Gym."""
from dataclasses import replace
import numpy as np

from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from tests.integration.test_spatial_ground_motion import spatial_factory


def test_partial_load_contact_reaches_gym_with_explicit_force_identity_and_work_audit():
    f=spatial_factory()
    suspension=replace(f.physics_config.independent_suspension_policy,force_model='road_relative_unilateral_v3')
    physics=replace(f.physics_config,independent_suspension_policy=suspension,
                    spawn_reference='legacy_vehicle_position_v1')
    vehicle=replace(f.initial_state.vehicle,position=f.initial_state.vehicle.position+Vec3(0.,.06,0.),
                    velocity=Vec3(0.,-1.,0.),orientation=Quaternion.from_axis_angle(Vec3(1.,0.,0.),-.03))
    state=replace(f.initial_state,vehicle=vehicle,suspension=replace(f.initial_state.suspension,
        height_m=vehicle.position.y,heave_velocity_mps=-1.,pitch_x_rad=-.03))
    navigation=replace(f.navigation,points=(vehicle.position,*f.navigation.points[1:]))
    f=replace(f,physics_config=physics,initial_state=state,navigation=navigation,
              simulation_config=replace(f.simulation_config,fixed_dt=.001))
    env=f.create()
    try:
        observation,info=env.reset()
        assert info['physics']['configuration']['independent_suspension_policy']['force_model']=='road_relative_unilateral_v3'
        assert info['physics']['sha256']==physics.sha256
        observation,_,terminated,truncated,info=env.step(np.array([0.,0.,0.],dtype=np.float32))
        assert env.observation_space.contains(observation) and not terminated and not truncated
        audit=info['physics']['spatial_ground']
        first=audit['substeps'][0]['contact_stages'][0]
        assert [w['normal_force_n'] for w in first[:2]]==[0.,0.]
        assert all(w['normal_force_n']>0. for w in first[2:])
        assert audit['dissipation_j']>=0.
        assert env.requires_scientific_admission
    finally:
        env.close()


def test_unilateral_registration_rejects_a_silent_reduced_solver_fallback():
    import pytest
    f=spatial_factory()
    suspension=replace(f.physics_config.independent_suspension_policy,force_model='road_relative_unilateral_v3')
    with pytest.raises(ValueError,match='unilateral.*spatial'):
        replace(f.physics_config,independent_suspension_policy=suspension,
                ground_motion_version='legacy_euler_v1',spatial_contact_policy=None)
