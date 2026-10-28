"""Versioned monotone phase crossings on continuous sloped road atlases."""
from dataclasses import replace
import pytest
from src.sandbox.geometry.vector import Vec3
from src.sandbox.vehicle import VehicleControl
from src.sandbox.spatial_contact_dynamics import integrate_spatial_lifecycle,SpatialContactPolicy
from src.sandbox.coupled_longitudinal_dynamics import LongitudinalContactPolicy
from src.sandbox.multi_surface_support import _atlas,_contacts
from src.sandbox.independent_suspension_dynamics import SuspensionResolutionError
from tests.sandbox.test_road_relative_suspension import ramp_translation


def sloped_landing():
    f,v,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    contacts=_contacts(v,s.contact,_atlas(f.track.surfaces,s.contact.seam_height_tolerance_m))
    extension=s.mass_kg*f.vehicle_parameters.gravity/(4.*s.spring_stiffness_npm)
    delta=max(c.deflection_m for c in contacts)+extension+.01
    v=replace(v,position=v.position+Vec3(0.,delta,0.),velocity=Vec3(0.,-1.,0.),grounded=False)
    policy=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.)
    return f,v,s,policy


def test_actual_train_ramp_landing_requires_explicit_atlas_phase_version():
    f,v,s,policy=sloped_landing()
    args=(v,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,VehicleControl(0.,0.,0.),policy)
    with pytest.raises(SuspensionResolutionError,match='local horizontal road'):
        integrate_spatial_lifecycle(*args,dt=.03)
    r=integrate_spatial_lifecycle(*args,dt=.03,phase_domain='atlas_gradient_v2')
    assert r.suspension.vehicle.grounded
    assert [b.kind for b in r.audit.phase_brackets]==['landing']
    assert r.audit.version=='spatial_compliant_atlas_lifecycle_rk4_v2'
    assert all(b.upper_time_s-b.lower_time_s<=1e-8 for b in r.audit.phase_brackets)
    assert all(abs(a.energy_residual_j)<=a.energy_tolerance_j and a.dissipation_j>=0. for a in r.audit.substeps)


def test_symmetric_mixed_normal_landing_uses_both_actual_patch_forces():
    from src.sandbox.track.patch import PlanarSurfacePatch
    f,_,_,s=ramp_translation()
    s=replace(s,force_model='road_relative_unilateral_v3')
    patches=tuple(PlanarSurfacePatch(str(sign),Vec3(sign*5.,.25,4.),Vec3(0.,0.,1.),
        Vec3(-sign*.05,1.,0.).normalized(),10.*(1.0025**.5),40.) for sign in (-1,1))
    v=f.initial_state.vehicle
    contacts=_contacts(v,s.contact,_atlas(patches,s.contact.seam_height_tolerance_m))
    extension=s.mass_kg*f.vehicle_parameters.gravity/(4.*s.spring_stiffness_npm)
    v=replace(v,position=v.position+Vec3(0.,max(c.deflection_m for c in contacts)+extension+.01,0.),
        velocity=Vec3(0.,-1.,0.),grounded=False)
    policy=SpatialContactPolicy(LongitudinalContactPolicy(.8,100000.),1800.,12000.)
    r=integrate_spatial_lifecycle(v,s,f.physics_config.body_geometry,patches,f.vehicle_parameters,
        VehicleControl(0.,0.,0.),policy,dt=.03,phase_domain='atlas_gradient_v2')
    assert r.suspension.vehicle.grounded
    assert [b.kind for b in r.audit.phase_brackets]==['landing']
    loaded={c.patch_id for a in r.audit.substeps for stage in a.contact_stages for c in stage if c.normal_force_n>0.}
    assert loaded=={'-1','1'}
    assert r.suspension.vehicle.angular_velocity.magnitude<1e-9


def test_public_simulation_accepts_versioned_sloped_landing_with_distinct_identity():
    from src.runtime.simulation import Simulation
    from src.sandbox.ballistic_flight import FlightState
    from src.sandbox.spatial_contact_dynamics import _suspension_state
    from tests.integration.test_spatial_compliant_gym import lifecycle_factory
    f,v,s,policy=sloped_landing()
    old=lifecycle_factory().physics_config
    config=replace(old,ground_motion_version='spatial_compliant_atlas_lifecycle_v2',
        independent_suspension_policy=s,spatial_contact_policy=policy)
    assert config.document()['schema_version']==7 and config.sha256!=old.sha256
    state=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v),flight=FlightState('airborne'))
    sim=Simulation(f.track,state,f.vehicle_parameters,replace(f.simulation_config,fixed_dt=.03),physics_config=config)
    snapshot=sim.step(VehicleControl(0.,0.,0.))
    assert snapshot.world.vehicle.grounded and snapshot.world.flight.phase=='grounded'
    assert snapshot.physics_transition is not None
    assert snapshot.spatial_ground_telemetry.version=='spatial_compliant_atlas_lifecycle_rk4_v2'


def test_sloped_touchdown_then_supported_recovery_preserves_passive_work_audit():
    f,v,s,policy=sloped_landing()
    events=[]
    initial_energy=None
    for _ in range(50):
        r=integrate_spatial_lifecycle(v,s,f.physics_config.body_geometry,f.track.surfaces,
            f.vehicle_parameters,VehicleControl(0.,0.,0.),policy,dt=.01,phase_domain='atlas_gradient_v2')
        if initial_energy is None:
            initial_energy=r.audit.substeps[0].initial_energy_j
        events.extend(b.kind for b in r.audit.phase_brackets)
        assert all(a.dissipation_j>=0. and abs(a.energy_residual_j)<=a.energy_tolerance_j for a in r.audit.substeps)
        v=r.suspension.vehicle
    assert events==['landing'] and v.grounded
    assert r.audit.substeps[-1].final_energy_j<=initial_energy+1e-5


def test_new_atlas_phase_domain_rejects_unknown_selector():
    f,v,s,policy=sloped_landing()
    with pytest.raises(ValueError,match='phase domain'):
        integrate_spatial_lifecycle(v,s,f.physics_config.body_geometry,f.track.surfaces,f.vehicle_parameters,
            VehicleControl(0.,0.,0.),policy,dt=.03,phase_domain='unregistered')


def test_actual_gym_owner_outcome_accepts_atlas_landing_with_valid_observation():
    import numpy as np
    from src.learning.wp7_race import WP7RaceConfig
    from src.sandbox.ballistic_flight import FlightState
    from src.sandbox.spatial_contact_dynamics import _suspension_state
    from tests.integration.test_spatial_compliant_gym import lifecycle_factory
    f,v,s,policy=sloped_landing()
    config=replace(lifecycle_factory().physics_config,ground_motion_version='spatial_compliant_atlas_lifecycle_v2',
        independent_suspension_policy=s,spatial_contact_policy=policy)
    f=replace(f,physics_config=config,initial_state=replace(f.initial_state,vehicle=v,
        suspension=_suspension_state(v),flight=FlightState('airborne')),
        simulation_config=replace(f.simulation_config,fixed_dt=.03),
        race_config=WP7RaceConfig(version='wp7_directed_lifecycle_owner_v2',hard_violation='terminate_valid_failure'),
        action_contract=lifecycle_factory().action_contract,
        navigation=replace(f.navigation,points=(v.position,*f.navigation.points[1:])),
        reset_config=None,reset_track_id=None)
    env=f.create()
    try:
        observation,_=env.reset()
        assert env.observation_space.contains(observation)
        observation,_,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
        assert env.observation_space.contains(observation) and not terminated and not truncated
        assert info['physics']['transition']['final_phase']=='grounded'
        assert env.requires_scientific_admission
    finally:
        env.close()


def test_previous_owner_profile_physics_hash_and_schema_do_not_migrate():
    from pathlib import Path
    from src.research.wp7_protocol import build_protocol
    protocol=build_protocol(Path(__file__).resolve().parents[2],profile='spatial_owner_v2')
    assert protocol['physics_sha256']=='1fc728b23f4fff8ccc5a683190608aaaa1d5d7a0f294c3079a4f771be4c2152f'
    assert protocol['physics']['schema_version']==6
    assert protocol['physics']['ground_motion_version']=='spatial_compliant_lifecycle_v1'


def test_atlas_transition_rejects_an_audit_claiming_the_previous_solver():
    from src.runtime.physics import PhysicsTransitionError
    from src.sandbox.world import step_world
    from src.sandbox.ballistic_flight import FlightState
    from src.sandbox.spatial_contact_dynamics import _suspension_state
    from tests.integration.test_spatial_compliant_gym import lifecycle_factory
    f,v,s,policy=sloped_landing()
    config=replace(lifecycle_factory().physics_config,ground_motion_version='spatial_compliant_atlas_lifecycle_v2',
        independent_suspension_policy=s,spatial_contact_policy=policy)
    state=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v),flight=FlightState('airborne'))
    r=step_world(f.track,state,VehicleControl(0.,0.,0.),f.vehicle_parameters,dt=.03,
        max_surface_distance=f.simulation_config.max_surface_distance,**config.world_options())
    forged=replace(r,spatial_ground_telemetry=replace(r.spatial_ground_telemetry,version='spatial_compliant_lifecycle_rk4_v1'))
    with pytest.raises(PhysicsTransitionError,match='audit missing or inconsistent'):
        config.validate_transition(forged,previous=state,parameters=f.vehicle_parameters,dt=.03,start_time_s=0.,end_time_s=.03)
