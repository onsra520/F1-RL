"""Declared canonical Ramp fixtures; no research policy or robustness claim."""
from dataclasses import replace
from math import atan2,cos,radians,tan
from pathlib import Path

import numpy as np
import pytest
from f1rl.research.wp7_protocol import load_registration,prepare_scenarios
from f1rl.sandbox.ballistic_flight import FlightState
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import SuspensionResolutionError
from f1rl.sandbox.multi_surface_support import _atlas,_contacts
from f1rl.sandbox.spatial_contact_dynamics import _suspension_state
from f1rl.track_assets.canonical import fingerprint_track_world

ROOT=Path(__file__).resolve().parents[2]
PHYSICS_SHA='fe6e753f1471e83f8103bae5ad64e050d316c3ddb9186c51cba3c27a3e8ddf5b'
RAMP_WORLD_SHA='775f795aa2763f64916e68e6a64fc763021e875c384050a974c747b6b4907978'


def recovery_factory(mixed_normal):
    f=next(row.factory for row in prepare_scenarios(ROOT,load_registration(ROOT,
        profile='spatial_rotating_owner_v3'),seed_index=0).scenarios if row.track_id=='ramp-v1')
    assert f.physics_config.sha256==PHYSICS_SHA
    assert fingerprint_track_world(f.track)==RAMP_WORLD_SHA
    ramp=next(p for p in f.track.surfaces if p.id=='road-incline')
    angle=atan2(ramp.forward.y,ramp.forward.z)
    z=50. if mixed_normal else 80.
    height=.5 if mixed_normal else ramp.origin.y+(z-ramp.origin.z)*tan(angle)+.5/cos(angle)
    pitch=radians(-4.) if mixed_normal else -angle
    v=replace(f.initial_state.vehicle,position=Vec3(0.,height,z),velocity=Vec3(0.,-1.,0.),
        orientation=Quaternion.from_axis_angle(Vec3(1.,0.,0.),pitch),grounded=False)
    config=replace(f.physics_config,spawn_reference='legacy_vehicle_position_v1')
    registered=f.physics_config.document();fixture=config.document()
    assert {k:v for k,v in registered.items() if k!='spawn_reference'}=={k:v for k,v in fixture.items() if k!='spawn_reference'}
    s=config.independent_suspension_policy
    contacts=_contacts(v,s.contact,_atlas(f.track.surfaces,s.contact.seam_height_tolerance_m))
    extension=s.mass_kg*f.vehicle_parameters.gravity/(4.*s.spring_stiffness_npm)
    v=replace(v,position=v.position+Vec3(0.,max(c.deflection_m for c in contacts)+extension+.01,0.))
    # Declared airborne fixture uses vehicle-position spawn; dynamics and assets stay pinned.
    return replace(f,physics_config=config,initial_state=replace(f.initial_state,vehicle=v,suspension=_suspension_state(v),
        flight=FlightState('airborne')),simulation_config=replace(f.simulation_config,fixed_dt=.01),
        navigation=replace(f.navigation,points=(v.position,*f.navigation.points[1:])),
        reset_config=None,reset_track_id=None)


@pytest.mark.parametrize('mixed_normal',[False,True],ids=['incline','approach-incline-seam'])
def test_canonical_ramp_touchdown_and_persistent_recovery_through_gym(mixed_normal):
    factory=recovery_factory(mixed_normal)
    env=factory.create()
    events=[];loaded_mixed_frames=0;landed=False
    normals={p.id:p.normal for p in factory.track.surfaces}
    try:
        observation,_=env.reset()
        assert env.observation_space.contains(observation)
        assert factory.initial_state.flight.phase=='airborne' and not factory.initial_state.vehicle.grounded
        for _ in range(50):
            observation,reward,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
            assert env.observation_space.contains(observation) and np.isfinite(reward)
            assert not terminated and not truncated
            assert info['research_outcome']['kind']=='active'
            audit=info['physics']['transition']
            assert audit['initial_phase']==('grounded' if landed else 'airborne')
            events.extend(audit['events'])
            landed=landed or any(e['kind']=='landing' for e in audit['events'])
            if landed:
                assert audit['final_phase']=='grounded' and info['telemetry']['grounded']
            assert 'spatial_rotating_compliant_lifecycle_rk4_v3' in audit['solvers']
            for bracket in info['physics']['spatial_ground']['phase_brackets']:
                assert 0.<=bracket['lower_time_s']<=bracket['upper_time_s']<=.01
                assert bracket['upper_time_s']-bracket['lower_time_s']<=1e-8
            for leaf in info['physics']['spatial_ground']['substeps']:
                assert leaf['actuator_work_j']==0. and leaf['dissipation_j']>=0.
                assert abs(leaf['energy_residual_j'])<=leaf['energy_tolerance_j']
                assert leaf['linear_impulse_residual_ns']<=1e-7 and leaf['angular_impulse_residual_nms']<=1e-7
                for stage in leaf['contact_stages']:
                    loaded={normals[c['patch_id']] for c in stage if c['normal_force_n']>0.}
                    loaded_mixed_frames+=len(loaded)>1
        assert [e['kind'] for e in events]==['landing'] and 0.<events[0]['time_s']<=.01
        assert landed and factory.physics_config.sha256=='a812522f2d60dad4059dcf735e6ec88785fb75915280bad213a0899f79f4a87e'
        assert fingerprint_track_world(factory.track)==RAMP_WORLD_SHA
        assert env.requires_scientific_admission
        if mixed_normal:
            assert loaded_mixed_frames>0
    finally:
        env.close()


@pytest.mark.parametrize('mixed_normal',[False,True])
def test_historical_horizontal_selector_rejects_sloped_touchdown(mixed_normal):
    f=recovery_factory(mixed_normal)
    f=replace(f,physics_config=replace(f.physics_config,ground_motion_version='spatial_compliant_lifecycle_v1'))
    env=f.create()
    try:
        env.reset()
        with pytest.raises(SuspensionResolutionError,match='local horizontal road'):
            env.step(np.zeros(3,dtype=np.float32))
    finally:
        env.close()
