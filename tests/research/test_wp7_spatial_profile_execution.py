"""Real all-TRAIN profile execution with source-bound replay and Gym integration."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from src.validation.wp7_executed_feasibility import certify_spatial_profile_execution,replay_planar_execution
from src.validation.wp7_protocol import load_registration,prepare_scenarios,build_protocol

ROOT=Path(__file__).resolve().parents[2]
TRACKS=('straight-sprint-v1','slalom-v1','right-turn-v1','ramp-v1')


@pytest.fixture(scope='module')
def nominal():
    return {t:certify_spatial_profile_execution(ROOT,t) for t in TRACKS}


@pytest.mark.parametrize('track',TRACKS)
def test_every_actual_train_has_same_registered_spatial_physics_finish_and_stop(track,nominal):
    r=nominal[track]
    protocol=build_protocol(ROOT,profile='spatial_ground_v1')
    assert r['status']=='executable_numerical_registered_spatial_path',r['blockers']
    assert r['schema_version']=='wp7_executed_spatial_profile_feasibility_v1'
    assert r['physics_sha256']==r['registered_physics_sha256']==protocol['physics_sha256']
    assert r['protocol_sha256']==protocol['sha256']
    assert r['execution_profile']=='spatial_ground_v1'
    assert r['gate_contract']=='directed_center_crossing_v1'
    assert r['gate_ids']==r['expected_gate_ids']
    assert r['finished'] and r['stopped'] and r['step_count']<=500
    assert r['max_sampled_route_deviation_m']<4.
    assert r['spatial_work_audit']['accepted_substeps']==r['proof_substeps']
    assert r['spatial_work_audit']['max_energy_residual_fraction']<=1.
    assert r['reset_metadata'] is None
    assert not r['training_authorized'] and not r['scientific_evidence']


@pytest.mark.parametrize('track',TRACKS)
def test_each_registered_spatial_control_tape_completes_actual_gym(track,nominal):
    r=nominal[track]
    f=next(s.factory for s in prepare_scenarios(ROOT,load_registration(ROOT,profile='spatial_ground_v1'),seed_index=0).scenarios
           if s.track_id==track)
    env=replace(f,reset_config=None,reset_track_id=None).create()
    try:
        observation,_=env.reset(seed=19)
        for control in r['controls']:
            observation,_,terminated,truncated,info=env.step(np.asarray(control,dtype=np.float32))
            assert env.observation_space.contains(observation)
            assert not truncated
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='successful_finish'
        assert info['physics']['sha256']==r['physics_sha256']
        assert env.requires_scientific_admission
    finally:
        env.close()


def test_registered_spatial_right_turn_replays_exact_profile_and_source(nominal):
    r=nominal['right-turn-v1']
    assert replay_planar_execution(ROOT,r,expected_sha256=r['sha256'])==r


@pytest.mark.parametrize('selection',[dict(seed_index=True),dict(seed_index=-1),dict(reset_rank=True),
                                     dict(reset_index=-1),dict(reset_index=True)])
def test_invalid_spatial_selection_rejected_before_source_access(tmp_path,selection):
    with pytest.raises(ValueError,match='selection'):
        certify_spatial_profile_execution(tmp_path/'absent','ramp-v1',**selection)


def test_perturbed_selection_records_actual_and_registered_initial_state_separately():
    from dataclasses import asdict
    from src.rl.train_reset import sample_train_spawn
    from src.rl.route_navigation import AuthoredRoute
    from src.validation.wp7_protocol import digest
    f=prepare_scenarios(ROOT,load_registration(ROOT,profile='spatial_ground_v1'),seed_index=1).scenarios[0].factory
    initial,metadata=sample_train_spawn(f.reset_config,TRACKS[0],f.track,f.initial_state,f.physics_config,
                                      AuthoredRoute(f.navigation),rank=2,reset_index=7)
    r=certify_spatial_profile_execution(ROOT,TRACKS[0],seed_index=1,reset_rank=2,reset_index=7,controls=[[0.,0.,0.]])
    assert r['initial_state_sha256']==digest(json.loads(json.dumps(asdict(initial),default=sorted)))
    assert r['registered_initial_state_sha256']==digest(json.loads(json.dumps(asdict(f.initial_state),default=sorted)))
    assert r['initial_state_sha256']!=r['registered_initial_state_sha256']
    assert r['reset_metadata']==json.loads(json.dumps(metadata))
    assert replay_planar_execution(ROOT,r,expected_sha256=r['sha256'])==r


@pytest.mark.parametrize('track,index',list(zip(TRACKS,(0,1,2,0))))
def test_declared_randomized_spawn_completes_actual_simulation_and_gym(track,index):
    r=certify_spatial_profile_execution(ROOT,track,seed_index=index,reset_rank=2,reset_index=7)
    assert r['status']=='executable_numerical_registered_spatial_path',r['blockers']
    assert r['reset_metadata']['seed']==1000+index
    assert r['reset_metadata']['rank']==2 and r['reset_metadata']['reset_index']==7
    assert abs(r['reset_metadata']['lateral_offset_m'])<=.5
    assert abs(r['reset_metadata']['heading_jitter_deg'])<=5.
    f=next(s.factory for s in prepare_scenarios(ROOT,load_registration(ROOT,profile='spatial_ground_v1'),seed_index=index).scenarios
           if s.track_id==track)
    env=f.create(reset_rank=2,reset_sequence=7)
    try:
        observation,info=env.reset(seed=843)
        assert json.loads(json.dumps(info['train_reset']))==r['reset_metadata']
        for control in r['controls']:
            observation,_,terminated,truncated,info=env.step(np.asarray(control,dtype=np.float32))
            assert env.observation_space.contains(observation) and not truncated
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='successful_finish'
    finally:
        env.close()
