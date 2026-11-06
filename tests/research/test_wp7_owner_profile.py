"""Explicit coherent owner-v2 profile; no training/evaluation authorization."""
from pathlib import Path
from dataclasses import replace
import pytest
from f1rl.research.wp7_protocol import build_protocol,load_registration,prepare_scenarios,protocol_profile
from f1rl.research.wp7_execution import build_policy_execution
ROOT=Path(__file__).resolve().parents[2]
PROFILE='spatial_owner_v2'


def test_owner_profile_binds_compliant_lifecycle_race_reset_action_and_source():
    p=build_protocol(ROOT,profile=PROFILE)
    assert p['schema_version']=='wp7_bound_protocol_v3' and protocol_profile(p)==PROFILE
    assert p['physics']['mode']=='wp7_lifecycle_v1'
    assert p['physics']['ground_motion_version']=='spatial_compliant_lifecycle_v1'
    assert p['physics']['independent_suspension_policy']['force_model']=='road_relative_unilateral_v3'
    assert p['action']['version']=='steering-throttle-brake-v2'
    assert 'configs/research/wp7-hard-route-outcome-v2.json' in p['source_files']
    assert len(p['policies'])==15 and p['required_cells']==120 and p['minimum_evaluation_episodes']==7200
    assert not p['training_authorized']
    assert 'owner_failure_policy_missing' not in p['registration_blockers']
    scenarios=prepare_scenarios(ROOT,load_registration(ROOT,profile=PROFILE),seed_index=1)
    assert len(scenarios.scenarios)==4
    for row in scenarios.scenarios:
        f=row.factory
        assert f.initial_state.flight.phase=='grounded'
        assert f.physics_config.sha256==p['physics_sha256']
        assert f.race_config.version=='wp7_directed_lifecycle_owner_v2'
        assert f.reset_config.version=='wp7_train_reset_lifecycle_research_v1'
        assert f.reset_config.seed==1001 and f.requires_scientific_admission


def test_old_profiles_remain_distinct_and_profile_mismatch_rejects():
    old=build_protocol(ROOT)
    grounded=build_protocol(ROOT,profile='spatial_ground_v1')
    assert old['schema_version']=='wp7_bound_protocol_v1'
    assert grounded['schema_version']=='wp7_bound_protocol_v2'
    assert old['action']['version']=='steering-throttle-brake0-v1'
    assert grounded['physics']['mode']=='wp7_suspension_v1'
    bound=build_policy_execution(ROOT,'multitrack-seed1',profile=PROFILE)
    bound.validate(bound.configuration,bound.scenarios)
    with pytest.raises(ValueError):
        replace(bound,profile='spatial_ground_v1').validate(bound.configuration,bound.scenarios)


def test_actual_train_gym_resets_and_steps_all_four_bound_owner_factories():
    import numpy as np
    scenarios=prepare_scenarios(ROOT,load_registration(ROOT,profile=PROFILE),seed_index=2)
    for row in scenarios.scenarios:
        env=row.factory.create(reset_rank=1,reset_sequence=3)
        try:
            observation,info=env.reset(seed=42)
            assert env.observation_space.contains(observation)
            assert info['train_reset']['seed']==1002 and info['train_reset']['rank']==1
            observation,_,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
            assert env.observation_space.contains(observation) and not terminated and not truncated
            assert info['physics']['transition']['physical_mode']=='wp7_lifecycle_v1'
            assert info['race']['outcome_contract']['configuration']['outcome']=='valid_terminal_failure'
            assert info['race']['outcome_contract']['configuration']['training_permitted'] is False
        finally:
            env.close()


def test_owner_run_manifest_and_denied_runner_keep_exact_profile_before_artifacts(tmp_path):
    from f1rl.research.wp7_run_manifest import capture_registered_run,verify_run_manifest
    from f1rl.learning.training.runner import TrainingRunner
    from f1rl.research.wp7_admission import ScientificAdmissionError
    b=build_policy_execution(ROOT,'multitrack-seed2',profile=PROFILE)
    record=capture_registered_run(b,run_id='owner-v2-preparation')
    assert record['protocol']['schema_version']=='wp7_bound_protocol_v3'
    assert record['protocol_sha256']==b.protocol_sha256
    assert verify_run_manifest(record,None,expected_sha256=record['sha256'])['training_authorized'] is False
    bad=replace(b.configuration,artifacts_root=tmp_path/'never-allocate')
    with pytest.raises(ValueError):
        TrainingRunner(bad).run_registered(b)
    assert not bad.artifacts_root.exists()
    with pytest.raises(ScientificAdmissionError) as denied:
        TrainingRunner(b.configuration).run_registered(b)
    assert denied.value.report['protocol']['sha256']==b.protocol_sha256
    assert all(not value for value in denied.value.report['owner_permissions'].values())


def test_owner_evaluation_preparation_binds_independent_rng_without_running_a_model():
    from f1rl.research.wp7_protocol import prepare_evaluation_factory
    p=build_protocol(ROOT,profile=PROFILE)
    first,identity=prepare_evaluation_factory(ROOT,p,'multitrack-seed1','ramp-v1','randomized_train',7)
    second,repeated=prepare_evaluation_factory(ROOT,p,'multitrack-seed1','ramp-v1','randomized_train',7)
    assert first==second and identity==repeated
    assert first.reset_config.seed==identity['environment_seed']
    assert identity['environment_seed']!=identity['policy_rng_seed']
    assert not identity['training_authorized'] and not identity['deterministic_prediction']


@pytest.mark.parametrize('field,value',[('owner_outcome_contract','configs/research/unknown.json'),
    ('schema_version','wp7_experiment_registration_v2')])
def test_owner_registration_cannot_silently_select_another_contract(monkeypatch,field,value):
    import json
    registration=load_registration(ROOT,profile=PROFILE)
    registration[field]=value
    original=Path.read_text
    def read(path,*args,**kwargs):
        return json.dumps(registration) if path.name=='wp7-owner-execution-protocol-v2.json' else original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(ValueError,match='protocol'):
        load_registration(ROOT,profile=PROFILE)


def test_actual_train_slalom_scripted_departure_is_a_valid_failure_not_a_physics_error():
    import numpy as np
    f=next(row.factory for row in prepare_scenarios(ROOT,load_registration(ROOT,profile=PROFILE),seed_index=0).scenarios
           if row.track_id=='slalom-v1')
    env=replace(f,reset_config=None,reset_track_id=None).create()
    try:
        env.reset()
        for _ in range(100):
            observation,reward,terminated,truncated,info=env.step(np.asarray([.8,.5,0.],dtype=np.float32))
            assert env.observation_space.contains(observation)
            assert not truncated
            if terminated:
                break
        assert terminated and not info['finished']
        assert info['research_outcome']['kind']=='valid_terminal_failure'
        assert info['research_outcome']['valid_sample']
        assert info['race']['terminal_failure']
        assert 0.<=info['race']['reward_valid_prefix_s']<=f.simulation_config.fixed_dt
        assert set(info['reward_components'])=={'route_progress','collision','time'}
        assert env.requires_scientific_admission
    finally:
        env.close()


def test_owner_profile_accepts_exact_lf_parent_checkout_and_rejects_changes(monkeypatch):
    parent=ROOT/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml'
    original=Path.read_bytes
    checkout=original(parent).replace(b"\r\n",b"\n")
    def read(path):
        return checkout if path==parent else original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    assert load_registration(ROOT,profile=PROFILE)['schema_version']=='wp7_experiment_registration_v3'
    checkout+=b'# changed historical source\n'
    with pytest.raises(ValueError,match='historical parent'):
        load_registration(ROOT,profile=PROFILE)
