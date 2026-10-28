"""Actual rotating owner profile binding, TRAIN reset and closed execution."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from src.research.wp7_protocol import build_protocol,load_registration,prepare_scenarios,protocol_profile
from src.research.wp7_execution import build_policy_execution
ROOT=Path(__file__).resolve().parents[2]
PROFILE='spatial_rotating_owner_v3'


def test_rotating_owner_profile_is_explicit_and_preserves_every_registered_cohort_threshold():
    p=build_protocol(ROOT,profile=PROFILE)
    old=build_protocol(ROOT,profile='spatial_owner_v2')
    assert p['schema_version']=='wp7_bound_protocol_v4' and protocol_profile(p)==PROFILE
    assert p['physics']['schema_version']==8
    assert p['physics']['ground_motion_version']=='spatial_rotating_compliant_lifecycle_v3'
    assert p['physics_sha256']!=old['physics_sha256']
    assert old['physics_sha256']=='1fc728b23f4fff8ccc5a683190608aaaa1d5d7a0f294c3079a4f771be4c2152f'
    for key in ('policies','required_cells','minimum_evaluation_episodes','action','rewards'):
        assert p[key]==old[key]
    assert not p['training_authorized']
    assert 'configs/research/wp7-hard-route-outcome-v2.json' in p['source_files']


def test_four_actual_train_factories_reset_and_step_with_one_rotating_physics_identity():
    p=build_protocol(ROOT,profile=PROFILE)
    scenarios=prepare_scenarios(ROOT,load_registration(ROOT,profile=PROFILE),seed_index=1)
    for row in scenarios.scenarios:
        assert row.factory.physics_config.sha256==p['physics_sha256']
        assert row.factory.race_config.version=='wp7_directed_lifecycle_owner_v2'
        env=row.factory.create(reset_rank=1,reset_sequence=2)
        try:
            observation,info=env.reset(seed=20)
            assert info['train_reset']['seed']==1001 and info['train_reset']['rank']==1
            assert env.observation_space.contains(observation)
            observation,_,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
            assert env.observation_space.contains(observation) and not terminated and not truncated
            assert 'spatial_rotating_compliant_lifecycle_rk4_v3' in info['physics']['transition']['solvers']
            assert env.requires_scientific_admission
        finally:
            env.close()


def test_binding_rejects_profile_migration_and_training_stays_closed():
    from src.research.wp7_admission import ScientificAdmissionError
    from src.learning.training.runner import TrainingRunner
    b=build_policy_execution(ROOT,'multitrack-seed2',profile=PROFILE)
    b.validate(b.configuration,b.scenarios)
    with pytest.raises(ValueError):
        replace(b,profile='spatial_owner_v2').validate(b.configuration,b.scenarios)
    with pytest.raises(ScientificAdmissionError) as denied:
        TrainingRunner(b.configuration).run_registered(b)
    assert denied.value.report['protocol']['sha256']==b.protocol_sha256
    assert all(not value for value in denied.value.report['owner_permissions'].values())
    from src.research.wp7_run_manifest import capture_registered_run,verify_run_manifest
    manifest=capture_registered_run(b,run_id='rotating-owner-preparation')
    assert manifest['protocol']['schema_version']=='wp7_bound_protocol_v4'
    assert verify_run_manifest(manifest,None,expected_sha256=manifest['sha256'])['training_authorized'] is False


def test_new_registration_cannot_silently_change_motion_version(monkeypatch):
    import json
    registration=load_registration(ROOT,profile=PROFILE)
    registration['ground_motion_version']='spatial_compliant_lifecycle_v1'
    original=Path.read_text
    def read(path,*args,**kwargs):
        return json.dumps(registration) if path.name=='wp7-rotating-owner-protocol-v3.json' else original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(ValueError,match='unregistered protocol'):
        load_registration(ROOT,profile=PROFILE)


def test_rotating_evaluation_preparation_separates_policy_and_environment_rng():
    from src.research.wp7_protocol import prepare_evaluation_factory
    p=build_protocol(ROOT,profile=PROFILE)
    first,identity=prepare_evaluation_factory(ROOT,p,'multitrack-seed1','ramp-v1','randomized_train',7)
    second,repeated=prepare_evaluation_factory(ROOT,p,'multitrack-seed1','ramp-v1','randomized_train',7)
    assert first==second and identity==repeated
    assert first.physics_config.sha256==p['physics_sha256']
    assert first.reset_config.seed==identity['environment_seed']
    assert identity['environment_seed']!=identity['policy_rng_seed']
    assert not identity['training_authorized'] and not identity['deterministic_prediction']


def test_actual_rotating_slalom_run_retains_owner_failure_and_reward_contract():
    scenarios=prepare_scenarios(ROOT,load_registration(ROOT,profile=PROFILE),seed_index=0)
    f=next(row.factory for row in scenarios.scenarios if row.track_id=='slalom-v1')
    env=replace(f,reset_config=None,reset_track_id=None).create()
    try:
        env.reset()
        for _ in range(100):
            observation,_,terminated,truncated,info=env.step(np.asarray([.8,.5,0.],dtype=np.float32))
            assert env.observation_space.contains(observation) and not truncated
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='valid_terminal_failure'
        assert info['research_outcome']['valid_sample']
        assert set(info['reward_components'])=={'route_progress','collision','time'}
        assert env.requires_scientific_admission
    finally:
        env.close()
