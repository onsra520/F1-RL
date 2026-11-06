"""Explicit all-TRAIN spatial registration, never scientific training permission."""
from dataclasses import replace
from pathlib import Path

import pytest

from f1rl.research.wp7_protocol import build_protocol,load_registration,prepare_scenarios,validate_protocol
from f1rl.research.wp7_execution import build_policy_execution

ROOT=Path(__file__).resolve().parents[2]
PROFILE='spatial_ground_v1'


def test_new_profile_binds_same_spatial_physics_and_three_actions_for_all_train():
    old=build_protocol(ROOT)
    p=build_protocol(ROOT,profile=PROFILE)
    assert old['schema_version']=='wp7_bound_protocol_v1'
    assert old['action']['version']=='steering-throttle-brake0-v1'
    assert p['schema_version']=='wp7_bound_protocol_v2'
    assert p['physics']['ground_motion_version']=='spatial_contact_v1'
    assert p['action']['version']=='steering-throttle-brake-v2'
    assert p['action']['low']==[-1.,0.,0.]
    assert p['physics_sha256']!=old['physics_sha256']
    assert p['registration']['required_domains']==old['registration']['required_domains']
    assert p['registration']['evaluation']==old['registration']['evaluation']
    assert p['registration']['seed_roster']==old['registration']['seed_roster']
    assert len(p['policies'])==15 and p['required_cells']==120
    assert p['minimum_evaluation_episodes']==7200
    assert not p['training_authorized'] and p['registration_blockers']
    validate_protocol(p,ROOT)
    factories=prepare_scenarios(ROOT,load_registration(ROOT,profile=PROFILE),seed_index=0)
    assert len(factories.scenarios)==4
    assert {s.factory.physics_config.sha256 for s in factories.scenarios}=={p['physics_sha256']}
    assert {s.factory.action_contract.version for s in factories.scenarios}=={'steering-throttle-brake-v2'}


@pytest.mark.parametrize('index',[0,1,2])
def test_spatial_policy_binding_reconstructs_exact_profile_and_seed_bundle(index):
    b=build_policy_execution(ROOT,f'multitrack-seed{index}',profile=PROFILE)
    assert b.profile==PROFILE
    assert b.receipt['seed_bundle']==dict(ppo=index,sampling=index,perturbation=1000+index)
    assert b.receipt['action_contract']['version']=='steering-throttle-brake-v2'
    assert len(b.scenarios.scenarios)==4
    b.validate(b.configuration,b.scenarios)
    with pytest.raises(ValueError,match='protocol/source drift|configuration|TRAIN'):
        replace(b,profile='pv5_v1').validate(b.configuration,b.scenarios)


def test_spatial_run_manifest_and_denial_keep_the_actual_profile_before_allocation(tmp_path):
    from f1rl.research.wp7_run_manifest import capture_registered_run,verify_run_manifest
    from f1rl.learning.training.runner import TrainingRunner
    from f1rl.research.wp7_admission import ScientificAdmissionError
    b=build_policy_execution(ROOT,'multitrack-seed1',profile=PROFILE)
    record=capture_registered_run(b,run_id='spatial-preparation')
    assert record['protocol_sha256']==b.protocol_sha256
    assert record['protocol']['physics']['ground_motion_version']=='spatial_contact_v1'
    assert verify_run_manifest(record,None,expected_sha256=record['sha256'])['training_authorized'] is False
    # Replacing an output path is an execution mismatch, before any allocation.
    bad=replace(b.configuration,artifacts_root=tmp_path/'absent')
    with pytest.raises(ValueError):
        TrainingRunner(bad).run_registered(b)
    assert not bad.artifacts_root.exists()
    with pytest.raises(ScientificAdmissionError) as denied:
        TrainingRunner(b.configuration).run_registered(b)
    assert denied.value.report['protocol']['sha256']==b.protocol_sha256
    assert not denied.value.report['training_authorized']


@pytest.mark.parametrize('profile',[None,True,'unknown','../wp7-pv5-protocol-v1.json'])
def test_unknown_profile_never_falls_back(profile):
    with pytest.raises(ValueError,match='profile'):
        build_protocol(ROOT,profile=profile)


def test_spatial_audit_requires_actual_execution_before_source_access(tmp_path):
    from f1rl.research.wp7_track_feasibility import audit_train_tracks
    with pytest.raises(ValueError,match='profile.*execution'):
        audit_train_tracks(tmp_path/'absent',execution_profile=PROFILE)


@pytest.mark.parametrize('field,value',[('yaw_inertia_kg_m2',-1.),('lateral_damping_nspm',float('nan')),
                                       ('max_angular_speed_radps',False),('unknown',1.)])
def test_spatial_registration_rejects_malformed_contact_policy(tmp_path,field,value):
    import json
    registration=load_registration(ROOT,profile=PROFILE)
    registration['engineering_spatial_contact'][field]=value
    folder=tmp_path/'configs/research'
    folder.mkdir(parents=True)
    (folder/'wp7-spatial-ground-protocol-v1.json').write_text(json.dumps(registration),encoding='utf-8')
    with pytest.raises(ValueError):
        load_registration(tmp_path,profile=PROFILE)


def test_cli_never_ignores_reset_selection_for_a_different_execution_mode():
    import os
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-B','scripts/wp7_certify_planar_execution.py',
                           '--track','straight-sprint-v1','--reset-index','7'],cwd=ROOT,
                          env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'},
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==2
    assert 'selectors require --spatial-profile-track' in result.stderr
    assert not result.stdout
