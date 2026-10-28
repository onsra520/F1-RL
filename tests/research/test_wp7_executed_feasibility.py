"""Control-tape physics witnesses on real TRAIN assets, never PPO evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.validation.wp7_executed_feasibility import certify_planar_execution, replay_planar_execution

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def audited():
    from src.validation.wp7_track_feasibility import audit_train_tracks
    return {row['track_id']:row['executed_planar_trajectory'] for row in
            audit_train_tracks(ROOT,execute_controls=True)['tracks']}


@pytest.fixture(scope='module',params=['straight-sprint-v1','slalom-v1'])
def executed(request,audited):
    return audited[request.param]


def test_real_train_path_has_executed_body_gate_contact_speed_and_stop_proof(executed):
    assert executed['status']=='executable_numerical_planar_path',executed['blockers']
    assert executed['finished'] and executed['stopped']
    assert executed['gate_ids']==executed['expected_gate_ids']
    assert executed['full_body_gate_passages']==executed['expected_gate_ids']
    assert executed['step_count']==len(executed['controls'])
    assert executed['max_speed_mps']<=executed['vehicle_parameters']['max_ground_speed']
    assert executed['proof_substeps']>executed['step_count']
    assert executed['training_authorized'] is False and executed['scientific_evidence'] is False
    assert any(c[2]>0 for c in executed['controls'])


def test_straight_control_tape_replays_exact_source_and_physics(audited):
    original = audited['straight-sprint-v1']
    restored = replay_planar_execution(ROOT,original,expected_sha256=original['sha256'])
    assert restored['sha256']==original['sha256']


@pytest.mark.parametrize('identity',['right-turn-v1','ramp-v1'])
def test_unresolved_real_track_is_never_promoted_by_a_control_tape(identity,audited):
    result = audited[identity]
    assert result['status']=='unresolved' and result['blockers']
    assert result['training_authorized'] is False


def test_replay_rejects_self_rehashed_tampered_controls_without_external_digest(executed):
    from src.validation.wp7_protocol import digest
    changed = deepcopy(executed)
    changed['controls'][0][0]=1.
    changed['sha256']=digest({k:v for k,v in changed.items() if k!='sha256'})
    with pytest.raises(ValueError,match='identity'):
        replay_planar_execution(ROOT,changed,expected_sha256=executed['sha256'])


def test_non_train_identity_rejected_before_asset_access():
    with pytest.raises(ValueError,match='TRAIN'):
        certify_planar_execution(ROOT,'not-registered')


@pytest.mark.parametrize('controls',[[],True,[[0.,0.]],[[True,0.,0.]],[[float('nan'),0.,0.]],[[0.,0.,2.]]])
def test_bad_tape_is_rejected_before_repository_access(tmp_path,controls):
    with pytest.raises(ValueError):
        certify_planar_execution(tmp_path/'absent','straight-sprint-v1',controls=controls)


def test_self_rehashed_short_tape_cannot_claim_recorded_success(audited):
    from src.validation.wp7_protocol import digest
    changed = deepcopy(audited['straight-sprint-v1'])
    changed['controls']=[[0.,0.,0.]]
    changed['sha256']=digest({k:v for k,v in changed.items() if k!='sha256'})
    with pytest.raises(ValueError,match='physical evidence identity'):
        replay_planar_execution(ROOT,changed,expected_sha256=changed['sha256'])


def test_gate_enclosure_rejects_rotation_that_endpoint_apertures_would_miss():
    from dataclasses import replace
    from math import pi
    from src.validation.wp7_executed_feasibility import _gate_segment_clear
    from src.simulation.geometry.rotation import Quaternion
    from src.simulation.geometry.vector import Vec3
    from src.simulation.track.volume import OrientedBoxVolume
    from src.simulation.vehicle import VehicleState
    zero = Vec3(0.,0.,0.)
    first = VehicleState(Vec3(0.,.5,0.),zero,Quaternion.from_axis_angle(Vec3(0.,1.,0.),-pi/2),zero,True)
    last = replace(first,orientation=Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/2))
    gate = OrientedBoxVolume(Vec3(0.,.8,0.),Vec3(2.1,2.,.001),Quaternion.identity())
    assert not _gate_segment_clear(first,last,gate)


def test_certificate_cannot_bind_a_different_checkout_root(tmp_path):
    with pytest.raises(ValueError,match='source origin'):
        certify_planar_execution(tmp_path,'straight-sprint-v1',controls=[[0.,0.,0.]])


def test_parent_simulation_monkeypatch_cannot_enter_execution_worker(monkeypatch):
    import src.validation.wp7_executed_feasibility as executor
    def forbidden(*args,**kwargs):
        raise AssertionError('parent Simulation reached')
    monkeypatch.setattr(executor,'Simulation',forbidden)
    result = certify_planar_execution(ROOT,'straight-sprint-v1',controls=[[0.,0.,0.]])
    assert result['status']=='unresolved' and result['step_count']==1
    assert result['execution_source_sha256']


def test_loaded_wrapper_is_rejected_after_its_source_changes(tmp_path):
    import shutil
    import subprocess
    import sys
    copied = tmp_path/'checkout'
    shutil.copytree(ROOT/'src',copied/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    program = '''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import src.validation.wp7_executed_feasibility as executor
source = Path(executor.__file__)
source.write_bytes(source.read_bytes()+b'\\n# changed after import\\n')
try:
    executor.certify_planar_execution(Path(sys.argv[1]),'straight-sprint-v1',controls=[[0.,0.,0.]])
except ValueError as error:
    assert 'source origin' in str(error)
    print('stale wrapper rejected')
else:
    raise AssertionError('modified loaded wrapper was accepted')
'''
    completed = subprocess.run([sys.executable,'-I','-B','-c',program,str(copied)],
                               text=True,capture_output=True,check=True,timeout=45)
    assert completed.stdout.strip()=='stale wrapper rejected'
