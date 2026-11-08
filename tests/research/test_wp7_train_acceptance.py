"""Acceptance IPC/selection boundaries; actual four-track Gym tests live alongside."""
from pathlib import Path
import json
import pytest
from f1rl.research.wp7_protocol import digest
from f1rl.research.wp7_train_acceptance import verify_gym_finish_prefix,run_acceptance,_write_new

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('certificate',[None,{}, {'training_authorized':True},
    {'execution_profile':'pv5_v1','track_id':'heldout-track'}])
def test_invalid_or_unadmitted_certificate_denied_before_factory(certificate,monkeypatch):
    import f1rl.research.wp7_train_acceptance as module
    def forbidden(*args,**kwargs):
        raise AssertionError('invalid certificate must not inspect source or construct factory')
    monkeypatch.setattr(module,'build_protocol',forbidden)
    monkeypatch.setattr(module,'prepare_scenarios',forbidden)
    with pytest.raises(ValueError,match='TRAIN execution certificate'):
        verify_gym_finish_prefix(ROOT,certificate)


def test_acceptance_refuses_existing_output_and_source_tree_before_allocation(tmp_path,monkeypatch):
    import f1rl.research.wp7_train_acceptance as module
    monkeypatch.setattr(module,'build_protocol',lambda *a,**k:pytest.fail('output must be rejected first'))
    with pytest.raises(ValueError,match='new evidence directory'):
        run_acceptance(ROOT,tmp_path)
    with pytest.raises(ValueError,match='outside source tree'):
        run_acceptance(ROOT,ROOT/'not-created-acceptance-output')
    assert not (ROOT/'not-created-acceptance-output').exists()


def test_evidence_is_checksummed_and_never_overwrites_or_repairs_tampering(tmp_path):
    path=tmp_path/'evidence.json'
    sealed=_write_new(path,dict(status='software_only',training_authorized=False))
    assert json.loads(path.read_text(encoding='utf-8'))==sealed
    with pytest.raises(FileExistsError):
        _write_new(path,sealed)
    with pytest.raises(ValueError,match='checksum'):
        _write_new(tmp_path/'tampered.json',{**sealed,'training_authorized':True})
    assert not (tmp_path/'tampered.json').exists()


def test_foreign_loaded_code_is_rejected_before_factory_or_protocol(tmp_path,monkeypatch):
    import f1rl.learning.training.env_factory as factory_module
    import f1rl.research.wp7_train_acceptance as module
    monkeypatch.setattr(factory_module,'__file__',str(tmp_path/'foreign-install'/'env_factory.py'))
    def forbidden(*args,**kwargs):
        raise AssertionError('foreign loaded module must be rejected first')
    monkeypatch.setattr(module,'build_protocol',forbidden)
    c=dict(schema_version='wp7_executed_spatial_profile_feasibility_v2',execution_profile=module.PROFILE,
        track_id='straight-sprint-v1',status='executable_numerical_registered_spatial_path',
        finished=True,stopped=True,blockers=[],training_authorized=False,scientific_evidence=False,
        paired_seed_index=0,reset_rank=0,reset_index=0)
    c['sha256']=digest(c)
    with pytest.raises(ValueError,match='foreign F1RL source'):
        verify_gym_finish_prefix(ROOT,c)
