"""Registered worker plumbing never authorizes a research rollout."""
from hashlib import sha256
from pathlib import Path
import pytest
from src.validation.wp7_protocol import build_protocol
from src.validation.wp7_admission import ScientificAdmissionError
from src.validation.wp7_cell_evaluation import evaluate_registered_cell
ROOT=Path(__file__).resolve().parents[3]
PROFILE='spatial_rotating_owner_v3'


def test_registered_parent_and_worker_deny_before_model_or_environment_access(tmp_path):
    protocol=build_protocol(ROOT,profile=PROFILE)
    with pytest.raises(ScientificAdmissionError):
        evaluate_registered_cell(ROOT,protocol,policy_id='multitrack-seed1',track_id='ramp-v1',
            stratum='randomized_train',model_path=tmp_path/'missing.zip',expected_model_sha256='0'*64)
    from scripts.internal.ppo_fixture_worker import run_registered
    result=run_registered(dict(protocol=protocol,policy_id='multitrack-seed1',track_id='ramp-v1',
        stratum='randomized_train',authorization=True,model_bytes=b'invalid archive'))
    assert result[0]=='error' and result[1]=='ScientificAdmissionError'
    assert 'blocked' in result[2]


def bound_model(tmp_path):
    from src.validation.wp7_execution import build_policy_execution
    from src.validation.wp7_run_manifest import capture_registered_run,bind_run_artifacts
    binding=build_policy_execution(ROOT,'multitrack-seed1',profile=PROFILE)
    prepared=capture_registered_run(binding,run_id='opaque-model-binding-not-training')
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    model=artifacts/'ppo-multitrack-seed1-sample1-final-model.zip';model.write_bytes(b'opaque model bytes; never deserialized')
    manifest=bind_run_artifacts(prepared,artifacts)
    return dict(protocol=prepared['protocol'],policy_id='multitrack-seed1',model_path=model,
        expected_model_sha256=sha256(model.read_bytes()).hexdigest(),run_manifest=manifest,
        expected_run_manifest_sha256=manifest['sha256'],artifact_directory=artifacts)


def test_checkpoint_snapshot_binds_actual_bytes_policy_and_run_without_authorizing(tmp_path):
    from src.validation.wp7_cell_evaluation import _registered_model_snapshot
    args=bound_model(tmp_path)
    blob,identity=_registered_model_snapshot(**args)
    assert blob==args['model_path'].read_bytes()
    assert identity['sha256']==args['expected_model_sha256']
    assert identity['run_manifest_sha256']==args['expected_run_manifest_sha256']
    assert identity['artifact_path']=='ppo-multitrack-seed1-sample1-final-model.zip'
    for change in (dict(policy_id='multitrack-seed0'),dict(expected_model_sha256='0'*64),
                   dict(expected_run_manifest_sha256='0'*64)):
        with pytest.raises(ValueError):
            _registered_model_snapshot(**{**args,**change})
    args['model_path'].write_bytes(b'changed')
    with pytest.raises(ValueError):
        _registered_model_snapshot(**args)


def test_parent_boolean_cannot_admit_the_independent_worker_or_publish_evidence(tmp_path,monkeypatch):
    import src.validation.wp7_cell_evaluation as module
    args=bound_model(tmp_path)
    fake=dict(training_authorized=True,authorization=dict(verified=True,
        protocol_sha256=args['protocol']['sha256'],run_manifest_sha256=args['expected_run_manifest_sha256'],
        model_sha256=args['expected_model_sha256'],policy_id=args['policy_id']),
        owner_permissions=dict(training_permitted=True))
    monkeypatch.setattr(module,'evaluate_admission',lambda *a,**k:fake)
    output=tmp_path/'must-not-exist.json'
    monkeypatch.setattr(module, '_spawn_ppo_cell', lambda *a, **k: pytest.fail('unsigned report must not start worker'))
    with pytest.raises(ScientificAdmissionError,match='authenticated'):
        evaluate_registered_cell(ROOT,**args,track_id='ramp-v1',stratum='nominal',evidence_path=output)
    assert not output.exists()


def test_child_rechecks_signatures_even_if_its_report_booleans_are_forged(monkeypatch):
    import src.validation.wp7_cell_evaluation as module
    from scripts.internal.ppo_fixture_worker import run_registered
    protocol = build_protocol(ROOT, profile=PROFILE)
    fake = dict(training_authorized=True, authorization=dict(verified=True,
        protocol_sha256=protocol['sha256'], policy_id='multitrack-seed1',
        model_sha256='a' * 64, run_manifest_sha256='b' * 64),
        owner_permissions=dict(training_permitted=True))
    monkeypatch.setattr(module, 'evaluate_admission', lambda *a, **k: fake)
    result = run_registered(dict(protocol=protocol, policy_id='multitrack-seed1',
        track_id='ramp-v1', stratum='nominal', expected_model_sha256='a' * 64,
        expected_run_manifest_sha256='b' * 64, model_bytes=b'never deserialize'))
    assert result[0:2] == ('error', 'ScientificAdmissionError')
    assert 'authenticated' in result[2]
