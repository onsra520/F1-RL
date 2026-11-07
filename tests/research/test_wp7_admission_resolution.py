"""Admission diagnostics distinguish authored sweeps, execution and authority."""
from pathlib import Path
import json

import pytest

from f1rl.research.wp7_admission import ScientificAdmissionError, evaluate_admission
from f1rl.research.wp7_protocol import build_protocol

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'spatial_rotating_owner_v3'


def test_rotating_admission_preserves_authored_findings_without_claiming_missing_controller():
    report = evaluate_admission(ROOT, capture_runtime=False, profile=PROFILE)
    blockers = {row['id'] for row in report['blockers']}
    assert 'speed_and_braking_protocol_missing' not in blockers
    assert not any(identity.startswith('track_feasibility:') for identity in blockers)
    assert 'registered_execution_evidence_required' in blockers
    assert 'general_contact_domain_incomplete' not in blockers
    assert report['track_audit']['tracks']
    assert any(row['geometry_status'] != 'certified_authored_sweep'
               for row in report['track_audit']['tracks'])
    assert report['authored_track_audit_scope'] == 'historical_authored_sweep_not_alternate_execution'
    assert not report['training_authorized']
    assert all(value is False for value in report['owner_permissions'].values())


def test_forged_admission_booleans_fail_at_parent_before_worker_or_model_access(monkeypatch):
    import f1rl.research.wp7_cell_evaluation as cell
    protocol = build_protocol(ROOT, profile=PROFILE)
    fake = dict(training_authorized=True, authorization=dict(verified=True,
        protocol_sha256=protocol['sha256'], policy_id='multitrack-seed1',
        model_sha256='a' * 64, run_manifest_sha256='b' * 64),
        owner_permissions=dict(training_permitted=True))
    monkeypatch.setattr(cell, 'evaluate_admission', lambda *args, **kwargs: fake)
    with pytest.raises(ScientificAdmissionError, match='authenticated'):
        cell._registered_cell_admission(ROOT, protocol, 'multitrack-seed1',
            model_sha256='a' * 64, run_manifest_sha256='b' * 64)


def test_repository_anchor_verifier_authenticates_signatures_and_denies_self_anchor(tmp_path):
    from tests.research.test_wp7_trusted_reviews import signed_review, NOW
    import f1rl.research.wp7_trusted_reviews as reviews
    envelope, registry = signed_review()
    directory = tmp_path / 'configs/research'
    directory.mkdir(parents=True)
    (directory / 'wp7-trusted-reviewers-v1.json').write_text(json.dumps(registry), encoding='utf-8')
    result = reviews.verify_repository_run_reviews(tmp_path, envelope,
        protocol_sha256='a' * 64, run_manifest_sha256='b' * 64, now=NOW)
    assert result['state'] == 'trusted_reviews_verified'
    assert not any(result['permissions'].values())
    (directory / 'wp7-trusted-reviewers-v1.json').write_text(
        json.dumps(dict(schema_version='wp7_trusted_reviewers_v1', reviewers=[])), encoding='utf-8')
    with pytest.raises(ValueError, match='unknown'):
        reviews.verify_repository_run_reviews(tmp_path, envelope,
            protocol_sha256='a' * 64, run_manifest_sha256='b' * 64, now=NOW)


def test_registered_runner_denial_carries_its_verified_run_identity_before_allocation():
    from f1rl.research.wp7_execution import build_policy_execution
    from f1rl.learning.training.runner import TrainingRunner
    binding = build_policy_execution(ROOT, 'multitrack-seed0', profile=PROFILE)
    # Output is not a registered parameter: keep the actual config and check no allocation.
    runner = TrainingRunner(binding.configuration)
    with pytest.raises(ScientificAdmissionError) as caught:
        runner.run_registered(binding)
    request = caught.value.report['execution_request']
    assert request['policy_id'] == binding.policy_id
    assert request['protocol_sha256'] == binding.protocol_sha256
    assert request['run_manifest_sha256'] == runner.registered_run_manifest['sha256']
    assert request['state'] == 'preparation_verified'
    assert not caught.value.report['training_authorized']


def test_review_without_run_manifest_is_rejected_instead_of_accepted_as_authority():
    with pytest.raises(ValueError, match='run manifest'):
        evaluate_admission(ROOT, profile=PROFILE, capture_runtime=False,
            review_envelope=dict(verified=True))


def test_preparation_verification_does_not_claim_restored_artifact_bytes(tmp_path):
    from f1rl.research.wp7_execution import build_policy_execution
    import f1rl.research.wp7_run_manifest as manifests
    binding = build_policy_execution(ROOT, 'multitrack-seed0', profile=PROFILE)
    prepared = manifests.capture_registered_run(binding, run_id='metadata-only-fixture')
    artifact = tmp_path / 'opaque.zip'
    artifact.write_bytes(b'opaque fixture; never deserialized')
    bound = manifests.bind_run_artifacts(prepared, tmp_path)
    artifact.unlink()
    result = manifests.verify_run_preparation(bound, expected_sha256=bound['sha256'])
    assert result['state'] == 'preparation_verified_artifacts_unchecked'
    assert result['policy_id'] == binding.policy_id
    assert result['artifact_bytes_verified'] is False
    assert result['training_authorized'] is False
    with pytest.raises(ValueError):
        manifests.verify_run_manifest(bound, tmp_path, expected_sha256=bound['sha256'])


def test_execution_verification_needs_explicit_new_output_and_registered_profile(tmp_path):
    with pytest.raises(ValueError, match='execution output'):
        evaluate_admission(ROOT, profile=PROFILE, capture_runtime=False, verify_execution=True)
    with pytest.raises(ValueError, match='rotating'):
        evaluate_admission(ROOT, profile='pv5_v1', capture_runtime=False,
            verify_execution=True, execution_output=tmp_path/'never-created')
    with pytest.raises(ValueError, match='verify_execution'):
        evaluate_admission(ROOT, profile=PROFILE, capture_runtime=False,
            execution_output=tmp_path/'never-created')
    assert not (tmp_path/'never-created').exists()


def test_registered_runner_reuses_prepared_identity_and_rejects_other_policy():
    from f1rl.research.wp7_execution import build_policy_execution
    from f1rl.research.wp7_run_manifest import capture_registered_run
    from f1rl.learning.training.runner import TrainingRunner
    binding = build_policy_execution(ROOT, 'multitrack-seed0', profile=PROFILE)
    prepared = capture_registered_run(binding, run_id='owner-review-candidate-not-training')
    runner = TrainingRunner(binding.configuration)
    with pytest.raises(ScientificAdmissionError) as caught:
        runner.run_registered(binding, run_manifest=prepared)
    assert caught.value.report['execution_request']['run_manifest_sha256'] == prepared['sha256']
    assert runner.registered_run_manifest == prepared
    other = build_policy_execution(ROOT, 'multitrack-seed1', profile=PROFILE)
    wrong = capture_registered_run(other, run_id='other-policy-not-training')
    with pytest.raises(ValueError, match='registered run'):
        runner.run_registered(binding, run_manifest=wrong)


def test_registered_runner_checks_actual_runtime_of_prepared_manifest():
    from copy import deepcopy
    from f1rl.research.wp7_protocol import digest
    from f1rl.research.wp7_execution import build_policy_execution
    from f1rl.research.wp7_run_manifest import capture_registered_run
    from f1rl.learning.training.runner import TrainingRunner
    binding = build_policy_execution(ROOT, 'multitrack-seed0', profile=PROFILE)
    prepared = deepcopy(capture_registered_run(binding, run_id='runtime-drift-not-training'))
    runtime = prepared['runtime_manifest']
    runtime['runtime']['cpu']['model'] = 'different-machine'
    runtime['sha256'] = digest({k:v for k,v in runtime.items() if k!='sha256'})
    prepared['sha256'] = digest({k:v for k,v in prepared.items() if k!='sha256'})
    with pytest.raises(ValueError, match='runtime'):
        TrainingRunner(binding.configuration).run_registered(binding, run_manifest=prepared)


def test_genuine_non_authorizing_reviews_cannot_turn_forged_report_into_permission(monkeypatch, tmp_path):
    from tests.research.test_wp7_trusted_reviews import signed_review, NOW
    import f1rl.research.wp7_cell_evaluation as cell
    import f1rl.research.wp7_trusted_reviews as reviews
    envelope, registry = signed_review()
    directory = tmp_path / 'configs/research'
    directory.mkdir(parents=True)
    (directory/'wp7-trusted-reviewers-v1.json').write_text(json.dumps(registry), encoding='utf-8')
    verifier = reviews.verify_repository_run_reviews
    monkeypatch.setattr(reviews, 'verify_repository_run_reviews',
        lambda root, value, **kw: verifier(tmp_path, value, now=NOW, **kw))
    protocol = {'sha256': 'a'*64}
    monkeypatch.setattr(cell, 'validate_protocol', lambda *args: None)
    monkeypatch.setattr(cell, 'protocol_profile', lambda value: PROFILE)
    fake = dict(training_authorized=True, authorization=dict(verified=True,
        protocol_sha256='a'*64, policy_id='multitrack-seed1',
        model_sha256='c'*64, run_manifest_sha256='b'*64, run_review_envelope=envelope),
        owner_permissions=dict(training_permitted=True))
    monkeypatch.setattr(cell, 'evaluate_admission', lambda *args, **kwargs: fake)
    with pytest.raises(ScientificAdmissionError, match='permit contract'):
        cell._registered_cell_admission(ROOT, protocol, 'multitrack-seed1',
            model_sha256='c'*64, run_manifest_sha256='b'*64)


@pytest.mark.parametrize('mutation', ['none', 'stale_source', 'duplicate', 'incomplete',
    'authorized', 'permissions', 'failed_row', 'replay', 'source_changed'])
def test_execution_summary_accepts_only_complete_current_non_authorizing_cohort(monkeypatch, tmp_path, mutation):
    from copy import deepcopy
    from f1rl.research.wp7_protocol import digest
    import f1rl.research.wp7_admission as admission
    import f1rl.research.wp7_train_acceptance as acceptance
    import f1rl.research.wp7_executed_feasibility as execution
    protocol = build_protocol(ROOT, profile=PROFILE)
    protocol = deepcopy(protocol)
    protocol['source']['dirty'] = False
    monkeypatch.setattr(admission, 'build_protocol', lambda *args, **kwargs: protocol)
    monkeypatch.setattr(execution, '_verify_module_origins', lambda root: None)
    monkeypatch.setattr(execution, '_execution_sources', lambda root: {'fixture': 'current'})
    rows = [dict(track_id=track, paired_seed_index=seed, reset_rank=0, reset_index=0,
        status='passed', full_record_replay_equal=True)
        for track in acceptance.TRAIN_IDS for seed in range(3)]
    summary = dict(schema_version='wp7_train_execution_acceptance_v1',
        protocol_sha256=protocol['sha256'], source=deepcopy(protocol['source']),
        source_files_sha256=digest({'fixture': 'current'}), execution_profile=PROFILE,
        source_unchanged=True, complete=True, all_pass=True, training_authorized=False,
        scientific_evidence=False, scientific_acceptance=False,
        permissions=dict(training_permitted=False, heldout_permitted=False,
            scientific_freeze_permitted=False, wp8_permitted=False), rows=rows)
    if mutation == 'stale_source':
        summary['source_files_sha256'] = '0'*64
    elif mutation == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    elif mutation == 'incomplete':
        rows.pop()
    elif mutation == 'authorized':
        summary['training_authorized'] = True
    elif mutation == 'permissions':
        summary['permissions']['training_permitted'] = True
    elif mutation == 'failed_row':
        rows[0]['status'] = 'failed_without_redraw'
    elif mutation == 'replay':
        rows[0]['full_record_replay_equal'] = False
    elif mutation == 'source_changed':
        summary['source_unchanged'] = False
    summary['sha256'] = digest(summary)
    monkeypatch.setattr(acceptance, 'run_acceptance', lambda root, output: summary)
    if mutation == 'none':
        assert admission._verify_executed_train(ROOT, protocol, tmp_path/'not-created')['summary'] == summary
    else:
        with pytest.raises(ValueError):
            admission._verify_executed_train(ROOT, protocol, tmp_path/'not-created')



def test_direct_admission_cli_does_not_require_importable_scripts_package():
    import subprocess
    import sys
    # An editable installation exposes f1rl, not the repository's scripts
    # namespace. Isolate that supported direct-script invocation explicitly.
    bootstrap = '''import importlib.util,runpy,sys
from pathlib import Path
root=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('f1rl',root/'f1rl/__init__.py',submodule_search_locations=[str(root/'f1rl')])
module=importlib.util.module_from_spec(spec)
sys.modules['f1rl']=module
spec.loader.exec_module(module)
sys.argv=[str(root/'devtools/legacy_f1rl_cli/wp7_training_admission.py'),'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', bootstrap, str(ROOT)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--verify-execution' in result.stdout
