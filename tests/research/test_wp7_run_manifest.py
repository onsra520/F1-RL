from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest

from src.research.wp7_execution import build_policy_execution
from src.research.wp7_protocol import digest
from src.research.wp7_run_manifest import capture_registered_run, bind_run_artifacts, verify_run_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_binding_runtime_and_checkpoint_bytes_verify_on_independent_restore(tmp_path, monkeypatch):
    binding = build_policy_execution(ROOT, 'multitrack-seed1')
    start = capture_registered_run(binding, run_id='fixture-not-a-training-run')
    assert start['state'] == 'prepared_not_started'
    assert start['training_executed'] is False
    assert start['runtime_manifest']['runtime']['os'] in ('Windows', 'Linux', 'Darwin')
    assert start['receipt']['seed_bundle'] == dict(ppo=1, sampling=1, perturbation=1001)
    original, restore = tmp_path/'original', tmp_path/'restore'
    original.mkdir()
    restore.mkdir()
    # Opaque byte fixtures: no deserialization or claim of learned timesteps.
    (original/'checkpoint.zip').write_bytes(b'opaque-checkpoint-fixture')
    (restore/'checkpoint.zip').write_bytes((original/'checkpoint.zip').read_bytes())
    sealed = bind_run_artifacts(start, original)
    import src.research.wp7_protocol as protocol_module
    def forbidden_recapture():
        raise AssertionError('restore must not substitute verifier-host runtime')
    monkeypatch.setattr(protocol_module, '_runtime', forbidden_recapture)
    result = verify_run_manifest(sealed, restore, expected_sha256=sealed['sha256'])
    assert result['state'] == 'artifact_bytes_verified'
    assert result['training_authorized'] is False
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(sealed), encoding='utf-8')
    completed = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/wp7_verify_run_evidence.py'),
        '--manifest',str(manifest_path),'--artifacts',str(restore),'--expected-sha256',sealed['sha256']],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(completed.stdout)['training_authorized'] is False
    (restore/'checkpoint.zip').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        verify_run_manifest(sealed, restore, expected_sha256=sealed['sha256'])


def test_manifest_requires_external_digest_and_rejects_rehashed_runtime_or_receipt(tmp_path):
    start = capture_registered_run(build_policy_execution(ROOT, 'multitrack-seed0'), run_id='fixture')
    for field in ('runtime_manifest', 'receipt'):
        changed = deepcopy(start)
        changed[field]['protocol_sha256'] = 'a'*64
        changed['sha256'] = digest({k:v for k,v in changed.items() if k != 'sha256'})
        with pytest.raises(ValueError):
            verify_run_manifest(changed, None, expected_sha256=changed['sha256'])
    with pytest.raises(ValueError):
        verify_run_manifest(start, None, expected_sha256='a'*64)
    with pytest.raises(ValueError):
        bind_run_artifacts({**start, 'training_executed': True}, tmp_path)
    for field in ('physics_sha256', 'reset_sha256', 'seed_bundle', 'vector_environment',
                  'action_contract', 'race_sha256_by_track', 'checkpoint_plan'):
        changed = deepcopy(start)
        changed['receipt'][field] = 'contradictory identity'
        changed['receipt']['sha256'] = digest({k:v for k,v in changed['receipt'].items() if k != 'sha256'})
        changed['sha256'] = digest({k:v for k,v in changed.items() if k != 'sha256'})
        with pytest.raises(ValueError, match='receipt and recorded execution protocol'):
            verify_run_manifest(changed, None, expected_sha256=changed['sha256'])


def test_runner_captures_preexecution_manifest_but_denies_research_before_artifacts():
    from src.learning.training.runner import TrainingRunner
    from src.research.wp7_admission import ScientificAdmissionError
    binding = build_policy_execution(ROOT, 'multitrack-seed2')
    runner = TrainingRunner(binding.configuration)
    with pytest.raises(ScientificAdmissionError):
        runner.run_registered(binding)
    record = runner.registered_run_manifest
    assert record['state'] == 'prepared_not_started'
    assert not record['training_executed']
    record['receipt']['training_authorized'] = True
    assert runner.registered_run_manifest['receipt']['training_authorized'] is False


@pytest.mark.parametrize('payload', ('{"sha256":"a","sha256":"b"}', '{"value":NaN}'))
def test_cli_rejects_ambiguous_or_nonfinite_json(tmp_path, payload):
    from scripts.wp7_verify_run_evidence import read_json
    path = tmp_path/'invalid.json'
    path.write_text(payload)
    with pytest.raises(ValueError):
        read_json(path)
