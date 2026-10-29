import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'scripts.evidence.verify_historical_records'
HISTORY = ROOT / 'docs/research/evidence/wp3-historical-admission-inputs'


def test_historical_record_verification_is_independent_of_current_source():
    result = subprocess.run([sys.executable, '-m', MODULE, '--history', str(HISTORY)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'HISTORICAL_RECORD_INTEGRITY_VERIFIED' in result.stdout
    assert 'ORIGINAL_SOURCE_RECONSTRUCTED' not in result.stdout
import shutil
import pytest
from src.validation.historical_source_records import verify_historical_records


@pytest.mark.parametrize('relative', [
    'docs/research/WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json',
    'docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json',
    'docs/research/WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json',
    'docs/research/WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt',
    'docs/research/evidence/wp3-historical-admission-inputs/evaluation/evaluation-roster.json',
    'docs/research/evidence/wp3-historical-admission-inputs/WP3-EVAL-SOURCE-PROVENANCE.json',
    'docs/research/evidence/wp3-issue51-frozen-final-models-20261028/models/A-seed0/model-final.zip',
])
def test_tampered_historical_input_is_rejected(tmp_path, relative):
    for path in (ROOT / 'docs/research').glob('WP3-*20261024.*'):
        if path.is_file():
            dest = tmp_path / 'docs/research' / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    for path in (ROOT / 'docs/research/evidence/wp3-historical-admission-inputs').rglob('*.json'):
        dest = tmp_path / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for path in (ROOT / 'docs/research/evidence/wp3-issue51-frozen-final-models-20261028/models').glob('*/model-final.zip'):
        dest = tmp_path / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    target = tmp_path / relative
    target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises(ValueError, match='mismatch'):
        verify_historical_records(tmp_path, tmp_path / HISTORY.relative_to(ROOT))

@pytest.mark.parametrize('module', [
    'scripts.evidence.verify_technical_acceptance',
    'scripts.evidence.verify_crash_terminal_v1',
])
def test_grouped_verify_entrypoints_report_historical_integrity(module):
    import os
    env = dict(os.environ, F1RL_WP3_EVIDENCE=str(HISTORY))
    result = subprocess.run([sys.executable, '-m', module], cwd=ROOT, text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'HISTORICAL_RECORD_INTEGRITY_VERIFIED' in result.stdout

def test_forged_historical_source_bundle_is_rejected(tmp_path):
    import json
    import hashlib
    from src.validation.historical_source_records import verify_historical_records
    name = 'WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json'
    original = ROOT / 'docs/research' / name
    dest = tmp_path / 'docs/research' / name
    dest.parent.mkdir(parents=True)
    record = json.loads(original.read_bytes())
    record['physics_source_bundle_sha256'] = '0' * 64
    body = {key: value for key, value in record.items() if key != 'record_sha256'}
    record['record_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    dest.write_text(json.dumps(record), encoding='utf8')
    with pytest.raises(ValueError, match='historical record SHA-256 mismatch'):
        verify_historical_records(tmp_path, HISTORY)


def test_current_attestation_cannot_replace_historical_record(tmp_path):
    import json
    path = tmp_path / 'docs/research/WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'schema_version': 'engineering_source_attestation_v1'}), encoding='utf8')
    with pytest.raises(ValueError, match='historical record SHA-256 mismatch'):
        verify_historical_records(tmp_path, HISTORY)


def test_historical_allowlist_does_not_accept_migrated_physics():
    from src.rl.freeze_readiness import verify_training_source_diff
    with pytest.raises(ValueError, match='training/simulator source changed'):
        verify_training_source_diff(('src/simulation/spatial_contact_dynamics.py',))


def test_legacy_receipt_builder_rejects_current_source_identity():
    from src.validation.wp3_v4_technical_acceptance import source_bound_acceptance
    import os
    history = Path(os.environ['F1RL_WP3_EVIDENCE'])
    replay = ROOT / 'docs/research/WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json'
    receipt = ROOT / 'docs/research/WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt'
    with pytest.raises(ValueError, match='historical source identity'):
        source_bound_acceptance(ROOT, history, replay, test_receipt_file=receipt)


def test_legacy_crash_builder_rejects_current_source_identity():
    from src.validation.wp3_crash_contract_registration import assemble_registration
    import os
    history = Path(os.environ['F1RL_WP3_EVIDENCE'])
    replay = ROOT / 'docs/research/WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json'
    receipt = ROOT / 'docs/research/WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt'
    with pytest.raises(ValueError, match='historical source identity'):
        assemble_registration(ROOT, history, replay, receipt)


@pytest.mark.parametrize("kind", ["physics", "crash"])
def test_legacy_builder_never_reissues_when_current_bytes_match_record(monkeypatch, kind):
    import os
    from src.validation import wp3_v4_technical_acceptance as physics
    from src.validation import wp3_crash_contract_registration as crash
    history = Path(os.environ["F1RL_WP3_EVIDENCE"])
    if kind == "physics":
        module = physics
        record = json.loads((ROOT / "docs/research/WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json").read_bytes())
        bindings = record["source_bindings_sha256"]
        replay = ROOT / "docs/research/WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json"
        receipt = ROOT / "docs/research/WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt"
    else:
        module = crash
        record = json.loads((ROOT / "docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json").read_bytes())
        bindings = record["source_files_sha256"]
        replay = ROOT / "docs/research/WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json"
        receipt = ROOT / "docs/research/WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt"
    mapping = (("f1rl/learning/training/", "src/training/"), ("f1rl/learning/evaluation/", "src/evaluation/"), ("f1rl/learning/", "src/rl/"), ("f1rl/sandbox/", "src/simulation/"), ("f1rl/runtime/", "src/core/"))
    current = {}
    for historical, digest in bindings.items():
        for before, after in mapping:
            if historical.startswith(before):
                current[(ROOT / (after + historical[len(before):])).resolve()] = digest
                break
        else:
            raise AssertionError(historical)
    actual_digest = module.digest_file
    def matching_digest(path):
        resolved = Path(path).resolve()
        return current[resolved] if resolved in current else actual_digest(path)
    monkeypatch.setattr(module, "digest_file", matching_digest)
    with pytest.raises(ValueError, match="historical source identity"):
        if kind == "physics":
            module.source_bound_acceptance(ROOT, history, replay, test_receipt_file=receipt)
        else:
            module.assemble_registration(ROOT, history, replay, receipt)
