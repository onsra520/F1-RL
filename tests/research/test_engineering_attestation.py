import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'scripts.evidence.engineering_attestation'


def test_refuses_to_attest_a_dirty_source_tree(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / 'source.py').write_text('x = 1\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'source.py'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial'], check=True)
    (repo / 'source.py').write_text('x = 2\n')
    result = subprocess.run([sys.executable, '-m', MODULE, '--root', str(repo), '--output', str(tmp_path / 'attestation.json')], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'dirty' in result.stderr.lower()
    assert not (tmp_path / 'attestation.json').exists()


def test_historical_receipt_cannot_be_current_attestation():
    import json
    from src.validation.engineering_attestation import verify_current_attestation
    historical = json.loads((ROOT / 'docs/research/WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json').read_bytes())
    import pytest
    with pytest.raises(ValueError, match='attestation schema'):
        verify_current_attestation(ROOT, historical)


def make_clean_repo(tmp_path):
    import hashlib
    import json
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    source = repo / 'src/module.py'
    source.parent.mkdir()
    source.write_text('x = 1\n', encoding='utf8')
    import shutil
    for name in ('WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt', 'WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt', 'WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json', 'WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json'):
        target = repo / 'docs/research' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'docs/research' / name, target)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    source_map = {'src/module.py': hashlib.sha256(source.read_bytes()).hexdigest()}
    bundle = hashlib.sha256(json.dumps(source_map, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    doc = {'schema_version': 'engineering_source_attestation_v1', 'source_commit': commit, 'source_files': source_map, 'source_bundle_sha256': bundle, 'independent_scientific_approval': False, 'heldout_access': False, 'registered_scientific_evaluation_authorized': False, 'training_performed': False, 'runtime_versions': {}, 'test_paths': [], 'new_execution': {}, 'model_sha256': {}, 'historical_record_sha256': {}}
    return repo, doc


def test_rejects_current_source_path_and_hash_mismatch(tmp_path):
    import pytest
    from src.validation.engineering_attestation import verify_current_attestation
    repo, doc = make_clean_repo(tmp_path)
    doc['source_files'] = {'src/other.py': '0' * 64}
    with pytest.raises(ValueError, match='source file or bundle mismatch'):
        verify_current_attestation(repo, doc)


def test_rejects_wrong_commit_and_dirty_tree(tmp_path):
    import pytest
    from src.validation.engineering_attestation import verify_current_attestation
    repo, doc = make_clean_repo(tmp_path)
    doc['source_commit'] = '0' * 40
    with pytest.raises(ValueError, match='source commit mismatch'):
        verify_current_attestation(repo, doc)
    (repo / 'src/module.py').write_text('changed\n')
    with pytest.raises(ValueError, match='dirty'):
        verify_current_attestation(repo, doc)


def test_rejects_scientific_approval_and_heldout(tmp_path):
    import pytest
    from src.validation.engineering_attestation import verify_current_attestation
    repo, doc = make_clean_repo(tmp_path)
    for field in ('independent_scientific_approval', 'heldout_access', 'registered_scientific_evaluation_authorized', 'training_performed'):
        bad = dict(doc, **{field: True})
        with pytest.raises(ValueError, match='scientific status mismatch'):
            verify_current_attestation(repo, bad)


def test_historical_receipt_and_replay_cannot_be_used_for_current_execution(tmp_path):
    import hashlib
    import pytest
    from src.validation.engineering_attestation import REQUIRED_TESTS, verify_current_attestation
    repo, doc = make_clean_repo(tmp_path)
    old_receipt = (repo / 'docs/research/WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt').read_text(encoding='utf8')
    old_replay = (repo / 'docs/research/WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json').read_text(encoding='utf8')
    doc['test_paths'] = list(REQUIRED_TESTS)
    for test_output, smoke_output in ((old_receipt, 'fresh smoke'), ('prefix\n' + old_receipt + '\nsuffix', 'fresh smoke'), ('fresh pytest', old_replay), ('fresh pytest', 'prefix\n' + old_replay + '\nsuffix')):
        doc['new_execution'] = {'pytest_stdout': test_output, 'pytest_stdout_sha256': hashlib.sha256(test_output.encode()).hexdigest(), 'pytest_summary': '300 passed in 3s', 'smoke_stdout': smoke_output, 'smoke_stdout_sha256': hashlib.sha256(smoke_output.encode()).hexdigest(), 'smoke_terminal': 'step_limit'}
        with pytest.raises(ValueError, match='historical receipt'):
            verify_current_attestation(repo, doc)


def test_rejects_incorrect_hash_for_current_source(tmp_path):
    import pytest
    from src.validation.engineering_attestation import verify_current_attestation
    repo, doc = make_clean_repo(tmp_path)
    doc['source_files']['src/module.py'] = 'f' * 64
    with pytest.raises(ValueError, match='source file or bundle mismatch'):
        verify_current_attestation(repo, doc)


def test_attester_rejects_a_different_clean_checkout(tmp_path):
    repo, _ = make_clean_repo(tmp_path)
    result = subprocess.run([sys.executable, '-m', MODULE, '--root', str(repo), '--output', str(tmp_path / 'attestation.json')], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'target checkout' in result.stderr
    assert not (tmp_path / 'attestation.json').exists()
