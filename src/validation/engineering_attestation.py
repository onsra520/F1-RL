from hashlib import sha256
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys

SCHEMA = 'engineering_source_attestation_v1'
REQUIRED_TESTS = (
    'tests/research/test_wp3_roll_envelope_terminal_v4.py',
    'tests/integration/test_wp3_applied_crash_end_to_end.py',
    'tests/integration/test_episode_integration.py',
    'tests/integration/test_gym_environment_integration.py',
    'tests/integration/test_ppo_baseline_integration.py::test_real_env_is_sb3_compatible_and_accepts_predicted_action',
)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def _identity(root):
    if _git(root, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('dirty source tree cannot be attested')
    return _git(root, 'rev-parse', 'HEAD')


def _source_hashes(root):
    paths = _git(root, 'ls-files', '-z').split('\0')
    selected = sorted(p for p in paths if p and (p.startswith(('f1rl/', 'src/', 'tests/', 'scripts/', 'configs/')) or p == 'pyproject.toml'))
    if not selected:
        raise ValueError('source files are missing')
    return {p: sha256((root / p).read_bytes()).hexdigest() for p in selected}


def _execute(root):
    env = dict(os.environ)
    env.pop('PYTEST_ADDOPTS', None)
    env.pop('PYTEST_PLUGINS', None)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    command = [sys.executable, '-m', 'pytest', '-q', *REQUIRED_TESTS]
    collected = subprocess.run(command + ['--collect-only'], cwd=root, capture_output=True, text=True, env=env)
    count_match = re.search(r'(?m)^(\d+) tests? collected in ', collected.stdout)
    if collected.returncode or not count_match or int(count_match.group(1)) < len(REQUIRED_TESTS):
        raise ValueError('required test collection incomplete')
    nodes = [line for line in collected.stdout.splitlines() if '::' in line]
    if len(nodes) != int(count_match.group(1)) or any(not any(node.startswith(path.split('::')[0] + '::') for node in nodes) for path in REQUIRED_TESTS):
        raise ValueError('required test collection incomplete')
    tests = subprocess.run(command, cwd=root, capture_output=True, text=True, env=env)
    summary = re.search(r'(?m)^(\d+) passed in [\d.]+s', tests.stdout)
    if tests.returncode or not summary or int(summary.group(1)) != len(nodes):
        raise ValueError('current regression did not run every collected test successfully')
    smoke = subprocess.run([sys.executable, '-m', 'scripts.inference.smoke_frozen_policy'], cwd=root, capture_output=True, text=True, env=env)
    if smoke.returncode or 'checkpoint timesteps: 65536' not in smoke.stdout or "'failure_reason': 'step_limit'" not in smoke.stdout:
        raise ValueError('current model smoke did not complete as expected')
    return {'test_nodes': nodes, 'pytest_stdout': tests.stdout, 'pytest_stdout_sha256': sha256(tests.stdout.encode()).hexdigest(), 'pytest_summary': tests.stdout.splitlines()[-1], 'smoke_stdout': smoke.stdout, 'smoke_stdout_sha256': sha256(smoke.stdout.encode()).hexdigest(), 'smoke_terminal': 'step_limit'}


def create_current_attestation(root):
    root = Path(root).resolve(strict=True)
    commit = _identity(root)
    source = _source_hashes(root)
    manifest_path = root / 'docs/research/evidence/wp3-issue51-frozen-final-models-20261028/WP3-FROZEN-MODELS-SHA256-MANIFEST.json'
    manifest = json.loads(manifest_path.read_bytes())
    models = {}
    for row in manifest['models']:
        actual = sha256((manifest_path.parent / row['file']).read_bytes()).hexdigest()
        if actual != row['sha256'] or row['model_id'] in models:
            raise ValueError('checkpoint mismatch')
        models[row['model_id']] = actual
    if set(models) != {f'{candidate}-seed{seed}' for candidate in 'ABCD' for seed in range(3)}:
        raise ValueError('checkpoint roster mismatch')
    results = _execute(root)
    if _identity(root) != commit or _source_hashes(root) != source:
        raise ValueError('source changed during current tests')
    from .historical_source_records import verify_historical_records
    verify_historical_records(root, root / 'docs/research/evidence/wp3-historical-admission-inputs')
    refs = {}
    for name in ('WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json', 'WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json'):
        refs[name] = sha256((root / 'docs/research' / name).read_bytes()).hexdigest()
    return {'schema_version': SCHEMA, 'source_commit': commit, 'source_files': source, 'source_bundle_sha256': _digest(source), 'runtime_versions': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy', 'gymnasium', 'stable-baselines3', 'pytest', 'torch')}}, 'test_paths': list(REQUIRED_TESTS), 'new_execution': results, 'model_sha256': models, 'historical_record_sha256': refs, 'independent_scientific_approval': False, 'heldout_access': False, 'registered_scientific_evaluation_authorized': False, 'training_performed': False}


def verify_current_attestation(root, document):
    fields = {'schema_version', 'source_commit', 'source_files', 'source_bundle_sha256', 'runtime_versions', 'test_paths', 'new_execution', 'model_sha256', 'historical_record_sha256', 'independent_scientific_approval', 'heldout_access', 'registered_scientific_evaluation_authorized', 'training_performed'}
    if not isinstance(document, dict) or document.get('schema_version') != SCHEMA:
        raise ValueError('attestation schema mismatch')
    if set(document) != fields:
        raise ValueError('attestation fields mismatch')
    root = Path(root).resolve(strict=True)
    if document.get('source_commit') != _identity(root):
        raise ValueError('source commit mismatch')
    source = _source_hashes(root)
    if document.get('source_files') != source or document.get('source_bundle_sha256') != _digest(source):
        raise ValueError('source file or bundle mismatch')
    flags = ('independent_scientific_approval', 'heldout_access', 'registered_scientific_evaluation_authorized', 'training_performed')
    if any(document.get(key) is not False for key in flags):
        raise ValueError('scientific status mismatch')
    if document.get('test_paths') != list(REQUIRED_TESTS) or not isinstance(document.get('new_execution'), dict):
        raise ValueError('current execution receipt mismatch')
    for name in ('pytest_stdout_sha256', 'smoke_stdout_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(document['new_execution'].get(name, ''))):
            raise ValueError('current execution receipt mismatch')
    for label in ('pytest', 'smoke'):
        output = document['new_execution'].get(label + '_stdout')
        if not isinstance(output, str) or sha256(output.encode()).hexdigest() != document['new_execution'].get(label + '_stdout_sha256'):
            raise ValueError('current execution receipt mismatch')
    for old in ('WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt', 'WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt', 'WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json', 'WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json'):
        old_text = (root / 'docs/research' / old).read_text(encoding='utf8')
        if any(old_text.strip() in text.replace('\r\n', '\n') for text in (document['new_execution']['pytest_stdout'], document['new_execution']['smoke_stdout'])):
            raise ValueError('historical receipt cannot certify current source')
    lines = document['new_execution']['pytest_stdout'].splitlines()
    if not lines or lines[-1] != document['new_execution'].get('pytest_summary'):
        raise ValueError('current execution receipt mismatch')
    if document['new_execution'].get('smoke_terminal') != 'step_limit' or not re.search(r'^\d+ passed', document['new_execution'].get('pytest_summary', '')):
        raise ValueError('current execution receipt mismatch')
    current = create_current_attestation(root)
    normalize = lambda text: re.sub(r'in [\d.]+s(?: \(\d+:\d+:\d+\))?', 'in <duration>s', text.strip())
    if (document['new_execution'].get('test_nodes') != current['new_execution']['test_nodes'] or
        normalize(document['new_execution']['pytest_stdout']) != normalize(current['new_execution']['pytest_stdout']) or
        document['new_execution']['smoke_stdout'] != current['new_execution']['smoke_stdout']):
        raise ValueError('submitted execution receipt differs from fresh execution')
    for key in ('source_commit', 'source_files', 'source_bundle_sha256', 'runtime_versions', 'test_paths', 'model_sha256', 'historical_record_sha256', 'independent_scientific_approval', 'heldout_access', 'registered_scientific_evaluation_authorized', 'training_performed'):
        if document.get(key) != current[key]:
            raise ValueError(f'current attestation mismatch: {key}')
    return {'status': 'CURRENT_ENGINEERING_REGRESSION_REEXECUTED', 'source_commit': current['source_commit']}
