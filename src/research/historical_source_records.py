from hashlib import sha256
import json
from pathlib import Path

RECORDS = (
    ('WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json', '4ad0b1082ce7476a9c8b11d777009ff83f3df5a111d3508343b0fd5311c0a1cc', 'wp3_v4_source_bound_technical_acceptance_v1', 'source_bindings_sha256', 'physics_source_bundle_sha256', 'WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json', 'replay_file_sha256', 'WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt', 'engineering_pytest_receipt_sha256', 'original_final_weight_sha256', 'original_roster_file_sha256', 'original_provenance_file_sha256'),
    ('WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json', '086f27aef269d2cb999f121b008b3e168168fce274907a513585dfbf0641729d', 'wp3_crash_terminal_v1_technical_registry', 'source_files_sha256', 'source_bundle_sha256', 'WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json', 'replay_source_file_sha256', 'WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt', 'engineering_regression_receipt_sha256', 'verified_original_model_final_sha256', 'paired_roster_file_sha256', 'training_provenance_sha256'),
)


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate historical field')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def verify_historical_records(root, history):
    root, history = Path(root), Path(history)
    research = root / 'docs/research'
    models = root / 'docs/research/evidence/wp3-issue51-frozen-final-models-20261028/models'
    expected_models = {f'{candidate}-seed{seed}' for candidate in 'ABCD' for seed in range(3)}
    for name, record_hash, schema, sources_key, bundle_key, replay, replay_key, receipt, receipt_key, model_key, roster_key, provenance_key in RECORDS:
        path = research / name
        if _hash(path) != record_hash:
            raise ValueError(f'historical record SHA-256 mismatch: {name}')
        record = _read(path)
        if record.get('schema_version') != schema or _canonical_hash({k: v for k, v in record.items() if k != 'record_sha256'}) != record.get('record_sha256'):
            raise ValueError(f'historical record contract mismatch: {name}')
        sources = record[sources_key]
        if not sources or not all(k.startswith('f1rl/') and isinstance(v, str) and len(v) == 64 for k, v in sources.items()) or _canonical_hash(sources) != record[bundle_key]:
            raise ValueError(f'historical source bundle mismatch: {name}')
        for file, key in ((research / replay, replay_key), (research / receipt, receipt_key), (history / 'evaluation/evaluation-roster.json', roster_key), (history / 'WP3-EVAL-SOURCE-PROVENANCE.json', provenance_key)):
            if _hash(file) != record[key]:
                raise ValueError(f'historical reference SHA-256 mismatch: {file}')
        hashes = record[model_key]
        if set(hashes) != expected_models:
            raise ValueError('historical model roster mismatch')
        for model, expected in hashes.items():
            if _hash(models / model / 'model-final.zip') != expected:
                raise ValueError(f'historical model SHA-256 mismatch: {model}')
    return {'status': 'HISTORICAL_RECORD_INTEGRITY_VERIFIED', 'original_source_reconstructed': False, 'records': len(RECORDS)}
