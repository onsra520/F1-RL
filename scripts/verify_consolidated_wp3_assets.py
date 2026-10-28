"""Verify frozen WP3 models and FULL-06 output bytes against original manifests."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1] / 'docs/research/evidence'
MODEL = ROOT / 'wp3-issue51-frozen-final-models-20261028'
FULL = ROOT / 'wp3-issue51-full06-traindev-20261027'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    models = json.loads((MODEL / 'WP3-FROZEN-MODELS-SHA256-MANIFEST.json').read_bytes())
    output = json.loads((FULL / 'WP3-OUTPUT-SHA256-MANIFEST.json').read_bytes())
    receipt = json.loads((FULL / 'EVIDENCE-AUDIT-RECEIPT.json').read_bytes())
    run = json.loads((FULL / 'RUN-MANIFEST.json').read_bytes())
    canonical = json.dumps({k: v for k, v in run.items() if k != 'sha256'}, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')
    require(hashlib.sha256(canonical).hexdigest() == run['sha256'] == receipt['manifest_sha256'], 'run manifest mismatch')
    require(digest(FULL / 'WP3-OUTPUT-SHA256-MANIFEST.json') == receipt['run_output_hash_manifest_sha256'], 'output manifest mismatch')
    expected = {f'{candidate}-seed{seed}' for candidate in 'ABCD' for seed in range(3)}
    rows = models['models']
    require(len(rows) == 12 and {row['model_id'] for row in rows} == expected, 'model roster mismatch')
    for row in rows:
        path = MODEL / row['file']
        require(row['sha256'] == run['original_final_model_sha256'][row['model_id']], f'model binding mismatch: {path}')
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], f'model mismatch: {path}')
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, f'model ZIP CRC mismatch: {path}')
    require(len(output['files']) == 7 and len(output['cell_logs']) == 96, 'output counts mismatch')
    require({row['cell'] for row in output['cell_logs']} == set(range(96)), 'cell roster mismatch')
    for row in output['files']:
        path = FULL / row['name']
        require(digest(path) == row['sha256'], f'report mismatch: {path}')
    require(len(list((FULL / 'cells').glob('*.jsonl'))) == 96, 'cell file count mismatch')
    for row in output['cell_logs']:
        matches = list((FULL / 'cells').glob(f"{row['cell']:03d}-*.jsonl"))
        require(len(matches) == 1 and digest(matches[0]) == row['name'], f'cell mismatch: {row}')
    print('PASS: 12 models, 7 reports and 96 cell logs')


if __name__ == '__main__':
    main()
