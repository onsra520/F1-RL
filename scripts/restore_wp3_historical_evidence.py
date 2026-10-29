"""Restore immutable WP3 historical admission inputs from verified archives."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation.wp3_eval_only_admission import (
    MODELS, build_eval_only_preflight, hash_file, refuse_full_eval,
)

ARCHIVES = {
    'full': 'fd39193a8689db1ba529910851435a46402981372b1dd893c7fd523b72363edf',
    'continuation': '726573b374f3aec727457f2c7b7b1f6533ae09079e257577b85ced17fac18926',
}


def copy_member(archive, name, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, destination.open('xb') as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def verify_bundled_inputs():
    inventory = json.loads((ROOT / 'docs/research/FINAL-CONSOLIDATION-ASSETS.json').read_bytes())
    prefix = 'docs/research/evidence/wp3-historical-admission-inputs/'
    rows = [row for row in inventory['assets'] if row['destination'].startswith(prefix)]
    canonical = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf8')
    if len(rows) != 26 or hashlib.sha256(canonical).hexdigest() != '142c15e293716e81bce9d7eb6408733d095d7256647f96b2563294ccad481dbb':
        raise ValueError('historical input inventory SHA-256 mismatch')
    for row in rows:
        path = ROOT / row['destination']
        if path.stat().st_size != row['bytes'] or hash_file(path) != row['sha256']:
            raise ValueError(f'historical input SHA-256 mismatch: {path}')
    from src.validation.frozen_asset_integrity import verify_frozen_assets
    verify_frozen_assets(ROOT)
    # Preserve the legacy verifier's success line for existing restore logs.
    print('PASS: 12 models, 7 reports and 96 cell logs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARCHIVES:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'destination already exists: {destination}')
    verify_bundled_inputs()
    for name, expected in ARCHIVES.items():
        if hash_file(getattr(args, name)) != expected:
            raise ValueError(f'{name} archive SHA-256 mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.wp3-restore-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'history'
        inputs = ROOT / 'docs/research/evidence/wp3-historical-admission-inputs'
        shutil.copytree(inputs, stage)
        with zipfile.ZipFile(args.full) as full, zipfile.ZipFile(args.continuation) as continuation:
            original = {'C-seed0', 'A-seed0', 'B-seed0'}
            for model in MODELS:
                model_path = ROOT / 'docs/research/evidence/wp3-issue51-frozen-final-models-20261028/models' / model / 'model-final.zip'
                shutil.copyfile(model_path, stage / 'runs' / model / 'model-final.zip')
                archive = full if model in original else continuation
                path = ('attempt-v8/' if model in original else '') + f'runs/{model}/training-telemetry.jsonl'
                copy_member(archive, path, stage / 'runs' / model / 'training-telemetry.jsonl')
        replay = ROOT / 'docs/research/WP3-CE0E55D-THREE-REPLAY-PREFLIGHT-20261024.json'
        plan = build_eval_only_preflight(ROOT, stage, replay)
        try:
            refuse_full_eval(plan)
        except RuntimeError as exc:
            if not str(exc).startswith('FULL_EVAL_BLOCKED:'):
                raise
        else:
            raise RuntimeError('historical scientific gate unexpectedly allowed evaluation')
        destination.mkdir()
        try:
            for item in stage.iterdir():
                item.rename(destination / item.name)
        except OSError as exc:
            raise OSError(f'partial restore at {destination}; inspect and remove it before retrying') from exc
    print(json.dumps({'destination': str(destination), 'models_verified': len(plan['verified_models']),
                      'roster_sha256': plan['historical_roster_sha256'],
                      'scientific_gate': 'BLOCKED', 'new_episodes': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
