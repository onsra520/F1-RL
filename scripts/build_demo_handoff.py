"""Package a proven exploratory final model; never starts training."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args(argv)
    run, dest = args.run.resolve(), args.destination.resolve()
    summary = json.loads((run/'learning-summary.json').read_text())
    permit = json.loads((run/'permit.json').read_text())
    model = run/'trained-final.zip'
    checksum = sha256(model.read_bytes()).hexdigest()
    artifacts = [json.loads(line) for line in (run/'artifacts.jsonl').read_text().splitlines()]
    if (summary['learning_timesteps'] <= 0 or summary['optimizer_steps'] <= 0
            or summary['initial_weights_sha256'] == summary['final_weights_sha256']
            or not any(row['path'] == model.name and row['sha256'] == checksum for row in artifacts)):
        raise ValueError('genuine final learning evidence required')
    if dest.exists() or dest.with_suffix('.zip').exists():
        raise ValueError('handoff destination/archive must be fresh')
    dest.mkdir(parents=True)
    paths = [*ROOT.joinpath('f1rl').rglob('*.py'),
        ROOT/'scripts/run_inference.py', ROOT/'scripts/launch_exploratory_demo.py', ROOT/'START-DEMO.cmd',
        ROOT/'config.yml', ROOT/'assets/tracks/manifest.json',
        *ROOT.joinpath('assets/tracks/training').glob('*.json'),
        ROOT/'configs/research/wp7-rotating-owner-protocol-v3.json',
        ROOT/'configs/research/wp7-hard-route-outcome-v2.json',
        ROOT/'configs/recovery/wp7-owner-decisions-preregistered-v1.yml',
        ROOT/'configs/recovery/wp7-c2-candidate.yml',
        ROOT/'configs/experiments/ppo-generalization-v1.yml']
    for source in paths:
        target = dest/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ('models', 'evidence', 'config'):
        (dest/name).mkdir()
    shutil.copy2(model, dest/'models/ppo-straight-trained.zip')
    shutil.copy2(ROOT/'config.yml', dest/'config/demo-config.yml')
    shutil.copy2(run/'learning-summary.json', dest/'evidence/training-summary.json')
    rows = [json.loads(line) for line in (run/'evaluation.jsonl').read_text().splitlines()] if (run/'evaluation.jsonl').exists() else []
    save(dest/'evidence/evaluation-summary.json', {'episodes': rows,
        'status': json.loads((run/'state.json').read_text())['status']})
    save(dest/'evidence/model-manifest.json', {'model_sha256': checksum,
        'git_commit': permit['source'], 'config_sha256': permit['config_sha256'],
        'run_id': run.parent.name, 'seeds': permit['seeds'], 'limits': permit['limits'],
        'learning': summary, 'runtime': permit['runtime'], 'scientific_evidence': False})
    packages = permit['runtime']['packages']
    (dest/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name, version in packages.items()), encoding='utf-8')
    (dest/'README.md').write_text(f'''# F1-RL lecturer demo

Genuine exploratory PPO on TRAIN straight-sprint-v1. This is not scientific certification.
Training completion does not imply successful autonomous driving; inspect evidence/evaluation-summary.json.

Use Python {permit['runtime']['python']} (Windows CPU tested). From this extracted directory:

```text
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt
.venv\\Scripts\\python -B scripts/run_inference.py --output inference-result.json
```

Or launch START-DEMO.cmd after installation. This loads the saved model without training.
The console reports progress and all outcomes, including failed episodes. No GUI is required.
The supplied runtime source and TRAIN assets reconstruct the original observation/action/physics contracts.
No HELD-OUT track assets or execution authorization are included.
Check SHA256SUMS.txt for integrity; model provenance is in evidence/model-manifest.json.
''', encoding='utf-8')
    (dest/'SHA256SUMS.txt').write_text(''.join(
        f'{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(dest).as_posix()}\n'
        for path in sorted(dest.rglob('*')) if path.is_file()), encoding='utf-8')
    shutil.make_archive(str(dest), 'zip', dest.parent, dest.name)
    print(dest.with_suffix('.zip'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
