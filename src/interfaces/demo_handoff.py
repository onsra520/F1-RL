"""Package a proven exploratory final model; never starts training."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def demo_runtime_scripts(repository_root):
    """Minimum script package files required by the standalone demo handoff."""
    root = Path(repository_root)
    return tuple(root / relative for relative in (
        'scripts/__init__.py',
        'scripts/inference/__init__.py',
        'scripts/inference/run_policy.py',
        'scripts/training/__init__.py',
        'scripts/training/launch_demo.py',
        'scripts/training/exploratory_policy.py',
    ))


def main(repository_root, argv=None, *, torch_factory):
    ROOT = Path(repository_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--comparison-directory', type=Path)
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
    paired = None
    if args.comparison_directory:
        paired = json.loads((args.comparison_directory/'comparison.json').read_text())
        if (paired['models']['candidate']['model_sha256'] != checksum
                or paired['passes_prespecified_improvement'] is not True):
            raise ValueError('paired improvement evidence must match this model')
    torch_factory().set_num_threads(1)
    from src.rl.algorithms.ppo import PPOModel
    from src.exploratory.pilot import weights_sha256
    from src.validation.wp7_protocol import load_registration, prepare_scenarios
    loaded = PPOModel.load(model, device='cpu')
    if loaded.num_timesteps != summary['learning_timesteps'] or weights_sha256(loaded) != summary['final_weights_sha256']:
        raise ValueError('saved model differs from learning summary')
    factory = prepare_scenarios(ROOT, load_registration(ROOT, profile=permit['profile']), seed_index=0).scenarios[0].factory
    contracts = {'observation_shape': list(loaded._model.observation_space.shape),
        'observation_features': list(factory.observation_encoder.feature_names),
        'action_shape': list(loaded._model.action_space.shape),
        'action_order': ['steering', 'throttle', 'brake'],
        'action_low': loaded._model.action_space.low.tolist(),
        'action_high': loaded._model.action_space.high.tolist()}
    dest.mkdir(parents=True)
    paths = [*ROOT.joinpath('src').rglob('*.py'),
        *demo_runtime_scripts(ROOT), ROOT/'START-DEMO.cmd',
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
    if paired:
        for name in ('comparison.json', 'original-episodes.jsonl', 'candidate-episodes.jsonl'):
            shutil.copy2(args.comparison_directory/name, dest/'evidence'/('paired-'+name))
    shutil.copy2(model, dest/'models/ppo-straight-trained.zip')
    shutil.copy2(ROOT/'config.yml', dest/'config/demo-config.yml')
    shutil.copy2(run/'learning-summary.json', dest/'evidence/training-summary.json')
    rows = [json.loads(line) for line in (run/'evaluation.jsonl').read_text().splitlines()] if (run/'evaluation.jsonl').exists() else []
    trained_rows = [row for row in rows if row['policy'] == 'trained_final']
    progress = ', '.join(f"{row['progress']:.1%}" for row in trained_rows) or 'not evaluated'
    save(dest/'evidence/evaluation-summary.json', {'episodes': rows,
        'status': json.loads((run/'state.json').read_text())['status']})
    save(dest/'evidence/model-manifest.json', {'model_sha256': checksum,
        'git_commit': permit['source'], 'config_sha256': permit['config_sha256'],
        'run_id': run.parent.name, 'seeds': permit['seeds'], 'limits': permit['limits'],
        'learning': summary, 'runtime': permit['runtime'], 'contracts': contracts,
        'training_task': permit.get('demo_training', {'version': 'registered'}),
        'scientific_evidence': False})
    packages = permit['runtime']['packages']
    comparison_text = ''
    if paired:
        old, new = paired['models']['original'], paired['models']['candidate']
        comparison_text = f"""
Paired original-task evaluation (all eight seeds, including six reserved from tuning):
- Original: mean {old['all']['mean_progress']:.1%}, median {old['all']['median_progress']:.1%}, worst {old['all']['worst_progress']:.1%} progress; {old['all']['successes']}/8 finishes.
- Supplied model: mean {new['all']['mean_progress']:.1%}, median {new['all']['median_progress']:.1%}, worst {new['all']['worst_progress']:.1%} progress; {new['all']['successes']}/8 finishes.
- Reserved-seed mean progress: {old['reserved']['mean_progress']:.1%} to {new['reserved']['mean_progress']:.1%}.
- This model drives farther but still fails to finish. All eight outcomes, errors and comparison rules are in evidence/paired-comparison.json and paired episode logs.
"""
    (dest/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name, version in packages.items()), encoding='utf-8')
    (dest/'README.md').write_text(f'''# F1-RL lecturer demo

Genuine exploratory PPO on TRAIN straight-sprint-v1. This is not scientific certification.
Training completion does not imply successful autonomous driving; inspect evidence/evaluation-summary.json.
The supplied {summary['learning_timesteps']:,}-step model completed
{sum(row['success'] for row in trained_rows)}/{len(trained_rows)} evaluation episodes successfully.
Recorded progress: {progress}. All outcomes and failure reasons are preserved in the evaluation summary.
Training task: {permit.get('demo_training', {'version': 'registered'})}.
Evaluation and demonstration use the original registered task, without training-only penalties or corridor stops.
{comparison_text}

Use Python {permit['runtime']['python']} (Windows CPU tested). From this extracted directory:

```text
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt
.venv\\Scripts\\python -B -m scripts.inference.run_policy --output inference-result.json
```

Or launch START-DEMO.cmd after installation. This loads the saved model without training.
The console reports progress and all outcomes, including failed episodes. No GUI is required.
The supplied runtime source and TRAIN assets reconstruct the original observation/action/physics contracts.
Observations are 16 float32 features; exact order is in evidence/model-manifest.json.
Actions are [steering, throttle, brake], bounded by [-1, 0, 0] and [1, 1, 1].
No HELD-OUT track assets or execution authorization are included.
Check SHA256SUMS.txt for integrity; model provenance is in evidence/model-manifest.json.
''', encoding='utf-8')
    (dest/'SHA256SUMS.txt').write_text(''.join(
        f'{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(dest).as_posix()}\n'
        for path in sorted(dest.rglob('*')) if path.is_file()), encoding='utf-8')
    shutil.make_archive(str(dest), 'zip', dest.parent, dest.name)
    print(dest.with_suffix('.zip'))
    return 0
