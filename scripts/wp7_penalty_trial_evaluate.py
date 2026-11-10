"""TRAIN-only paired evaluation for a completed penalty trial."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

from f1rl.learning.algorithms.ppo import PPOModel
from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS
from scripts.wp7_penalty_trial_authorization import verify_trial_authorization
from scripts.wp7_train5_v4_development import evaluate_one, validate_loaded_policy
from scripts.wp7_train5_v4_factory_preview import preview_registered_factories


def evaluate_trial(*, root, output, proposal, envelope, evidence_root):
    root, output = Path(root).resolve(strict=True), Path(output).resolve(strict=True)
    verify_trial_authorization(root, proposal, output, envelope, allow_existing=True)
    data = json.loads((output / 'training-result.json').read_text(encoding='utf8'))
    runs = data.get('runs', [])
    expected = {(kind, row['cohort'], row['replicate']) for kind in proposal['interventions'] for row in proposal['pairs']}
    if len(runs) != 12 or {(r['intervention'], r['cohort'], r['replicate']) for r in runs} != expected:
        raise ValueError('complete unique paired training runs required')
    policies, factories = {}, {}
    for r in runs:
        key = (r['intervention'], r['cohort'], r['replicate'])
        model_path = Path(r['model']).resolve(strict=True)
        if not model_path.is_relative_to(output) or sha256(model_path.read_bytes()).hexdigest() != r['sha256'] or r['steps'] != proposal['budget_per_run']['actual_final_steps']:
            raise ValueError('trained policy path, hash or budget differs')
        scenario_key = key[1:]
        if scenario_key not in factories:
            factories[scenario_key] = preview_registered_factories(root, Path(evidence_root), cohort=key[1], replicate=key[2])
        policies[key] = PPOModel.load(model_path, device='cpu')
        validate_loaded_policy(policies[key], factories[scenario_key])
    originals = {name: sha256((root / 'models' / name / 'model-final.zip').read_bytes()).hexdigest() for name in FROZEN_CHECKPOINTS}
    if originals != FROZEN_CHECKPOINTS: raise ValueError('source checkpoints changed')
    journal = output / 'paired-evaluation-journal.jsonl'
    with journal.open('x', encoding='utf8'): pass
    counts, done = defaultdict(Counter), 0
    for cohort, rep, track, condition, index, env_seed, policy_seed in proposal['evaluation_seed_rows']:
        scenario = next(s for s in factories[(cohort, rep)].scenarios if s.track_id == track)
        for kind in proposal['interventions']:
            identity = {'intervention': kind, 'cohort': cohort, 'replicate': rep, 'track_id': track,
                        'condition': condition, 'episode_index': index, 'environment_seed': env_seed, 'policy_seed': policy_seed}
            try:
                metrics = evaluate_one(factory=scenario.factory, policy=policies[(kind, cohort, rep)], track_id=track,
                                       condition=condition, environment_seed=env_seed, policy_seed=policy_seed)
            except Exception as exc:
                with journal.open('a', encoding='utf8') as f:
                    f.write(json.dumps({'event': 'failed', **identity, 'error': str(exc)}) + '\n'); f.flush(); os.fsync(f.fileno())
                raise
            with journal.open('a', encoding='utf8') as f:
                f.write(json.dumps({'event': 'completed', **identity, **metrics}, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
            counts[(kind, cohort, rep, track, condition)][metrics['outcome']] += 1
            done += 1
            if done % 120 == 0: print(f'Paired TRAIN evaluation {done}/{proposal["evaluation_episodes"]}', flush=True)
    if done != proposal['evaluation_episodes']: raise ValueError('evaluation count differs')
    cells = [{'intervention': k[0], 'cohort': k[1], 'replicate': k[2], 'track': k[3], 'condition': k[4],
              'n': sum(c.values()), 'outcomes': dict(c)} for k, c in counts.items()]
    outcomes = defaultdict(Counter)
    for (kind, cohort, rep, track, condition), c in counts.items():
        outcomes[(kind, cohort)].update(c)
    by_cohort = []
    for (kind, cohort), c in sorted(outcomes.items()):
        n = sum(c.values())
        by_cohort.append({'intervention': kind, 'cohort': cohort, 'episodes': n,
                          'finish_rate': c['finished'] / n,
                          'certified_physical_crash_rate': c['certified_physical_crash'] / n,
                          'outcomes': dict(c)})
    summary = {'schema': 'penalty_trial_development_v1', 'episodes': done, 'cells': cells, 'cohort_summary': by_cohort,
               'note': 'Internal TRAIN descriptive comparison only; no HELD-OUT or scientific admission.',
               'heldout_evaluated': False, 'frozen_checkpoint_sha256': originals}
    with (output / 'paired-evaluation-result.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    return summary
