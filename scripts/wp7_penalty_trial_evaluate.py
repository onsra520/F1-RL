"""TRAIN-only paired evaluation for a completed penalty trial (sequential, sharded and merge paths)."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path

from f1rl.learning.algorithms.ppo import PPOModel
from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS
from scripts.wp7_penalty_trial_authorization import verify_trial_scope
from scripts.wp7_train5_v4_development import evaluate_one, validate_loaded_policy
from scripts.wp7_train5_v4_factory_preview import preview_registered_factories


def _originals(root):
    return {name: sha256((root / 'models' / name / 'model-final.zip').read_bytes()).hexdigest() for name in FROZEN_CHECKPOINTS}


def _load_runs(root, output, proposal, evidence_root):
    """Load and check all 12 trained policies and their factories (the same checks for every path)."""
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
    return policies, factories


def _record(kind, cohort, rep, track, condition, index, env_seed, policy_seed):
    return {'intervention': kind, 'cohort': cohort, 'replicate': rep, 'track_id': track,
            'condition': condition, 'episode_index': index, 'environment_seed': env_seed, 'policy_seed': policy_seed}


def _summarize(*, proposal, counts, done, originals):
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
    return {'schema': 'penalty_trial_development_v1', 'episodes': done, 'cells': cells, 'cohort_summary': by_cohort,
            'note': 'Internal TRAIN descriptive comparison only; no HELD-OUT or scientific admission.',
            'heldout_evaluated': False, 'frozen_checkpoint_sha256': originals}


def evaluate_trial(*, root, output, proposal, evidence_root):
    """Sequential reference path: one episode after another, journal written in seed-row order."""
    root = Path(root).resolve(strict=True)
    output = Path(output).resolve(strict=True)
    verify_trial_scope(root, proposal, output, allow_existing=True)
    policies, factories = _load_runs(root, output, proposal, evidence_root)
    originals = _originals(root)
    if originals != FROZEN_CHECKPOINTS: raise ValueError('source checkpoints changed')
    journal = output / 'paired-evaluation-journal.jsonl'
    with journal.open('x', encoding='utf8'): pass
    counts, done = defaultdict(Counter), 0
    for cohort, rep, track, condition, index, env_seed, policy_seed in proposal['evaluation_seed_rows']:
        scenario = next(s for s in factories[(cohort, rep)].scenarios if s.track_id == track)
        for kind in proposal['interventions']:
            identity = _record(kind, cohort, rep, track, condition, index, env_seed, policy_seed)
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
    summary = _summarize(proposal=proposal, counts=counts, done=done, originals=originals)
    with (output / 'paired-evaluation-result.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    return summary


def shard_rows(proposal, cohort, replicate, track):
    return [row for row in proposal['evaluation_seed_rows'] if row[0] == cohort and row[1] == replicate and row[2] == track]


def shard_name(cohort, replicate, track):
    return f'{cohort}-{replicate}-{track}.jsonl'


def evaluate_shard(*, root, output, proposal, evidence_root, cohort, replicate, track):
    """One (cohort, replicate, track) shard: its episodes for both interventions, written line by line."""
    root = Path(root).resolve(strict=True)
    output = Path(output).resolve(strict=True)
    verify_trial_scope(root, proposal, output, allow_existing=True)
    policies, factories = _load_runs(root, output, proposal, evidence_root)
    if _originals(root) != FROZEN_CHECKPOINTS: raise ValueError('source checkpoints changed')
    shards = output / 'eval-shards'
    shards.mkdir(exist_ok=True)
    journal = shards / shard_name(cohort, replicate, track)
    scenario = next(s for s in factories[(cohort, replicate)].scenarios if s.track_id == track)
    rows = shard_rows(proposal, cohort, replicate, track)
    with journal.open('x', encoding='utf8') as f:
        for _, _, _, condition, index, env_seed, policy_seed in rows:
            for kind in proposal['interventions']:
                identity = _record(kind, cohort, replicate, track, condition, index, env_seed, policy_seed)
                metrics = evaluate_one(factory=scenario.factory, policy=policies[(kind, cohort, replicate)], track_id=track,
                                       condition=condition, environment_seed=env_seed, policy_seed=policy_seed)
                f.write(json.dumps({'event': 'completed', **identity, **metrics}, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
    return len(rows) * len(proposal['interventions'])


def merge_shards(*, root, output, proposal):
    """Merge all shards into the same journal and summary the sequential path writes."""
    root = Path(root).resolve(strict=True)
    output = Path(output).resolve(strict=True)
    verify_trial_scope(root, proposal, output, allow_existing=True)
    if _originals(root) != FROZEN_CHECKPOINTS: raise ValueError('source checkpoints changed')
    by_key = {}
    for row in proposal['evaluation_seed_rows']:
        cohort, rep, track = row[0], row[1], row[2]
        path = output / 'eval-shards' / shard_name(cohort, rep, track)
        if not path.is_file():
            raise ValueError('missing evaluation shard')
    for path in sorted((output / 'eval-shards').glob('*.jsonl')):
        for line in path.read_text(encoding='utf8').splitlines():
            record = json.loads(line)
            key = (record['intervention'], record['cohort'], record['replicate'], record['track_id'],
                   record['condition'], record['episode_index'])
            if key in by_key:
                raise ValueError('duplicate evaluation record')
            by_key[key] = line
    journal = output / 'paired-evaluation-journal.jsonl'
    counts, done = defaultdict(Counter), 0
    with journal.open('x', encoding='utf8') as f:
        for cohort, rep, track, condition, index, env_seed, policy_seed in proposal['evaluation_seed_rows']:
            for kind in proposal['interventions']:
                key = (kind, cohort, rep, track, condition, index)
                if key not in by_key:
                    raise ValueError('missing evaluation record')
                line = by_key.pop(key)
                f.write(line + '\n')
                record = json.loads(line)
                counts[(kind, cohort, rep, track, condition)][record['outcome']] += 1
                done += 1
    if by_key or done != proposal['evaluation_episodes']:
        raise ValueError('evaluation count differs')
    summary = _summarize(proposal=proposal, counts=counts, done=done, originals=_originals(root))
    with (output / 'paired-evaluation-result.json').open('x', encoding='utf8') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    return summary
