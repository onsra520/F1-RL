"""One final paired TRAIN-only evaluation on prespecified tuning/reserved seeds."""
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (10000, 10001, 11000, 11001, 11002, 11003, 11004, 11005)


def metrics(rows):
    return {'episodes':len(rows),'successes':sum(row['success'] for row in rows),
        'all_valid':all(row['valid_sample'] and row['error'] is None for row in rows),
        'mean_progress':mean(row['progress'] for row in rows),
        'median_progress':median(row['progress'] for row in rows),
        'worst_progress':min(row['progress'] for row in rows),
        'collisions':sum(row['collisions'] for row in rows),
        'checkpoints_passed':sum(len(row['completed_checkpoint_ids']) for row in rows),
        'terminal_reasons':dict(Counter(row['outcome'].get('reason','unknown') for row in rows))}


def improved(old, candidate):
    return (candidate['all_valid'] and candidate['successes'] >= old['successes']
        and candidate['mean_progress'] >= old['mean_progress']+.05
        and candidate['median_progress'] > old['median_progress']
        and candidate['worst_progress'] >= old['worst_progress'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from src.learning.algorithms.ppo import PPOModel
    from src.research.wp7_protocol import load_registration, prepare_scenarios
    from src.exploratory.pilot import evaluate_episode
    factory = prepare_scenarios(ROOT,load_registration(ROOT,profile='spatial_rotating_owner_v3'),seed_index=0).scenarios[0].factory
    args.output.mkdir(parents=True, exist_ok=False)

    class InferenceSession:
        permit = {'seeds':{'evaluation':list(SEEDS)}}

        def create_env(self, *, learning, episode):
            if learning:
                raise ValueError('comparison cannot train')
            return replace(factory,reset_config=replace(factory.reset_config,seed=SEEDS[episode])).create()

    results = {}
    for label, path in (('original',args.old),('candidate',args.candidate)):
        model = PPOModel.load(path,device='cpu')
        rows = []
        with (args.output/f'{label}-episodes.jsonl').open('x') as stream:
            for episode in range(len(SEEDS)):
                row = evaluate_episode(InferenceSession(), model, label, episode)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                rows.append(row)
                print(json.dumps({'model':label,'seed':SEEDS[episode],
                    'success':row['success'],'progress':row['progress'],'error':row['error']}),flush=True)
        results[label] = {'model_sha256':sha256(path.read_bytes()).hexdigest(),
            'all':metrics(rows),'tuning':metrics(rows[:2]),'reserved':metrics(rows[2:])}
    approved = improved(results['original']['reserved'],results['candidate']['reserved'])
    summary = {'track':'straight-sprint-v1','seed_roster':list(SEEDS),
        'reserved_seeds':list(SEEDS[2:]),'models':results,'passes_prespecified_improvement':approved,
        'training_executed':False,'scientific_evidence':False}
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if approved else 2


if __name__ == '__main__':
    raise SystemExit(main())
