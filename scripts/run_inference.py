"""Run the saved PPO policy on TRAIN Straight, without creating a training session."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    preferred = ROOT/'models/ppo-straight-improved-16384.zip'
    parser.add_argument('--model', type=Path,
        default=preferred if preferred.is_file() else ROOT/'models/ppo-straight-trained.zip')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed-start', type=int, default=10000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 32:
        parser.error('episodes must be between 1 and 32')
    if not 0 <= args.seed_start <= 2**32-args.episodes:
        parser.error('seed roster must fit uint32')
    from launch_exploratory_demo import ensure_runtime
    alternate_result = ensure_runtime([str(Path(__file__).resolve()), *sys.argv[1:]])
    if alternate_result is not None:
        return alternate_result
    import torch
    torch.set_num_threads(1)
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.research.wp7_protocol import load_registration, prepare_scenarios
    from f1rl.exploratory.pilot import evaluate_episode

    registration = load_registration(ROOT, profile='spatial_rotating_owner_v3')
    scenarios = prepare_scenarios(ROOT, registration, seed_index=0)
    factory = next(row.factory for row in scenarios.scenarios if row.track_id == 'straight-sprint-v1')
    model = PPOModel.load(args.model, device='cpu')

    class InferenceSession:
        permit = {'seeds': {'evaluation': list(range(args.seed_start, args.seed_start+args.episodes))}}

        def create_env(self, *, learning, episode):
            if learning:
                raise ValueError('demonstration is inference only')
            env = replace(factory, reset_config=replace(factory.reset_config, seed=args.seed_start+episode)).create()
            original_step = env.step
            steps = 0

            def step(action):
                nonlocal steps
                result = original_step(action)
                steps += 1
                if steps % 10 == 0 or result[2] or result[3]:
                    print(json.dumps({'episode': episode, 'step': steps,
                        'progress_m': result[4].get('route_highwater_meters'),
                        'elapsed_time': result[4].get('elapsed_time')}), flush=True)
                return result

            env.step = step
            return env

    rows = [evaluate_episode(InferenceSession(), model, 'trained_final', i) for i in range(args.episodes)]
    document = {'training_started': False, 'track': 'straight-sprint-v1', 'episodes': rows}
    content = json.dumps(document, indent=2, allow_nan=False)+'\n'
    print(content, flush=True)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(content)
    return 0 if all(row['valid_sample'] for row in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
