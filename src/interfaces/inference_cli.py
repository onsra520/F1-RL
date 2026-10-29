"""Run the saved PPO policy on TRAIN Straight, without creating a training session."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


def main(repository_root, argv=None, *, torch_factory):
    ROOT = Path(repository_root)
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
    from src.interfaces.runtime_selection import ensure_runtime
    launcher = ROOT / 'scripts/inference/run_policy.py'
    alternate_result = ensure_runtime(
        ROOT, launcher,
        ['-m', 'scripts.inference.run_policy', *(sys.argv[1:] if argv is None else argv)],
    )
    if alternate_result is not None:
        return alternate_result
    torch_factory().set_num_threads(1)
    from src.rl.algorithms.ppo import PPOModel
    from src.validation.wp7_protocol import load_registration, prepare_scenarios
    from src.evaluation.rollout import evaluate_policy_episode

    registration = load_registration(ROOT, profile='spatial_rotating_owner_v3')
    scenarios = prepare_scenarios(ROOT, registration, seed_index=0)
    factory = next(row.factory for row in scenarios.scenarios if row.track_id == 'straight-sprint-v1')
    model = PPOModel.load(args.model, device='cpu')

    def report_step(episode, step, transition):
        if step % 10 == 0 or transition[2] or transition[3]:
            info = transition[4]
            print(json.dumps({'episode': episode, 'step': step,
                'progress_m': info.get('route_highwater_meters'),
                'elapsed_time': info.get('elapsed_time')}), flush=True)

    rows = []
    for episode in range(args.episodes):
        seed = args.seed_start + episode
        env = replace(factory, reset_config=replace(factory.reset_config, seed=seed)).create()
        rows.append(evaluate_policy_episode(
            env,
            model,
            policy_name='trained_final',
            episode=episode,
            environment_seed=seed,
            on_step=lambda step, transition, episode=episode: report_step(episode, step, transition),
        ))
    document = {'training_started': False, 'track': 'straight-sprint-v1', 'episodes': rows}
    content = json.dumps(document, indent=2, allow_nan=False)+'\n'
    print(content, flush=True)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(content)
    return 0 if all(row['valid_sample'] for row in rows) else 2
