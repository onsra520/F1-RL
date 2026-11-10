"""Construct the paired trial vector environment without granting learning authority."""
from stable_baselines3.common.vec_env import DummyVecEnv
from f1rl.learning.training.env_factory import TrainingScenarioSet
from f1rl.learning.training.vector_env import _SampledF1RLEnv
from scripts.wp7_terminal_crash_penalty import TerminalCrashPenalty, TerminalCrashPenaltyConfig
from scripts.wp7_penalty_trial_protocol import require_execution_capability


def _require_clean_source(root):
    import subprocess
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
    if status:
        raise ValueError('clean committed source required')


def build_trial_environment(scenarios, *, sampling_seed, intervention):
    if not isinstance(scenarios, TrainingScenarioSet) or len(scenarios.scenarios) != 5:
        raise ValueError('five registered TRAIN scenarios required')
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32:
        raise ValueError('sampling seed must be uint32')
    if intervention not in ('baseline', 'terminal_crash_penalty'):
        raise ValueError('unknown trial intervention')
    amount = 0 if intervention == 'baseline' else 5
    def make(rank):
        return lambda: TerminalCrashPenalty(_SampledF1RLEnv(scenarios, sampling_seed, rank),
                                            TerminalCrashPenaltyConfig(amount))
    return DummyVecEnv([make(rank) for rank in range(8)])


def run_trial(*, root, proposal, output, envelope, evidence_root):
    """Execute all signed runs once, writing results outside the checkout."""
    from dataclasses import replace
    from hashlib import sha256
    from pathlib import Path
    import json
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.research.wp7_train5_v4_checkpoint import _prepare_v4_runner, FROZEN_CHECKPOINTS
    from scripts.wp7_penalty_trial_authorization import verify_trial_authorization
    from scripts.wp7_penalty_trial_capability import activate
    from scripts.wp7_train5_v4_factory_preview import preview_registered_factories

    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    _require_clean_source(root)
    verify_trial_authorization(root, proposal, output, envelope)
    originals = lambda: {name: sha256((root / 'models' / name / 'model-final.zip').read_bytes()).hexdigest() for name in FROZEN_CHECKPOINTS}
    if originals() != FROZEN_CHECKPOINTS:
        raise ValueError('source checkpoints changed')
    output.mkdir(parents=True, exist_ok=False)
    with (output / ('claim-' + envelope['payload']['nonce'])).open('x', encoding='ascii') as claim:
        claim.write('one-use penalty trial\n')
    completed = []
    journal = output / 'trial-journal.jsonl'
    for intervention in ('baseline', 'terminal_crash_penalty'):
        for row in proposal['pairs']:
            identity = {'intervention': intervention, 'cohort': row['cohort'], 'replicate': row['replicate']}
            with journal.open('a', encoding='utf8') as f:
                f.write(json.dumps({'event': 'started', **identity}) + '\n'); f.flush()
            run_output = output / f"{intervention}-{row['cohort']}-{row['replicate']}"
            run_output.mkdir(exist_ok=False)
            scenarios = preview_registered_factories(root, Path(evidence_root), cohort=row['cohort'], replicate=row['replicate'])
            scenarios = TrainingScenarioSet(tuple(replace(s, factory=replace(s.factory, reset_config=replace(
                s.factory.reset_config, seed=row['seeds']['perturbation']))) for s in scenarios.scenarios))
            reference = _prepare_v4_runner(root, run_output, ppo_seed=row['seeds']['ppo'],
                                            sampling_seed=row['seeds']['sampling'], frozen_checkpoint=row['initial_checkpoint'])
            env = build_trial_environment(scenarios, sampling_seed=row['seeds']['sampling'], intervention=intervention)
            try:
                with activate(env, root=root, proposal=proposal, output=output, envelope=envelope,
                              scenarios=scenarios, sampling_seed=row['seeds']['sampling'], intervention=intervention,
                              cohort=row['cohort'], replicate=row['replicate'], evidence_root=evidence_root) as state:
                    model = PPOModel.create(env, reference._config.ppo)
                    if reference._model_initializer is not None:
                        reference._model_initializer(model)
                    model.learn(reference._config.total_timesteps)
                    actual = model.num_timesteps
                    if actual != state['max_steps'] or state['steps'] != actual:
                        raise ValueError('trial model and environment step counts differ')
                    model_path = model.save(run_output / 'final-model.zip')
                    reference._final_model_observer(model_path, actual)
            finally:
                env.close()
            if originals() != FROZEN_CHECKPOINTS:
                raise ValueError('source checkpoints changed during trial')
            entry = {**identity, 'steps': actual, 'model': str(model_path), 'sha256': sha256(model_path.read_bytes()).hexdigest()}
            completed.append(entry)
            with journal.open('a', encoding='utf8') as f:
                f.write(json.dumps({'event': 'completed', **entry}) + '\n'); f.flush()
            print(f"Completed {intervention} {row['cohort']} replicate {row['replicate']}", flush=True)
    with (output / 'training-result.json').open('x', encoding='utf8') as f:
        json.dump({'runs': completed, 'frozen_checkpoint_sha256': originals(), 'heldout_evaluated': False}, f, indent=2)
    return completed
