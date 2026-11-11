"""Paired penalty-trial runner: one process per training run; output outside the checkout."""
from pathlib import Path
import json
from hashlib import sha256

from stable_baselines3.common.vec_env import DummyVecEnv
from f1rl.learning.training.env_factory import TrainingScenarioSet
from f1rl.learning.training.vector_env import _SampledF1RLEnv
from scripts.wp7_terminal_crash_penalty import TerminalCrashPenalty, TerminalCrashPenaltyConfig

INTERVENTIONS = ('baseline', 'terminal_crash_penalty')


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
    if intervention not in INTERVENTIONS:
        raise ValueError('unknown trial intervention')
    amount = 0 if intervention == 'baseline' else 5
    def make(rank):
        return lambda: TerminalCrashPenalty(_SampledF1RLEnv(scenarios, sampling_seed, rank),
                                            TerminalCrashPenaltyConfig(amount))
    return DummyVecEnv([make(rank) for rank in range(8)])


def run_dir_name(intervention, cohort, replicate):
    return f'{intervention}-{cohort}-{replicate}'


def prepare_output(*, root, proposal, output):
    """Create the single batch output directory and its batch claim (no signature)."""
    from scripts.wp7_penalty_trial_authorization import verify_trial_scope
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    _require_clean_source(root)
    verify_trial_scope(root, proposal, output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'claim-batch').open('x', encoding='ascii') as claim:
        claim.write('one-use penalty trial batch\n')
    return output


def run_one(*, root, proposal, output, evidence_root, intervention, cohort, replicate):
    """Train one (intervention, cohort, replicate) run in this process, single torch thread."""
    from dataclasses import replace
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.research.wp7_train5_v4_checkpoint import _prepare_v4_runner, FROZEN_CHECKPOINTS
    from scripts.wp7_penalty_trial_capability import activate
    from scripts.wp7_train5_v4_factory_preview import preview_registered_factories
    import torch

    torch.set_num_threads(1)
    root, output = Path(root).resolve(strict=True), Path(output).resolve(strict=True)
    originals = lambda: {name: sha256((root / 'models' / name / 'model-final.zip').read_bytes()).hexdigest() for name in FROZEN_CHECKPOINTS}
    if originals() != FROZEN_CHECKPOINTS:
        raise ValueError('source checkpoints changed')
    rows = [row for row in proposal['pairs'] if row['cohort'] == cohort and row['replicate'] == replicate]
    if len(rows) != 1 or intervention not in INTERVENTIONS:
        raise ValueError('unknown run identity')
    row = rows[0]
    identity = {'intervention': intervention, 'cohort': cohort, 'replicate': replicate}
    run_output = output / run_dir_name(intervention, cohort, replicate)
    run_output.mkdir(exist_ok=False)
    journal = run_output / 'run-journal.jsonl'
    with journal.open('x', encoding='utf8') as f:
        f.write(json.dumps({'event': 'started', **identity}) + '\n'); f.flush()
    scenarios = preview_registered_factories(root, Path(evidence_root), cohort=cohort, replicate=replicate)
    scenarios = TrainingScenarioSet(tuple(replace(s, factory=replace(s.factory, reset_config=replace(
        s.factory.reset_config, seed=row['seeds']['perturbation']))) for s in scenarios.scenarios))
    reference = _prepare_v4_runner(root, run_output, ppo_seed=row['seeds']['ppo'],
                                    sampling_seed=row['seeds']['sampling'], frozen_checkpoint=row['initial_checkpoint'])
    env = build_trial_environment(scenarios, sampling_seed=row['seeds']['sampling'], intervention=intervention)
    try:
        with activate(env, root=root, proposal=proposal, output=output, scenarios=scenarios,
                      sampling_seed=row['seeds']['sampling'], intervention=intervention,
                      cohort=cohort, replicate=replicate, evidence_root=evidence_root) as state:
            model = PPOModel.create(env, reference._config.ppo)
            if reference._model_initializer is not None:
                reference._model_initializer(model)
            model.learn(reference._config.total_timesteps,
                        progress_hook=lambda steps: print(f'steps={steps}', flush=True))
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
    with journal.open('a', encoding='utf8') as f:
        f.write(json.dumps({'event': 'completed', **entry}) + '\n'); f.flush()
    with (run_output / 'run-result.json').open('x', encoding='utf8') as f:
        json.dump(entry, f, indent=2); f.write('\n')
    print(f"Completed {intervention} {cohort} replicate {replicate}", flush=True)
    return entry


def write_training_result(*, root, proposal, output):
    """Collect the 12 per-run results in the proposal's order and write the batch result once."""
    root, output = Path(root).resolve(strict=True), Path(output).resolve(strict=True)
    from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS
    originals = {name: sha256((root / 'models' / name / 'model-final.zip').read_bytes()).hexdigest() for name in FROZEN_CHECKPOINTS}
    if originals != FROZEN_CHECKPOINTS:
        raise ValueError('source checkpoints changed during trial')
    runs = []
    for intervention in INTERVENTIONS:
        for row in proposal['pairs']:
            path = output / run_dir_name(intervention, row['cohort'], row['replicate']) / 'run-result.json'
            entry = json.loads(path.read_text(encoding='utf8'))
            if entry['steps'] != proposal['budget_per_run']['actual_final_steps']:
                raise ValueError('run step count differs from budget')
            runs.append(entry)
    if len(runs) != 12:
        raise ValueError('twelve completed runs required')
    with (output / 'training-result.json').open('x', encoding='utf8') as f:
        json.dump({'runs': runs, 'frozen_checkpoint_sha256': originals, 'heldout_evaluated': False}, f, indent=2)
    return runs


def run_trial(*, root, proposal, output, evidence_root):
    """Serial reference path: same per-run code, in one process (used by tests and as the fallback)."""
    output = prepare_output(root=root, proposal=proposal, output=output)
    for intervention in INTERVENTIONS:
        for row in proposal['pairs']:
            run_one(root=root, proposal=proposal, output=output, evidence_root=evidence_root,
                    intervention=intervention, cohort=row['cohort'], replicate=row['replicate'])
    return write_training_result(root=root, proposal=proposal, output=output)
