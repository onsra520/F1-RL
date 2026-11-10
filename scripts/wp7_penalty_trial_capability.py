"""Narrow, process-local capability for one signed paired trial run."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from f1rl.learning.training.vector_env import _SampledF1RLEnv
from f1rl.research.wp7_trusted_reviews import _utc
from scripts.wp7_terminal_crash_penalty import TerminalCrashPenalty
from scripts.wp7_penalty_trial_authorization import verify_trial_authorization

_active = ContextVar('terminal_penalty_trial_capability', default=None)


def admitted(env):
    state = _active.get()
    if state is None or env is not state['env'] or type(env) is not DummyVecEnv or len(env.envs) != 8:
        return False
    for rank, child in enumerate(env.envs):
        if (type(child) is not TerminalCrashPenalty or type(child.unwrapped) is not _SampledF1RLEnv
                or child.config.amount != state['amount'] or child.unwrapped._rank != rank
                or child.unwrapped._sampling_seed != state['sampling_seed']
                or child.unwrapped._scenarios is not state['scenarios']):
            return False
    return datetime.now(timezone.utc) < _utc(state['envelope']['payload']['expires_at_utc'])


def before_step(child):
    state = _active.get()
    if state is None:
        return
    if not any(child is entry for entry in state['env'].envs) or not admitted(state['env']):
        raise ValueError('trial environment or authorization changed')
    if state['steps'] >= state['max_steps']:
        raise ValueError('signed trial step budget exhausted')
    state['steps'] += 1


@contextmanager
def activate(env, *, root, proposal, output, envelope, scenarios, sampling_seed, intervention, cohort, replicate, evidence_root):
    if _active.get() is not None:
        raise ValueError('nested trial capability prohibited')
    verify_trial_authorization(root, proposal, output, envelope, allow_existing=True)
    if intervention not in proposal['interventions']:
        raise ValueError('unknown intervention')
    rows = [row for row in proposal['pairs'] if row['cohort'] == cohort and row['replicate'] == replicate]
    if len(rows) != 1 or rows[0]['seeds']['sampling'] != sampling_seed:
        raise ValueError('run seed or identity differs from proposal')
    from dataclasses import replace
    from f1rl.learning.training.env_factory import TrainingScenarioSet
    from scripts.wp7_train5_v4_factory_preview import preview_registered_factories
    reference = preview_registered_factories(Path(root), Path(evidence_root), cohort=cohort, replicate=replicate)
    expected = TrainingScenarioSet(tuple(replace(s, factory=replace(s.factory, reset_config=replace(
        s.factory.reset_config, seed=rows[0]['seeds']['perturbation']))) for s in reference.scenarios))
    if scenarios != expected:
        raise ValueError('trial factories or reset seeds differ')
    claim = Path(output) / f'.claim-{intervention}-{cohort}-{replicate}'
    with claim.open('x', encoding='ascii') as marker:
        marker.write(envelope['payload']['nonce'] + '\n')
    state = {'env': env, 'envelope': envelope, 'scenarios': scenarios, 'sampling_seed': sampling_seed,
             'amount': proposal['interventions'][intervention], 'steps': 0,
             'max_steps': proposal['budget_per_run']['actual_final_steps']}
    token = _active.set(state)
    try:
        if not admitted(env):
            raise ValueError('trial environment differs from authorized structure')
        for rank, child in enumerate(env.envs):
            sampled = child.unwrapped
            seed = int.from_bytes(hashlib.sha256(
                f'f1rl/multi-track-sampling/v1/{sampling_seed}/{rank}'.encode('utf8')).digest(), 'big')
            if (sampled._reset_sequence != 0 or sampled._active_env is not None
                    or sampled._selection_rng.getstate() != random.Random(seed).getstate()):
                raise ValueError('trial selectors must be fresh and unmodified')
        yield state
    finally:
        _active.reset(token)
