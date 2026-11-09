"""Control-flow tests use fake workers/policies, never training compute."""
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


def pilot():
    return importlib.import_module('f1rl.exploratory.pilot')


def fake_session(factory=None, count=32):
    return SimpleNamespace(
        create_env=factory or (lambda **kw: None),
        permit={'seeds': {'evaluation': list(range(10000, 10000+count))},
                'evaluation': {'episodes_per_policy': count}})


def test_run_directory_cannot_be_reused(tmp_path):
    permit = {'sha256': 'a'*64, 'output_directory': str(tmp_path/'run'),
              'limits': {'wall_seconds': 3600}}
    pilot().reserve(permit)
    with pytest.raises(FileExistsError): pilot().reserve(permit)


def test_worker_claim_is_single_use(tmp_path):
    permit = {'sha256': 'a'*64, 'output_directory': str(tmp_path/'run'),
              'limits': {'wall_seconds': 3600}}
    pilot().reserve(permit)
    pilot().claim(permit)
    with pytest.raises(FileExistsError): pilot().claim(permit)


def test_timeout_marks_supervisor_and_keeps_checkpoint(tmp_path, monkeypatch):
    p = pilot()
    permit = {'sha256': 'a'*64, 'output_directory': str(tmp_path/'run'),
              'source': {'commit_sha':'b'*40}, 'config_sha256':'c'*64,
              'seeds': {'training_reset':1234, 'ppo':0},
              'limits': {'threads': 32, 'blas_threads': 16, 'wall_seconds': 3600}}
    monkeypatch.setattr(p, 'authorize', lambda *a: {})
    def timeout(command, **kwargs):
        root = Path(permit['output_directory'])
        (root/'checkpoint-8192.zip').write_bytes(b'opaque fixture, not a PPO model')
        assert 0 < kwargs['timeout'] <= 3600
        assert kwargs['env']['OMP_NUM_THREADS'] == str(permit['limits']['threads'])
        assert all(kwargs['env'][name] == str(permit['limits']['blas_threads']) for name in
            ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(p.subprocess, 'run', timeout)
    assert p.supervise(tmp_path, permit, {}) == 124
    root = Path(permit['output_directory'])
    assert json.loads((root/'supervisor.json').read_text())['status'] == 'timeout'
    manifest = json.loads((root/'engineering-manifest.json').read_text())
    assert manifest['engineering_only'] is True
    assert manifest['scientific_acceptance'] is False
    assert (root/'checkpoint-8192.zip').is_file()


def test_authorization_denied_before_output_allocation(tmp_path, monkeypatch):
    p = pilot()
    permit = {'sha256': 'a'*64, 'output_directory': str(tmp_path/'run')}
    def deny(*a): raise ValueError('no execution grant')
    monkeypatch.setattr(p, 'authorize', deny)
    with pytest.raises(ValueError): p.supervise(tmp_path, permit, {})
    assert not Path(permit['output_directory']).exists()


def test_evaluation_records_autonomous_failures_and_replay(monkeypatch, tmp_path):
    p = pilot()
    class Env:
        action_space = SimpleNamespace(low=np.array([-1., 0., 0.]), high=np.array([1., 1., 1.]))
        _authored_route = SimpleNamespace(total_meters=100.)
        def reset(self, seed): return np.array([0.], dtype=np.float32), {}
        def step(self, action):
            return np.array([1.], dtype=np.float32), 2., True, False, {
                'research_outcome': {'kind':'valid_terminal_failure', 'valid_sample':True,
                                     'reason':'hard_route_violation'},
                'race': {'terminal_failure':True,
                         'completed_checkpoint_ids':('cp-1',),
                         'directed_crossings':('cp-1','cp-uncompleted')},
                'telemetry': {'obstacle_hit_ids':('wall',)},
                'route_highwater_meters':25., 'elapsed_time':0.02}
        def close(self): pass
    session = fake_session(lambda **kw: Env())
    model = SimpleNamespace(predict=lambda obs, deterministic: np.array([0., 0.5, 0.]))
    first = p.evaluate_episode(session, model, 'trained_final', 0)
    second = p.evaluate_episode(session, model, 'trained_final', 0)
    assert first == second
    assert first['success'] is False and first['off_route_events'] == 1
    assert first['collisions'] == 1 and first['progress'] == .25
    assert first['duration_s'] == .02 and first['steps'] == 1
    assert first['completed_checkpoint_ids'] == ['cp-1']
    assert first['directed_crossings'] == ['cp-1','cp-uncompleted']


def test_evaluation_preserves_order_and_separates_completed_from_observed_crossings():
    p = pilot()
    infos = [
        {'research_outcome': {'kind':'active', 'valid_sample':True},
         'race': {'completed_checkpoint_ids':('cp-1',),
                  'directed_crossings':('cp-1','cp-3')},
         'telemetry': {'obstacle_hit_ids':()}, 'route_highwater_meters':10., 'elapsed_time':.1},
        {'research_outcome': {'kind':'successful_finish', 'valid_sample':True},
         'race': {'completed_checkpoint_ids':('cp-2',),
                  'directed_crossings':('cp-2','finish')},
         'telemetry': {'obstacle_hit_ids':()}, 'route_highwater_meters':100., 'elapsed_time':.2},
    ]
    class Env:
        action_space = SimpleNamespace(low=np.array([-1., 0., 0.]), high=np.array([1., 1., 1.]))
        _authored_route = SimpleNamespace(total_meters=100.)
        def __init__(self): self.index = 0
        def reset(self, seed): return np.array([0.], dtype=np.float32), {}
        def step(self, action):
            info = infos[self.index]
            terminated = self.index == 1
            self.index += 1
            return np.array([1.], dtype=np.float32), 1., terminated, False, info
        def close(self): pass
    env = Env()
    session = fake_session(lambda **kw: env)
    model = SimpleNamespace(predict=lambda obs, deterministic: np.array([0., 0.5, 0.]))

    row = p.evaluate_episode(session, model, 'trained_final', 0)

    assert row['success'] is True
    assert row['completed_checkpoint_ids'] == ['cp-1','cp-2']
    assert row['directed_crossings'] == ['cp-1','cp-3','cp-2','finish']


def test_incomplete_episode_keeps_crossings_without_promoting_them_to_completed():
    p = pilot()
    info = {'research_outcome': {'kind':'valid_truncation', 'valid_sample':True, 'reason':'step_limit'},
            'race': {'completed_checkpoint_ids':(), 'directed_crossings':('cp-2','cp-3')},
            'telemetry': {'obstacle_hit_ids':()}, 'route_highwater_meters':20., 'elapsed_time':1.}
    class Env:
        action_space = SimpleNamespace(low=np.array([-1., 0., 0.]), high=np.array([1., 1., 1.]))
        _authored_route = SimpleNamespace(total_meters=100.)
        def reset(self, seed): return np.array([0.], dtype=np.float32), {}
        def step(self, action): return np.array([1.], dtype=np.float32), 1., False, True, info
        def close(self): pass
    session = fake_session(lambda **kw: Env())
    model = SimpleNamespace(predict=lambda obs, deterministic: np.array([0., 0.5, 0.]))

    row = p.evaluate_episode(session, model, 'trained_final', 0)

    assert row['success'] is False
    assert row['completed_checkpoint_ids'] == []
    assert row['directed_crossings'] == ['cp-2','cp-3']


def test_aborted_episode_persists_prior_crossings_as_jsonl_before_rejection(tmp_path):
    p = pilot()
    info = {'research_outcome': {'kind':'active', 'valid_sample':True},
            'race': {'completed_checkpoint_ids':('cp-1',), 'directed_crossings':('cp-1','cp-3')},
            'telemetry': {'obstacle_hit_ids':()}, 'route_highwater_meters':10., 'elapsed_time':.1}
    class Env:
        action_space = SimpleNamespace(low=np.array([-1., 0., 0.]), high=np.array([1., 1., 1.]))
        _authored_route = SimpleNamespace(total_meters=100.)
        def __init__(self): self.steps = 0; self.last_research_outcome = None
        def reset(self, seed): return np.array([0.], dtype=np.float32), {}
        def step(self, action):
            self.steps += 1
            if self.steps == 1: return np.array([1.], dtype=np.float32), 1., False, False, info
            self.last_research_outcome = {'kind':'invalid_environment_transition', 'valid_sample':False,
                                          'reason':'aborted'}
            raise RuntimeError('aborted evaluation')
        def close(self): pass
    env = Env()
    session = fake_session(lambda **kw: env)
    model = SimpleNamespace(predict=lambda obs, deterministic: np.array([0., 0.5, 0.]))

    with pytest.raises(RuntimeError, match='invalid/aborted evaluation episode'):
        p.evaluate_roster(session, model, 'trained_final', tmp_path)

    row = json.loads((tmp_path/'evaluation.jsonl').read_text())
    assert row['valid_sample'] is False and row['error'] == 'RuntimeError: aborted evaluation'
    assert row['completed_checkpoint_ids'] == ['cp-1']
    assert row['directed_crossings'] == ['cp-1','cp-3']


def test_pilot_phases_do_not_select_best_checkpoint(tmp_path, monkeypatch):
    p = pilot()
    calls = []
    monkeypatch.setattr(p, 'evaluate_episode', lambda session, model, name, ep: calls.append((name, ep)) or {
        'success':False, 'progress':0., 'return':0., 'trace_sha256':'a'*64})
    rows = p.evaluate_roster(fake_session(count=7), None, 'untrained', tmp_path)
    assert len(rows) == 7
    assert calls == [('untrained', i) for i in range(7)]


@pytest.mark.parametrize('evaluation_fails', [False, True])
@pytest.mark.parametrize('schedule', ['learning_first', 'baseline_first'])
def test_worker_orchestration_with_fake_model_only(tmp_path, monkeypatch, schedule, evaluation_fails):
    p = pilot()
    from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
    from dataclasses import asdict
    import torch
    from f1rl.learning.train_reset import derive_train_reset_draw_seed
    permit = dict(sha256='a'*64, output_directory=str(tmp_path/'run'),
                  source={'commit_sha':'b'*40}, policy='singletrack-straight-sprint-v1-seed0',
                  track='straight-sprint-v1', seeds={'ppo':0, 'training_reset':1000},
                  limits={'device': 'cpu', 'threads': 32, 'interop_threads': 1, 'learning_timesteps': 4,
                          'checkpoint_steps': 1, 'wall_seconds': 3600},
                  logging={'progress_every_steps': 1},
                  evaluation={'episodes_per_policy': 32, 'replay_episode': 0},
                  execution={'schedule': schedule}, ppo=asdict(PPOConfig()))
    envelope = {'payload':{'expires_at_utc':'2099-01-01T00:00:00+00:00'}}
    output = p.reserve(permit)
    monkeypatch.setattr(p, 'authorize', lambda *a: {})
    for name in ('set_num_threads', 'set_num_interop_threads', 'use_deterministic_algorithms'):
        monkeypatch.setattr(torch, name, lambda *a: None)
    hook = SimpleNamespace(remove=lambda: None)
    optimizer = SimpleNamespace(register_step_post_hook=lambda callback: setattr(hook, 'callback', callback) or hook)
    backend = SimpleNamespace(policy=SimpleNamespace(optimizer=optimizer), _n_updates=0,
        num_timesteps=0, logger=SimpleNamespace(name_to_value={'train/value_loss': 0.5}),
        set_logger=lambda logger: None)
    model = SimpleNamespace(num_timesteps=0, _model=backend, learned=False)
    model.weights_sha256 = lambda: 'a'*64 if not model.learned else 'b'*64
    model.set_csv_logger = lambda directory: PPOModel.set_csv_logger(model, directory)
    model.register_optimizer_step_hook = lambda callback: PPOModel.register_optimizer_step_hook(model, callback)
    events = []
    def learn(steps, progress_hook, telemetry_observer=None, optimizer_step_hook=None):
        events.append(('learn', 'start'))
        assert steps == 4
        telemetry_observer.on_training_start(model, total_timesteps=steps, n_envs=1)
        for step in range(1, steps+1):
            model.num_timesteps = step
            backend.num_timesteps = step
            telemetry_observer.on_step({
                'obs': np.array([[0.0, 1.0]], dtype=np.float32),
                'actions': np.array([[0.1, 0.8]], dtype=np.float32),
                'clipped_actions': np.array([[0.1, 0.8]], dtype=np.float32),
                'rewards': np.array([0.5], dtype=np.float32),
                'dones': np.array([False]),
                'infos': [{
                    'track_id': 'straight-sprint-v1',
                    'train_reset': {'version':'wp7_train_reset_research_v1',
                                    'seed': 1000, 'rank': 0, 'reset_index': 0,
                                    'draw_seed': derive_train_reset_draw_seed(
                                        version='wp7_train_reset_research_v1', seed=1000, rank=0,
                                        track_id='straight-sprint-v1', reset_index=0),
                                    'track_id': 'straight-sprint-v1', 'configuration_sha256': 'd'*64,
                                    'world_sha256': 'e'*64},
                    'reward_components': {'progress': 0.5},
                    'training_telemetry': {
                        'schema_version':'f1rl_training_state_v1',
                        'track_id':'straight-sprint-v1',
                        'seed_identity': {'version':'wp7_train_reset_research_v1',
                                          'seed':1000, 'rank':0, 'reset_index':0,
                                          'draw_seed': derive_train_reset_draw_seed(
                                              version='wp7_train_reset_research_v1', seed=1000, rank=0,
                                              track_id='straight-sprint-v1', reset_index=0),
                                          'track_id':'straight-sprint-v1', 'configuration_sha256':'d'*64,
                                          'world_sha256':'e'*64},
                        'observation_schema': {'version':'f1rl_observation_features_v1',
                                               'feature_names':['target_x','target_z']},
                        'position':(0.,0.,float(step)), 'velocity':(0.,0.,1.), 'speed':1.,
                        'grounded':True, 'heading_rad':0., 'yaw_rate_radps':0.,
                        'lateral_route_deviation_m':0., 'route_progress_m':float(step),
                        'route_highwater_m':float(step), 'action_contract_version':'steering-throttle-brake0-v1',
                    },
                }],
            }, backend)
            telemetry_observer.on_optimizer_step()
            optimizer_step_hook()
            backend._n_updates += 1
            telemetry_observer.on_update(model)
            progress_hook(step)
        model.learned = True
    model.learn = learn
    monkeypatch.setattr(PPOModel, 'create', lambda *a: model)
    def load_model(path, **kwargs):
        events.append(('load', Path(path).stem))
        return SimpleNamespace(snapshot=Path(path).stem)
    monkeypatch.setattr(PPOModel, 'load', load_model)
    monkeypatch.setattr(p, 'weights_sha256', lambda m: 'b'*64 if m.learned else 'a'*64)
    saved = []
    def save(m, directory, name, updates, optimizer_steps):
        saved.append(name)
        path = directory/(name+'.zip')
        path.write_bytes(name.encode())
        return path
    monkeypatch.setattr(p, 'save_checkpoint', save)
    env = SimpleNamespace(close=lambda: None)
    env.training_reset_contract = {
        'version': 'wp7_train_reset_research_v1', 'seed': 1000,
        'configuration_sha256': 'd'*64, 'world_sha256': 'e'*64,
    }
    monkeypatch.setattr(p, 'Session', lambda *a, **kw: SimpleNamespace(
        create_env=lambda **kw: env, validate=lambda: None, permit=permit))
    def evaluation(session, evaluated_model, name, ep):
        if evaluation_fails and name == 'trained_final':
            raise RuntimeError('evaluation interrupted')
        events.append(('evaluate', name, getattr(evaluated_model, 'snapshot', 'live')))
        return {'success':False, 'progress':0., 'policy':name, 'episode':ep}
    monkeypatch.setattr(p, 'evaluate_episode', evaluation)
    if evaluation_fails:
        with pytest.raises(RuntimeError, match='evaluation interrupted'):
            p.worker(tmp_path, permit, envelope)
        assert saved[-1] == 'trained-final'
        assert json.loads((output/'learning-summary.json').read_text())['status'] == 'learning_completed'
        assert json.loads((output/'state.json').read_text())['status'] == 'learning_completed_evaluation_incomplete'
        return
    assert p.worker(tmp_path, permit, envelope) == 0
    comparison = json.loads((output/'comparison.json').read_text())
    assert comparison['timesteps'] == 4 and comparison['optimizer_steps'] == 4
    assert comparison['learning_proven'] is True and comparison['autonomous_completion_observed'] is False
    assert json.loads((output/'learning-summary.json').read_text())['status'] == 'learning_completed'
    observations = [x for x in events if x[0] == 'evaluate']
    assert len(observations) == 3 * permit['evaluation']['episodes_per_policy'] + 1
    if schedule == 'learning_first':
        assert events.index(('learn', 'start')) < events.index(('evaluate', 'trained_final', 'trained-final'))
        assert observations[0] == ('evaluate', 'trained_final', 'trained-final')
        assert ('evaluate', 'untrained', 'untrained') in observations
    else:
        assert observations[0] == ('evaluate', 'random', 'live')
        assert events.index(('evaluate', 'untrained', 'live')) < events.index(('learn', 'start'))
    assert saved == ['untrained', 'checkpoint-1', 'checkpoint-2', 'checkpoint-3', 'checkpoint-4', 'trained-final']
    assert json.loads((output/'state.json').read_text())['status'] == 'completed'


def test_expired_lease_cannot_be_claimed(tmp_path):
    p = pilot()
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run'),
              'limits': {'wall_seconds': 3600}}
    output = p.reserve(permit)
    p.write_json(output/'lease.json', {'permit_sha256':'a'*64, 'started_epoch':0.})
    with pytest.raises(ValueError): p.claim(permit)
    assert not (output/'worker.claim').exists()


def test_supervisor_does_not_spawn_after_validation_consumes_wall_budget(tmp_path, monkeypatch):
    p = pilot()
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run'),
              'limits': {'wall_seconds': 3600}}
    moments = iter([0., 3601., 3601.])
    monkeypatch.setattr(p.time, 'monotonic', lambda: next(moments))
    monkeypatch.setattr(p, 'authorize', lambda *a: {})
    monkeypatch.setattr(p.subprocess, 'run', lambda *a, **kw: pytest.fail('expired budget spawned worker'))
    with pytest.raises(RuntimeError, match='deadline'):
        p.supervise(tmp_path, permit, {})


def test_invalid_episode_is_preserved_and_never_redrawn(tmp_path, monkeypatch):
    p = pilot()
    calls = []
    row = dict(policy='untrained', episode=0, environment_seed=10000, valid_sample=False,
        outcome={'kind':'invalid', 'reason':'domain_error'}, steps=3, **{'return':2.}, trace_sha256='a'*64)
    def invalid(*args):
        calls.append(args[-1])
        return row
    monkeypatch.setattr(p, 'evaluate_episode', invalid)
    with pytest.raises(RuntimeError, match='invalid'):
        p.evaluate_roster(fake_session(), None, 'untrained', tmp_path)
    assert calls == [0]
    assert json.loads((tmp_path/'evaluation.jsonl').read_text()) == row


def test_claim_returns_fixed_monotonic_deadline_not_stale_duration(tmp_path, monkeypatch):
    p = pilot()
    monkeypatch.setattr(p, 'time', SimpleNamespace(time=lambda:200., monotonic=lambda:100.))
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run'),
              'limits': {'wall_seconds': 3600}}
    p.reserve(permit)
    deadline = p.claim(permit)
    assert 3699. <= deadline <= 3700.
    # Later initialization consumes time; it must not extend the lease.
    later = 120.
    budget = p.Budget(deadline=deadline, learning_limit=32768, clock=lambda:later)
    assert 3579. <= budget.deadline-budget.clock() <= 3580.
