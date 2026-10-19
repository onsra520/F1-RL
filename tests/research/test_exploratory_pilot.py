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


def test_run_directory_cannot_be_reused(tmp_path):
    permit = {'sha256': 'a'*64, 'output_directory': str(tmp_path/'run')}
    pilot().reserve(permit)
    with pytest.raises(FileExistsError): pilot().reserve(permit)


def test_worker_claim_is_single_use(tmp_path):
    permit = {'sha256': 'a'*64, 'output_directory': str(tmp_path/'run')}
    pilot().reserve(permit)
    pilot().claim(permit)
    with pytest.raises(FileExistsError): pilot().claim(permit)


def test_timeout_marks_supervisor_and_keeps_checkpoint(tmp_path, monkeypatch):
    p = pilot()
    permit = {'sha256': 'a'*64, 'output_directory': str(tmp_path/'run')}
    monkeypatch.setattr(p, 'authorize', lambda *a: {})
    def timeout(command, **kwargs):
        root = Path(permit['output_directory'])
        (root/'checkpoint-8192.zip').write_bytes(b'opaque fixture, not a PPO model')
        assert 0 < kwargs['timeout'] <= 3600
        assert kwargs['env']['OMP_NUM_THREADS'] == '4'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(p.subprocess, 'run', timeout)
    assert p.supervise(tmp_path, permit, {}) == 124
    root = Path(permit['output_directory'])
    assert json.loads((root/'supervisor.json').read_text())['status'] == 'timeout'
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
                'race': {'terminal_failure':True},
                'telemetry': {'obstacle_hit_ids':('wall',)},
                'route_highwater_meters':25., 'elapsed_time':0.02}
        def close(self): pass
    session = SimpleNamespace(create_env=lambda **kw: Env())
    model = SimpleNamespace(predict=lambda obs, deterministic: np.array([0., 0.5, 0.]))
    first = p.evaluate_episode(session, model, 'trained_final', 0)
    second = p.evaluate_episode(session, model, 'trained_final', 0)
    assert first == second
    assert first['success'] is False and first['off_route_events'] == 1
    assert first['collisions'] == 1 and first['progress'] == .25
    assert first['duration_s'] == .02 and first['steps'] == 1


def test_pilot_phases_do_not_select_best_checkpoint(tmp_path, monkeypatch):
    p = pilot()
    calls = []
    monkeypatch.setattr(p, 'evaluate_episode', lambda session, model, name, ep: calls.append((name, ep)) or {
        'success':False, 'progress':0., 'return':0., 'trace_sha256':'a'*64})
    rows = p.evaluate_roster(None, None, 'untrained', tmp_path)
    assert len(rows) == 32
    assert calls == [('untrained', i) for i in range(32)]


def test_worker_orchestration_with_fake_model_only(tmp_path, monkeypatch):
    p = pilot()
    from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
    from dataclasses import asdict
    import torch
    permit = dict(sha256='a'*64, output_directory=str(tmp_path/'run'), ppo=asdict(PPOConfig()))
    envelope = {'payload':{'expires_at_utc':'2099-01-01T00:00:00+00:00'}}
    output = p.reserve(permit)
    monkeypatch.setattr(p, 'authorize', lambda *a: {})
    for name in ('set_num_threads', 'set_num_interop_threads', 'use_deterministic_algorithms'):
        monkeypatch.setattr(torch, name, lambda *a: None)
    hook = SimpleNamespace(remove=lambda: None)
    optimizer = SimpleNamespace(register_step_post_hook=lambda callback: setattr(hook, 'callback', callback) or hook)
    backend = SimpleNamespace(policy=SimpleNamespace(optimizer=optimizer), _n_updates=0,
        set_logger=lambda logger: None)
    model = SimpleNamespace(num_timesteps=0, _model=backend, learned=False)
    def learn(steps, progress_hook):
        assert steps == 32768
        for step in range(8192, steps+1, 8192):
            model.num_timesteps = step
            hook.callback()
            backend._n_updates += 1
            progress_hook(step)
        model.learned = True
    model.learn = learn
    monkeypatch.setattr(PPOModel, 'create', lambda *a: model)
    monkeypatch.setattr(PPOModel, 'load', lambda *a, **kw: model)
    monkeypatch.setattr(p, 'weights_sha256', lambda m: 'after' if m.learned else 'before')
    saved = []
    def save(m, directory, name, updates, optimizer_steps):
        saved.append(name)
        return directory/(name+'.zip')
    monkeypatch.setattr(p, 'save_checkpoint', save)
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(p, 'Session', lambda *a, **kw: SimpleNamespace(
        create_env=lambda **kw: env, validate=lambda: None))
    monkeypatch.setattr(p, 'evaluate_episode', lambda session, model, name, ep:
        {'success':False, 'progress':0., 'policy':name, 'episode':ep})
    assert p.worker(tmp_path, permit, envelope) == 0
    comparison = json.loads((output/'comparison.json').read_text())
    assert comparison['timesteps'] == 32768 and comparison['optimizer_steps'] == 4
    assert comparison['learning_proven'] is True and comparison['autonomous_completion_observed'] is False
    assert saved == ['untrained', 'checkpoint-8192', 'checkpoint-16384', 'checkpoint-24576', 'checkpoint-32768', 'trained-final']
    assert json.loads((output/'state.json').read_text())['status'] == 'completed'


def test_expired_lease_cannot_be_claimed(tmp_path):
    p = pilot()
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run')}
    output = p.reserve(permit)
    p.write_json(output/'lease.json', {'permit_sha256':'a'*64, 'started_epoch':0.})
    with pytest.raises(ValueError): p.claim(permit)
    assert not (output/'worker.claim').exists()


def test_supervisor_does_not_spawn_after_validation_consumes_wall_budget(tmp_path, monkeypatch):
    p = pilot()
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run')}
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
        p.evaluate_roster(None, None, 'untrained', tmp_path)
    assert calls == [0]
    assert json.loads((tmp_path/'evaluation.jsonl').read_text()) == row


def test_claim_returns_fixed_monotonic_deadline_not_stale_duration(tmp_path, monkeypatch):
    p = pilot()
    monkeypatch.setattr(p, 'time', SimpleNamespace(time=lambda:200., monotonic=lambda:100.))
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run')}
    p.reserve(permit)
    deadline = p.claim(permit)
    assert 3699. <= deadline <= 3700.
    # Later initialization consumes time; it must not extend the lease.
    later = 120.
    budget = p.Budget(deadline=deadline, clock=lambda:later)
    assert 3579. <= budget.deadline-budget.clock() <= 3580.
