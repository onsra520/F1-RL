"""Permit checks never allocate PPO models or execute Gym interactions."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import os
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def api():
    return importlib.import_module('f1rl.exploratory.permit')


@pytest.fixture
def signed():
    private = Ed25519PrivateKey.generate()
    payload = dict(schema_version='straight_exploratory_execution_v1',
                   purpose='execute_exploratory_only', permit_sha256='a'*64,
                   issued_at_utc='2026-10-19T00:00:00+00:00',
                   expires_at_utc='2026-10-19T02:00:00+00:00')
    anchor = dict(key_id='test-owner', public_key_hex=private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw).hex(), revoked=False,
        not_before_utc='2026-01-01T00:00:00+00:00', expires_at_utc='2027-01-01T00:00:00+00:00')
    envelope = dict(payload=payload, key_id='test-owner',
                    signature_hex=private.sign(api().authorization_message(payload)).hex())
    return private, envelope, dict(schema_version='straight_exploratory_owner_keys_v1', owners=[anchor])


def verify(envelope, registry):
    return api().verify_authorization(envelope, registry, permit_sha256='a'*64,
        now=datetime(2026, 10, 19, 1, tzinfo=timezone.utc))


def test_signed_execution_is_separate_from_preparation(signed):
    _, envelope, registry = signed
    assert verify(envelope, registry)['purpose'] == 'execute_exploratory_only'


@pytest.mark.parametrize('mutation', ['empty_registry', 'revoked', 'expired', 'future',
    'wrong_permit', 'preparation_only', 'scientific_purpose', 'tampered_signature',
    'embedded_key', 'boolean_grant', 'duplicate_anchor', 'non_utc', 'extra_payload'])
def test_authorization_denials(signed, mutation):
    private, envelope, registry = signed
    envelope, registry = deepcopy(envelope), deepcopy(registry)
    if mutation == 'empty_registry': registry['owners'] = []
    elif mutation == 'revoked': registry['owners'][0]['revoked'] = True
    elif mutation == 'expired': envelope['payload']['expires_at_utc'] = '2026-10-19T00:30:00+00:00'
    elif mutation == 'future': envelope['payload']['issued_at_utc'] = '2026-10-19T01:30:00+00:00'
    elif mutation == 'wrong_permit': envelope['payload']['permit_sha256'] = 'b'*64
    elif mutation == 'preparation_only': envelope['payload']['purpose'] = 'prepare_only'
    elif mutation == 'scientific_purpose': envelope['payload']['purpose'] = 'scientific_run_review'
    elif mutation == 'tampered_signature': envelope['signature_hex'] = '0'*128
    elif mutation == 'embedded_key': envelope['public_key_hex'] = registry['owners'][0]['public_key_hex']
    elif mutation == 'boolean_grant': envelope = {'authorized': True}
    elif mutation == 'duplicate_anchor': registry['owners'].append(deepcopy(registry['owners'][0]))
    elif mutation == 'non_utc': envelope['payload']['issued_at_utc'] = '2026-10-19T00:00:00'
    elif mutation == 'extra_payload': envelope['payload']['training_authorized'] = True
    if 'payload' in envelope and mutation != 'tampered_signature':
        envelope['signature_hex'] = private.sign(api().authorization_message(envelope['payload'])).hex()
    with pytest.raises(ValueError): verify(envelope, registry)


def test_local_owner_consent_is_exactly_one_scoped_hour_without_signing_key():
    p = api()
    payload = dict(schema_version='straight_exploratory_execution_v1',
        purpose='execute_exploratory_only', permit_sha256='a'*64,
        issued_at_utc='2026-10-19T01:00:00+00:00',
        expires_at_utc='2026-10-19T01:59:00+00:00')
    envelope = dict(approval_mode='local_owner_consent_v1',
        acknowledgement='APPROVE_ONE_LOCAL_EXPLORATORY_RUN', payload=payload)
    now = datetime(2026, 10, 19, 1, 30, tzinfo=timezone.utc)
    assert p.verify_local_owner_approval(envelope, permit_sha256='a'*64, wall_seconds=3600, now=now) == payload
    with pytest.raises(ValueError):
        p.verify_local_owner_approval(envelope, permit_sha256='b'*64, wall_seconds=3600, now=now)
    with pytest.raises(ValueError):
        p.verify_local_owner_approval(envelope, permit_sha256='a'*64, wall_seconds=3600,
            now=datetime(2026, 10, 19, 2, tzinfo=timezone.utc))
    bad = deepcopy(envelope)
    bad['payload']['purpose'] = 'scientific_run_review'
    with pytest.raises(ValueError):
        p.verify_local_owner_approval(bad, permit_sha256='a'*64, wall_seconds=3600, now=now)
    bad = deepcopy(envelope)
    bad['acknowledgement'] = 'yes'
    with pytest.raises(ValueError):
        p.verify_local_owner_approval(bad, permit_sha256='a'*64, wall_seconds=3600, now=now)
    bad['acknowledgement'] = 'APPROVE_ONE_LOCAL_STRAIGHT_PPO_32768_CPU4_60M'
    with pytest.raises(ValueError):
        p.verify_local_owner_approval(bad, permit_sha256='a'*64, wall_seconds=3600, now=now)
    bad = deepcopy(envelope)
    bad['payload']['expires_at_utc'] = '2026-10-19T02:01:00+00:00'
    with pytest.raises(ValueError):
        p.verify_local_owner_approval(bad, permit_sha256='a'*64, wall_seconds=3600, now=now)


def test_local_consent_branch_does_not_require_owner_key_registry(tmp_path, monkeypatch):
    p = api()
    permit = {'sha256': 'a'*64, 'output_directory':str(tmp_path/'run'),
              'limits': {'wall_seconds': 3600}}
    monkeypatch.setattr(p, 'validate_permit', lambda *a: None)
    payload = dict(schema_version='straight_exploratory_execution_v1',
        purpose='execute_exploratory_only', permit_sha256='a'*64,
        issued_at_utc='2020-01-01T00:00:00+00:00',
        expires_at_utc='2099-01-01T00:00:00+00:00')
    envelope = dict(approval_mode='local_owner_consent_v1',
        acknowledgement='APPROVE_ONE_LOCAL_EXPLORATORY_RUN', payload=payload)
    # An approval spanning years is rejected, even when the digest matches.
    with pytest.raises(ValueError, match='one-hour'):
        p.authorize(tmp_path, permit, envelope)


def test_budget_stops_before_excess_step():
    budget = api().Budget(deadline=10., learning_limit=2, clock=lambda: 1.)
    budget.before_step(learning=True)
    budget.before_step(learning=True)
    with pytest.raises(api().PilotStopped, match='learning'): budget.before_step(learning=True)
    budget.before_step(learning=False)
    assert budget.learning_steps == 2


def test_wall_budget_applies_to_baselines_too():
    budget = api().Budget(deadline=10., learning_limit=2, clock=lambda: 10.)
    with pytest.raises(api().PilotStopped, match='deadline'): budget.before_step(learning=False)


def test_unscoped_research_env_does_not_gain_permission():
    from f1rl.learning.algorithms.ppo import _validate_training_admission
    from f1rl.learning.gym_environment import F1RLEnv
    from f1rl.research.wp7_admission import ScientificAdmissionError
    env = object.__new__(F1RLEnv)
    env._requires_scientific_admission = True
    assert api().admitted(env) is False
    with pytest.raises(ScientificAdmissionError): _validate_training_admission(env)


def test_cli_without_execution_grant_denies_before_worker(tmp_path):
    cli = importlib.import_module('f1rl.exploratory.cli')
    with pytest.raises(SystemExit) as exc:
        cli.main(['execute', '--permit', str(tmp_path/'absent.json')])
    assert exc.value.code == 2


def test_strict_json_rejects_duplicates(tmp_path):
    p = tmp_path/'bad.json'
    p.write_text('{"purpose":false,"purpose":true}', encoding='utf-8')
    with pytest.raises(ValueError): api().read_json(p)


@pytest.mark.parametrize('change', ['profile', 'track', 'source', 'runtime', 'limits', 'config_sha256', 'boolean_integer'])
def test_permit_drift_is_rejected(tmp_path, monkeypatch, change):
    p = api()
    expected = dict(sha256='a'*64, output_directory=str(tmp_path/'run'),
        profile=p.PROFILE, track=p.TRACK, config_sha256='f'*64, source={'commit_sha':'a'*40},
        runtime={'python':'3.12'}, limits={'n_envs':1, 'learning_timesteps':32768})
    actual = deepcopy(expected)
    if change == 'profile': actual['profile'] = 'legacy_v1'
    elif change == 'track': actual['track'] = 'held-out'
    elif change == 'source': actual['source']['commit_sha'] = 'b'*40
    elif change == 'runtime': actual['runtime']['python'] = '3.11'
    elif change == 'limits': actual['limits']['learning_timesteps'] += 1
    elif change == 'config_sha256': actual['config_sha256'] = 'b'*64
    elif change == 'boolean_integer': actual['limits']['n_envs'] = True
    monkeypatch.setattr(p, 'prepare', lambda *args: expected)
    with pytest.raises(ValueError): p.validate_permit(tmp_path, actual)


def test_forged_env_flags_never_become_capability():
    from f1rl.learning.gym_environment import F1RLEnv
    env = object.__new__(F1RLEnv)
    env._requires_scientific_admission = True
    env.exploratory_authorized = True
    env.exploratory_permit = {'authorized': True}
    assert api().admitted(env) is False


def test_inference_capability_cannot_authorize_learning():
    from f1rl.learning.gym_environment import F1RLEnv
    env = object.__new__(F1RLEnv)
    api()._sessions[env] = (object(), False)
    assert api().admitted(env) is False


def test_preflight_never_allocates_model_or_resets_gym(tmp_path, monkeypatch, capsys):
    cli = importlib.import_module('f1rl.exploratory.cli')
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.gym_environment import F1RLEnv
    def forbidden(*args, **kwargs): pytest.fail('preflight allocated execution compute')
    monkeypatch.setattr(PPOModel, 'create', forbidden)
    monkeypatch.setattr(F1RLEnv, 'reset', forbidden)
    monkeypatch.setattr(api(), 'prepare', lambda *a: {'sha256':'a'*64, 'state':'prepared_not_authorized'})
    permit = tmp_path/'permit.json'
    assert cli.main(['preflight', '--output-directory', str(tmp_path/'run'), '--permit', str(permit)]) == 0
    assert json.loads(permit.read_text())['state'] == 'prepared_not_authorized'
    assert not (tmp_path/'run').exists()


def test_session_requires_claimed_execution_not_just_a_signed_grant(tmp_path, monkeypatch):
    monkeypatch.setattr(api(), 'authorize', lambda *a: {})
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run')}
    with pytest.raises((ValueError, OSError)):
        api().Session(tmp_path, permit, {}, api().Budget(deadline=1e20, learning_limit=32768))


def test_scoped_scalar_and_vector_guards_reauthenticate_revocation(tmp_path, monkeypatch, signed):
    from datetime import timedelta
    import time
    import torch
    import numpy as np
    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import DummyVecEnv
    from f1rl.learning.gym_environment import F1RLEnv
    from f1rl.learning.algorithms.ppo import _validate_training_admission
    from f1rl.exploratory.pilot import reserve, claim
    p = api()
    private, envelope, registry = signed
    now = datetime.now(timezone.utc)
    envelope['payload']['issued_at_utc'] = (now-timedelta(seconds=1)).isoformat()
    envelope['payload']['expires_at_utc'] = (now+timedelta(minutes=10)).isoformat()
    envelope['signature_hex'] = private.sign(p.authorization_message(envelope['payload'])).hex()
    path = tmp_path/p.KEYS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry), encoding='utf-8')
    permit = {'sha256':'a'*64, 'output_directory':str(tmp_path/'run'),
              'limits': {'threads': 32, 'wall_seconds': 3600, 'learning_timesteps': 32768}}
    monkeypatch.setattr(p, 'prepare', lambda *a: permit)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 1)
    reserve(permit)
    claim(permit)
    session = p.Session(tmp_path, permit, envelope,
                        p.Budget(deadline=time.monotonic()+60, learning_limit=32768))
    env = object.__new__(F1RLEnv)
    env._requires_scientific_admission = True
    env.observation_space = Box(-1., 1., shape=(1,), dtype=np.float32)
    env.action_space = Box(-1., 1., shape=(3,), dtype=np.float32)
    env.render_mode = None
    p._sessions[env] = (session, True)
    vector = DummyVecEnv([lambda: env])
    _validate_training_admission(env)
    _validate_training_admission(vector)
    registry['owners'][0]['revoked'] = True
    path.write_text(json.dumps(registry), encoding='utf-8')
    with pytest.raises(ValueError, match='revoked'): _validate_training_admission(env)
    with pytest.raises(ValueError, match='revoked'): _validate_training_admission(vector)


def test_exploratory_threads_match_config_and_available_processors():
    from f1rl.exploratory.config import resolve
    from pathlib import Path
    config = resolve(Path(__file__).resolve().parents[2])
    import yaml
    document = yaml.safe_load((Path(__file__).resolve().parents[2]/'config.yml').read_text())
    requested = document['exploratory']['resources']['cpu']['threads']
    assert config['limits']['threads'] == (max(1, os.cpu_count() or 1) if requested == 'auto' else requested)
    assert config['limits']['threads'] <= max(1, os.cpu_count() or 1)


def test_session_rejects_cpu_threads_above_permit(monkeypatch):
    import torch
    p = api()
    session = object.__new__(p.Session)
    session.budget = SimpleNamespace(check=lambda: None)
    session.root, session.envelope = '.', {}
    session.permit = {'limits': {'threads': 32}}
    monkeypatch.setattr(p, 'authorize', lambda *a: None)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 1)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 32)
    session.validate()
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 33)
    with pytest.raises(ValueError, match='thread budget exceeded'):
        session.validate()


def test_cuda_loaded_model_cannot_learn_in_exploratory_context(monkeypatch):
    from types import SimpleNamespace
    from f1rl.learning.algorithms.ppo import PPO
    import f1rl.learning.algorithms.ppo as backend
    env = object()
    monkeypatch.setattr(backend, '_validate_training_admission', lambda env: None)
    monkeypatch.setattr(api(), 'admitted', lambda env: True)
    model = object.__new__(PPO)
    model.get_env = lambda: env
    model.device = SimpleNamespace(type='cuda')
    with pytest.raises(ValueError, match='CPU'): model.learn(256)
