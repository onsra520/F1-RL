"""Separate, TRAIN-only execution authority. Preparation never issues authority.

This is an application guard, not a sandbox against arbitrary Python code.
Owner keys must be authenticated externally and committed before preparation.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time
from weakref import WeakKeyDictionary

from ..research.wp7_protocol import digest
from ..research.wp7_run_manifest import read_json
from ..research.wp7_trusted_reviews import _fields, _hex, _utc

TRACK = 'straight-sprint-v1'
PROFILE = 'spatial_rotating_owner_v3'
POLICY = f'singletrack-{TRACK}-seed0'
KEYS = 'configs/exploratory/straight-owner-keys-v1.json'
_sessions = WeakKeyDictionary()
_issued_sessions = set()


def authorization_message(payload):
    return b'F1-RL/exploratory/straight-execution/v1\x00' + json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def verify_authorization(envelope, registry, *, permit_sha256, now=None):
    """Authenticate execution, not a preparation decision or scientific review."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    now = datetime.now(timezone.utc) if now is None else now
    _utc(now.isoformat())
    _hex(permit_sha256, 32)
    _fields(envelope, 'payload key_id signature_hex')
    payload = envelope['payload']
    _fields(payload, 'schema_version purpose permit_sha256 issued_at_utc expires_at_utc')
    if (payload['schema_version'] != 'straight_exploratory_execution_v1'
            or payload['purpose'] != 'execute_exploratory_only'
            or payload['permit_sha256'] != permit_sha256):
        raise ValueError('exploratory execution purpose/permit mismatch')
    issued, expires = _utc(payload['issued_at_utc']), _utc(payload['expires_at_utc'])
    if not issued <= now < expires:
        raise ValueError('execution authorization outside validity window')
    _fields(registry, 'schema_version owners')
    if registry['schema_version'] != 'straight_exploratory_owner_keys_v1' or type(registry['owners']) is not list:
        raise ValueError('invalid exploratory owner registry')
    anchors, keys = {}, set()
    for row in registry['owners']:
        _fields(row, 'key_id public_key_hex revoked not_before_utc expires_at_utc')
        key = _hex(row['public_key_hex'], 32)
        identity = row['key_id']
        if (not isinstance(identity, str) or not identity.strip() or identity in anchors
                or key in keys or type(row['revoked']) is not bool
                or _utc(row['not_before_utc']) >= _utc(row['expires_at_utc'])):
            raise ValueError('invalid or duplicate exploratory owner key')
        anchors[identity], keys = row, keys | {key}
    identity = envelope['key_id']
    if not isinstance(identity, str) or identity not in anchors:
        raise ValueError('no authenticated exploratory owner key')
    anchor = anchors[identity]
    if (anchor['revoked'] or not _utc(anchor['not_before_utc']) <= issued < expires <= _utc(anchor['expires_at_utc'])):
        raise ValueError('owner key revoked or outside validity window')
    try:
        Ed25519PublicKey.from_public_bytes(_hex(anchor['public_key_hex'], 32)).verify(
            _hex(envelope['signature_hex'], 64), authorization_message(payload))
    except InvalidSignature as exc:
        raise ValueError('invalid exploratory owner signature') from exc
    return dict(payload)


def runtime_identity():
    from ..research.wp7_protocol import _cpu_model
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
        system=platform.system(), release=platform.release(), architecture=platform.machine(),
        cpu=_cpu_model(), packages={p: metadata.version(p) for p in
            ('numpy', 'gymnasium', 'torch', 'stable-baselines3', 'PyYAML', 'cryptography')})


def prepare(root, output):
    """Pure preparation: source/config inspection, no reset/model/rollout/learning."""
    from ..research.wp7_protocol import build_protocol
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('pilot output must be external to source checkout')
    protocol = build_protocol(root, profile=PROFILE)
    if protocol['source']['dirty']:
        raise ValueError('clean committed source required for exploratory preparation')
    path = root / KEYS
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('owner registry must be repository bound')
    registry = read_json(path)
    if set(registry) != {'schema_version', 'owners'} or registry['schema_version'] != 'straight_exploratory_owner_keys_v1':
        raise ValueError('invalid owner registry')
    doc = dict(schema_version='straight_exploratory_permit_v1', state='prepared_not_authorized',
        purpose='exploratory_only_not_scientific_evidence', track=TRACK, profile=PROFILE,
        policy=POLICY, source=protocol['source'], protocol_sha256=protocol['sha256'],
        source_files=protocol['source_files'], physics_sha256=protocol['physics_sha256'],
        runtime=runtime_identity(), owner_registry_sha256=sha256(path.read_bytes()).hexdigest(),
        output_directory=str(output), limits=dict(learning_timesteps=32768, threads=4,
            wall_seconds=3600, device='cpu', n_envs=1, checkpoint_steps=8192),
        seeds=dict(ppo=0, training_reset=1000, evaluation=list(range(10000, 10032))),
        ppo=protocol['policies'][0]['ppo'], evaluation=dict(episodes_per_policy=32,
            policies=['random', 'untrained', 'trained_final'], actions='deterministic_for_PPO',
            selection='final_only_intermediate_diagnostic', replay='first_trained_episode'),
        scientific_permissions_unchanged=True, training_authorized=False,
        exploratory_execution_authorized=False)
    return {**doc, 'sha256': digest(doc)}


def validate_permit(root, permit):
    if (not isinstance(permit, dict)
            or digest(permit) != digest(prepare(root, permit.get('output_directory', '')))):
        raise ValueError('exploratory permit/source/runtime/scope drift')


def authorize(root, permit, envelope):
    validate_permit(root, permit)
    return verify_authorization(envelope, read_json(Path(root)/KEYS), permit_sha256=permit['sha256'])


class PilotStopped(RuntimeError):
    pass


@dataclass
class Budget:
    deadline: float
    learning_limit: int = 32768
    clock: object = time.monotonic
    learning_steps: int = 0

    def check(self):
        if self.clock() >= self.deadline:
            raise PilotStopped('pilot deadline reached')

    def before_step(self, *, learning):
        self.check()
        if learning:
            if self.learning_steps >= self.learning_limit:
                raise PilotStopped('pilot learning budget reached')
            self.learning_steps += 1


def admitted(env):
    """Recognize only live, specifically issued env capabilities, never flags."""
    from ..learning.gym_environment import F1RLEnv
    from stable_baselines3.common.vec_env import VecEnv
    if isinstance(env, F1RLEnv):
        entry = _sessions.get(env)
        if entry is None or not entry[1]:
            return False
        session = entry[0]
        session.validate()
        return True
    if isinstance(env, VecEnv) and hasattr(env, 'envs'):
        children = env.envs
        return len(children) == 1 and admitted(children[0].unwrapped)
    return False


class Session:
    """One claimed execution; PPO guards reauthenticate at each model boundary."""
    def __init__(self, root, permit, envelope, budget, episode_observer=None):
        self.root, self.permit, self.envelope, self.budget = root, permit, envelope, budget
        self.episode_observer = episode_observer
        output = Path(permit['output_directory'])
        claim = read_json(output/'worker.claim')
        lease = read_json(output/'lease.json')
        identity = (os.getpid(), str(output), permit['sha256'])
        remaining = lease['started_epoch']+3600-time.time()
        if (claim != dict(permit_sha256=permit['sha256'], pid=os.getpid())
                or lease['permit_sha256'] != permit['sha256'] or not 0 < remaining <= 3600
                or identity in _issued_sessions or type(budget.learning_limit) is not int
                or budget.learning_limit != 32768 or budget.learning_steps != 0
                or budget.clock is not time.monotonic
                or not time.monotonic() < budget.deadline <= time.monotonic()+remaining):
            raise ValueError('claimed one-use execution and bounded fresh budget required')
        self.validate()
        _issued_sessions.add(identity)

    def validate(self):
        self.budget.check()
        authorize(self.root, self.permit, self.envelope)
        import torch
        if torch.get_num_threads() > 4 or torch.get_num_interop_threads() > 4:
            raise ValueError('exploratory CPU thread budget exceeded')

    def create_env(self, *, learning=False, episode=None):
        from dataclasses import replace
        from ..research.wp7_protocol import build_protocol, prepare_training_scenarios
        self.validate()
        if type(learning) is not bool:
            raise ValueError('explicit learning/inference environment purpose required')
        protocol = build_protocol(self.root, profile=PROFILE)
        scenarios, _ = prepare_training_scenarios(self.root, protocol, POLICY)
        factory = scenarios.scenarios[0].factory
        if tuple(s.track_id for s in scenarios.scenarios) != (TRACK,):
            raise ValueError('Straight-only scope required')
        if not learning:
            if type(episode) is not int or not 0 <= episode < 32:
                raise ValueError('fixed evaluation episode roster required')
            factory = replace(factory, reset_config=replace(factory.reset_config,
                seed=self.permit['seeds']['evaluation'][episode]))
        env = factory.create()
        original_step = env.step
        episode_return = 0.

        def step(action):
            nonlocal episode_return
            self.budget.before_step(learning=learning)
            if datetime.now(timezone.utc) >= _utc(self.envelope['payload']['expires_at_utc']):
                raise PilotStopped('execution authorization expired')
            result = original_step(action)
            episode_return += float(result[1])
            if result[2] or result[3]:
                if learning and self.episode_observer:
                    self.episode_observer(self.budget.learning_steps, episode_return, result[4])
                episode_return = 0.
            return result

        env.step = step
        _sessions[env] = (self, learning)
        return env
