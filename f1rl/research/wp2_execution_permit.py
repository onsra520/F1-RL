"""Source/run-bound A/B/C/D execution permits; no trust anchor is bootstrapped here."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from .wp7_protocol import digest


PERMIT_DOMAIN = b'F1-RL/WP2/experiment-execution/v1\x00'
REGISTRY_PATH = 'configs/research/wp2-trusted-signers-v1.json'
PERMIT_FIELDS = {
    'schema_version', 'purpose', 'source_commit', 'source_tree_sha256',
    'experiment_manifest_sha256', 'run_id', 'run_manifest_sha256', 'candidate_id',
    'split', 'action', 'issued_at_utc', 'expires_at_utc', 'nonce',
}


def execution_permit_message(payload):
    return PERMIT_DOMAIN + json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False
    ).encode('ascii')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate trusted signer field')
        result[key] = value
    return result


def _hex(value, length, label):
    if (not isinstance(value, str) or len(value) != length
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'invalid {label}')
    return bytes.fromhex(value)


def _utc(value):
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError('UTC timestamp required')
        return stamp
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError('invalid UTC timestamp') from exc


def load_repository_signers(repository_root):
    root = Path(repository_root).resolve(strict=True)
    path = root / REGISTRY_PATH
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('execution trust registry must remain inside the source repository')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique)


def verify_wp2_execution_permit(envelope, trust_registry, *, expected, now=None):
    """Authenticate one exact TRAIN candidate run; never trusts envelope keys."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if (not isinstance(expected, dict)
            or set(expected) != {'source_commit', 'source_tree_sha256', 'experiment_manifest_sha256',
                                 'run_id', 'run_manifest_sha256', 'candidate_id', 'split'}):
        raise ValueError('complete expected source/run identity is required')
    _hex(expected['source_commit'], 40, 'source commit')
    for key in ('source_tree_sha256', 'experiment_manifest_sha256', 'run_manifest_sha256'):
        _hex(expected[key], 64, key)
    if (not isinstance(expected['run_id'], str) or not expected['run_id'].strip()
            or len(expected['run_id']) > 128 or expected['candidate_id'] not in ('A', 'B', 'C', 'D')
            or expected['split'] != 'TRAIN'):
        raise ValueError('only one named A/B/C/D TRAIN run may be authorized')
    now = datetime.now(timezone.utc) if now is None else now
    if not isinstance(now, datetime) or now.utcoffset() is None or now.utcoffset().total_seconds() != 0:
        raise ValueError('trusted UTC verification time required')
    if (not isinstance(envelope, dict) or set(envelope) != {'payload', 'signatures'}
            or not isinstance(envelope['payload'], dict) or set(envelope['payload']) != PERMIT_FIELDS):
        raise ValueError('invalid execution permit envelope')
    payload = envelope['payload']
    if (payload['schema_version'] != 'wp2_execution_permit_v1'
            or payload['purpose'] != 'authorize_one_abcd_training_run'
            or payload['action'] != 'train_candidate'
            or any(payload.get(key) != value for key, value in expected.items())):
        raise ValueError('permit purpose, source, run or candidate scope differs')
    _hex(payload['nonce'], 64, 'single-use nonce')
    issued, expires = _utc(payload['issued_at_utc']), _utc(payload['expires_at_utc'])
    if not issued <= now < expires or expires - issued > timedelta(hours=24):
        raise ValueError('permit is expired, future dated or exceeds 24-hour validity')
    if (not isinstance(trust_registry, dict) or set(trust_registry) != {'schema_version', 'signers'}
            or trust_registry['schema_version'] != 'wp2_trusted_signers_v1'
            or not isinstance(trust_registry['signers'], list)):
        raise ValueError('invalid trusted signer registry')
    anchors, keys = {}, set()
    for row in trust_registry['signers']:
        if not isinstance(row, dict) or set(row) != {
                'key_id', 'role', 'public_key_hex', 'revoked', 'not_before_utc', 'expires_at_utc'}:
            raise ValueError('invalid trusted signer record')
        identity, key = row['key_id'], _hex(row['public_key_hex'], 64, 'Ed25519 public key')
        if (not isinstance(identity, str) or not identity.strip() or identity in anchors or key in keys
                or row['role'] not in ('owner', 'independent_reviewer') or type(row['revoked']) is not bool
                or _utc(row['not_before_utc']) >= _utc(row['expires_at_utc'])):
            raise ValueError('invalid or duplicate trusted signer')
        anchors[identity], keys = row, keys | {key}
    signatures = envelope['signatures']
    if not isinstance(signatures, list) or len(signatures) != 2:
        raise ValueError('exact owner and independent-reviewer signatures required')
    roles, identities = set(), set()
    for signature in signatures:
        if not isinstance(signature, dict) or set(signature) != {'key_id', 'signature_hex'}:
            raise ValueError('invalid execution permit signature')
        identity = signature['key_id']
        if not isinstance(identity, str) or identity not in anchors or identity in identities:
            raise ValueError('unknown or duplicated execution permit signer')
        anchor = anchors[identity]
        if (anchor['role'] in roles or anchor['revoked']
                or not _utc(anchor['not_before_utc']) <= issued < expires <= _utc(anchor['expires_at_utc'])):
            raise ValueError('signer is revoked, expired or duplicates a required role')
        try:
            Ed25519PublicKey.from_public_bytes(_hex(anchor['public_key_hex'], 64, 'public key')).verify(
                _hex(signature['signature_hex'], 128, 'signature'), execution_permit_message(payload))
        except InvalidSignature as exc:
            raise ValueError('invalid execution permit signature') from exc
        identities.add(identity)
        roles.add(anchor['role'])
    if roles != {'owner', 'independent_reviewer'}:
        raise ValueError('distinct owner and independent-reviewer roles are required')
    body = dict(schema_version='wp2_verified_execution_permit_v1',
        state='authenticated_execution_permit_verified', payload=deepcopy(payload),
        permit_sha256=digest(envelope), signer_ids=sorted(identities),
        permissions=dict(training_permitted=False, heldout_permitted=False),
        authorization='signature_authenticated_only')
    return {**body, 'sha256': digest(body)}


def _replay_ledger_directory():
    if os.name == 'nt':
        return Path.home() / 'AppData' / 'Local' / 'F1-RL' / 'wp2-permit-ledger'
    return Path.home() / '.local' / 'state' / 'F1-RL' / 'wp2-permit-ledger'


def _reject_replay_path_redirect(path):
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        is_junction = getattr(current, 'is_junction', lambda: False)()
        if current.is_symlink() or is_junction:
            raise ValueError('replay ledger path may not contain a symlink or junction')


def _fsync_directory(path):
    if os.name == 'nt':
        import ctypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                         ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                         ctypes.c_void_p]
        kernel32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
        kernel32.FlushFileBuffers.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.CreateFileW(str(path), 0x40000000, 0x7, None, 3, 0x02000000, None)
        if handle in (None, ctypes.c_void_p(-1).value):
            error = ctypes.get_last_error()
            raise OSError(error, 'could not open replay-ledger directory for durable flush', str(path))
        try:
            if not kernel32.FlushFileBuffers(handle):
                error = ctypes.get_last_error()
                raise OSError(error, 'could not durably flush replay-ledger directory', str(path))
        finally:
            kernel32.CloseHandle(handle)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _claim_verified(verified):
    payload = verified['payload']
    directory = _replay_ledger_directory()
    _reject_replay_path_redirect(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _reject_replay_path_redirect(directory)
    directory = directory.resolve(strict=True)
    home = Path.home().resolve(strict=True)
    if not directory.is_relative_to(home):
        raise ValueError('replay ledger must remain inside the current user profile')
    for parent in [home, *reversed(directory.parents)]:
        if parent.is_relative_to(home):
            _fsync_directory(parent)
    claim_path = directory / f"{payload['nonce']}.json"
    record = dict(permit_sha256=verified['permit_sha256'], run_manifest_sha256=payload['run_manifest_sha256'],
                  run_id=payload['run_id'], claimed_at_utc=datetime.now(timezone.utc).isoformat())
    with claim_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(directory)
    return claim_path


def claim_wp2_execution_permit(envelope, trust_registry, *, expected, now=None):
    """Authenticate and atomically consume a nonce before any candidate work."""
    verified = verify_wp2_execution_permit(envelope, trust_registry, expected=expected, now=now)
    return verified, _claim_verified(verified)


def claim_repository_wp2_execution_permit(repository_root, envelope, *, expected, now=None):
    """Production boundary: require an approved manifest before granting training."""
    from ..exploratory.abcd import load_abcd_manifest

    manifest = load_abcd_manifest(repository_root)
    if manifest['status'] != 'APPROVED — PREREGISTERED':
        raise ValueError('A/B/C/D remains draft; approved preregistration is required before permit authorization')
    registry = load_repository_signers(repository_root)
    verified = verify_wp2_execution_permit(envelope, registry, expected=expected, now=now)
    claim_path = _claim_verified(verified)
    body = {key: value for key, value in verified.items() if key != 'sha256'}
    body['permissions'] = dict(training_permitted=True, heldout_permitted=False)
    body['authorization'] = 'approved_preregistered_manifest_and_signed_execution_permit'
    authorized = {**body, 'sha256': digest(body)}
    return authorized, claim_path
