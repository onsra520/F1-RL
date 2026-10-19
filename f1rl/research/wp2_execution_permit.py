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
        permissions=dict(training_permitted=True, heldout_permitted=False))
    return {**body, 'sha256': digest(body)}


def _claim_verified(verified, replay_directory):
    payload = verified['payload']
    directory = Path(replay_directory)
    if directory.is_symlink():
        raise ValueError('replay ledger directory may not be a symlink')
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve(strict=True)
    claim_path = directory / f"{payload['nonce']}.json"
    record = dict(permit_sha256=verified['permit_sha256'], run_manifest_sha256=payload['run_manifest_sha256'],
                  run_id=payload['run_id'], claimed_at_utc=datetime.now(timezone.utc).isoformat())
    with claim_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return claim_path


def claim_wp2_execution_permit(envelope, trust_registry, *, expected, replay_directory, now=None):
    """Authenticate and atomically consume a nonce before any candidate work."""
    verified = verify_wp2_execution_permit(envelope, trust_registry, expected=expected, now=now)
    return verified, _claim_verified(verified, replay_directory)


def claim_repository_wp2_execution_permit(repository_root, envelope, *, expected,
                                          replay_directory, now=None):
    """Production boundary: load the committed trust anchors, then verify and consume."""
    registry = load_repository_signers(repository_root)
    return claim_wp2_execution_permit(envelope, registry, expected=expected,
                                      replay_directory=replay_directory, now=now)
