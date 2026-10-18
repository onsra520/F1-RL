"""Ed25519 review authentication; scientific permissions remain independently closed.

The trust registry must come from a separately authenticated owner-controlled
source. Keys embedded in review envelopes are never trust anchors.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from .wp7_protocol import digest, _unique


def _fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError('invalid trusted-review fields')


def _hex(value, size):
    if not isinstance(value, str) or len(value) != size*2 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid canonical hexadecimal identity')
    return bytes.fromhex(value)


def _utc(value):
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError('UTC timestamp required')
        return stamp
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid UTC review timestamp') from exc


def review_message(payload):
    """Stable domain-separated bytes signed by both independent reviewers."""
    return b'F1-RL/WP7/run-review/v1\x00' + json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def verify_repository_run_reviews(repository_root, envelope, *, protocol_sha256,
                                  run_manifest_sha256, now=None):
    """Use repository trust anchors, never keys or verification booleans in a report."""
    root = Path(repository_root).resolve(strict=True)
    path = root / 'configs/research/wp7-trusted-reviewers-v1.json'
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('trust registry must remain inside the repository')
    registry = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique)
    return verify_run_reviews(envelope, registry, protocol_sha256=protocol_sha256,
                              run_manifest_sha256=run_manifest_sha256, now=now)


def verify_run_reviews(envelope, trust_registry, *, protocol_sha256, run_manifest_sha256, now=None):
    """Authenticate exact run/protocol review signatures, never authorization.

    Verification time defaults to the actual UTC clock. Explicit time is for
    reproducible audits, whose result does not establish present validity.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    now = datetime.now(timezone.utc) if now is None else now
    if not isinstance(now, datetime) or now.utcoffset() is None or now.utcoffset().total_seconds() != 0:
        raise ValueError('UTC verification clock required')
    _hex(protocol_sha256, 32)
    _hex(run_manifest_sha256, 32)
    _fields(envelope, 'payload signatures')
    payload = envelope['payload']
    _fields(payload, 'schema_version purpose protocol_sha256 run_manifest_sha256 issued_at_utc expires_at_utc')
    if (payload['schema_version'] != 'wp7_run_review_v1' or payload['purpose'] != 'scientific_run_review'
            or payload['protocol_sha256'] != protocol_sha256 or payload['run_manifest_sha256'] != run_manifest_sha256):
        raise ValueError('review purpose, protocol or run mismatch')
    issued, expires = _utc(payload['issued_at_utc']), _utc(payload['expires_at_utc'])
    if not issued <= now < expires:
        raise ValueError('review outside validity window')
    _fields(trust_registry, 'schema_version reviewers')
    if trust_registry['schema_version'] != 'wp7_trusted_reviewers_v1' or not isinstance(trust_registry['reviewers'], list):
        raise ValueError('invalid trust registry')
    anchors, public_keys = {}, set()
    for row in trust_registry['reviewers']:
        _fields(row, 'key_id role public_key_hex revoked not_before_utc expires_at_utc')
        identity = row['key_id']
        key = _hex(row['public_key_hex'], 32)
        if (not isinstance(identity, str) or not identity.strip() or len(identity) > 128
                or identity in anchors or key in public_keys
                or row['role'] not in ('owner', 'independent_reviewer') or type(row['revoked']) is not bool
                or _utc(row['not_before_utc']) >= _utc(row['expires_at_utc'])):
            raise ValueError('invalid or non-independent trust identity')
        anchors[identity] = row
        public_keys.add(key)
    signatures = envelope['signatures']
    if not isinstance(signatures, list) or len(signatures) != 2:
        raise ValueError('exactly owner and independent reviewer signatures required')
    identities, roles = set(), set()
    for signature in signatures:
        _fields(signature, 'key_id signature_hex')
        identity = signature['key_id']
        if not isinstance(identity, str) or identity not in anchors or identity in identities:
            raise ValueError('unknown or duplicated signer')
        anchor = anchors[identity]
        if (anchor['revoked'] or anchor['role'] in roles
                or not _utc(anchor['not_before_utc']) <= issued <= now < expires <= _utc(anchor['expires_at_utc'])):
            raise ValueError('signer revoked, expired, future or role duplicated')
        try:
            Ed25519PublicKey.from_public_bytes(_hex(anchor['public_key_hex'], 32)).verify(
                _hex(signature['signature_hex'], 64), review_message(payload))
        except InvalidSignature as exc:
            raise ValueError('invalid trusted review signature') from exc
        identities.add(identity)
        roles.add(anchor['role'])
    document = dict(schema_version='wp7_authenticated_run_review_v1', state='trusted_reviews_verified',
        protocol_sha256=protocol_sha256, run_manifest_sha256=run_manifest_sha256,
        envelope_sha256=digest(envelope), trust_registry_sha256=digest(trust_registry),
        verified_at_utc=now.isoformat(), reviewer_ids=sorted(identities),
        permissions=dict(training_permitted=False, heldout_permitted=False,
                         scientific_freeze_permitted=False, wp8_permitted=False))
    return {**document, 'sha256': digest(document)}
