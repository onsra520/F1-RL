"""Verify signed penalty-trial scopes; signature verification does not run training."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from f1rl.research.wp7_trusted_reviews import _hex, _utc
from scripts.wp7_penalty_trial_protocol import validate_proposal
from f1rl.research.wp7_train5_internal_execution import OWNER_KEY, _owner_public_key

DOMAIN = b'F1-RL/TRAIN5/penalty-trial/v1\x00'


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def authorization_message(payload):
    return DOMAIN + canonical(payload)


def verify_trial_authorization(root, proposal, output, envelope, *, now=None, allow_existing=False):
    validate_proposal(root, proposal)
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    owner_key = _owner_public_key(root, proposal['source_bindings'][OWNER_KEY])
    if output == root or output.is_relative_to(root) or (output.exists() and not allow_existing):
        raise ValueError('trial output must be a new directory outside checkout')
    if not isinstance(envelope, dict) or set(envelope) != {'payload', 'signature_hex'}:
        raise ValueError('invalid penalty-trial envelope')
    payload = envelope['payload']
    fields = {'schema', 'purpose', 'proposal_sha256', 'output_root', 'nonce', 'issued_at_utc', 'expires_at_utc',
              'accept_missing_seed_history', 'accept_missing_c2_replay', 'accept_internal_review_pending'}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError('invalid penalty-trial payload')
    if (payload['schema'] != 'train5_penalty_trial_authorization_v1'
            or payload['purpose'] != 'paired_internal_train_only'
            or payload['proposal_sha256'] != sha256(canonical(proposal)).hexdigest()
            or payload['output_root'] != str(output)
            or any(payload[key] is not True for key in ('accept_missing_seed_history', 'accept_missing_c2_replay',
                                                         'accept_internal_review_pending'))):
        raise ValueError('penalty-trial authorization scope differs')
    _hex(payload['nonce'], 16)
    now = datetime.now(timezone.utc) if now is None else now
    if not isinstance(now, datetime) or now.utcoffset() is None or now.utcoffset().total_seconds() != 0:
        raise ValueError('UTC verification clock required')
    issued, expires = _utc(payload['issued_at_utc']), _utc(payload['expires_at_utc'])
    if not issued <= now < expires or (expires - issued).total_seconds() > 168 * 3600:
        raise ValueError('invalid or expired penalty-trial authorization window')
    if not isinstance(owner_key, bytes) or len(owner_key) != 32:
        raise ValueError('owner Ed25519 public key required')
    try:
        Ed25519PublicKey.from_public_bytes(owner_key).verify(_hex(envelope['signature_hex'], 64), authorization_message(payload))
    except InvalidSignature as exc:
        raise ValueError('invalid penalty-trial signature') from exc
    return {'signature_verified': True, 'training_permitted': False}
