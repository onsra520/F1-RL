from datetime import datetime, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from f1rl.research.wp2_execution_permit import (
    claim_repository_wp2_execution_permit,
    claim_wp2_execution_permit,
    execution_permit_message,
    verify_wp2_execution_permit,
)


NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)


def identity():
    return dict(source_commit='a' * 40, source_tree_sha256='b' * 64,
        experiment_manifest_sha256='c' * 64, run_id='run-001', run_manifest_sha256='d' * 64,
        candidate_id='A', split='TRAIN')


def signed_permit():
    payload = dict(schema_version='wp2_execution_permit_v1', purpose='authorize_one_abcd_training_run',
        **identity(), action='train_candidate', issued_at_utc='2026-10-20T11:00:00+00:00',
        expires_at_utc='2026-10-20T13:00:00+00:00', nonce='e' * 64)
    registry_rows, signatures, keys = [], [], []
    for role in ('owner', 'independent_reviewer'):
        key = Ed25519PrivateKey.generate()
        public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        key_id = role + '-fixture'
        registry_rows.append(dict(key_id=key_id, role=role, public_key_hex=public.hex(), revoked=False,
            not_before_utc='2026-10-01T00:00:00+00:00', expires_at_utc='2027-01-01T00:00:00+00:00'))
        signatures.append(dict(key_id=key_id, signature_hex=key.sign(execution_permit_message(payload)).hex()))
        keys.append(key)
    envelope = dict(payload=payload, signatures=signatures)
    registry = dict(schema_version='wp2_trusted_signers_v1', signers=registry_rows)
    return envelope, registry


def test_distinct_owner_and_reviewer_can_verify_exact_source_run_bound_train_permit():
    envelope, registry = signed_permit()
    verified = verify_wp2_execution_permit(envelope, registry, expected=identity(), now=NOW)
    assert verified['state'] == 'authenticated_execution_permit_verified'
    assert verified['permissions'] == dict(training_permitted=True, heldout_permitted=False)


@pytest.mark.parametrize('mutation', ['manifest', 'source', 'run', 'candidate', 'split', 'action', 'expiry', 'replay_nonce'])
def test_permit_rejects_scope_change_expiry_and_invalid_replay_nonce(mutation):
    envelope, registry = signed_permit()
    if mutation in ('manifest', 'source', 'run', 'candidate', 'split'):
        expected = identity()
        key = {'manifest':'run_manifest_sha256','source':'source_tree_sha256','run':'run_id',
               'candidate':'candidate_id','split':'split'}[mutation]
        expected[key] = 'f' * len(expected[key])
        with pytest.raises(ValueError):
            verify_wp2_execution_permit(envelope, registry, expected=expected, now=NOW)
    else:
        if mutation == 'action':
            envelope['payload']['action'] = 'heldout_evaluation'
        elif mutation == 'expiry':
            envelope['payload']['expires_at_utc'] = '2026-10-20T11:59:00+00:00'
        else:
            envelope['payload']['nonce'] = 'not-a-hex-nonce'
        with pytest.raises(ValueError):
            verify_wp2_execution_permit(envelope, registry, expected=identity(), now=NOW)


def test_permit_requires_two_distinct_roles_and_atomic_single_use_claim(tmp_path):
    envelope, registry = signed_permit()
    registry['signers'][1]['role'] = 'owner'
    with pytest.raises(ValueError):
        verify_wp2_execution_permit(envelope, registry, expected=identity(), now=NOW)
    envelope, registry = signed_permit()
    claim_wp2_execution_permit(envelope, registry, expected=identity(), replay_directory=tmp_path, now=NOW)
    with pytest.raises(FileExistsError):
        claim_wp2_execution_permit(envelope, registry, expected=identity(), replay_directory=tmp_path, now=NOW)


def test_tampered_signature_and_overlong_lifetime_are_rejected():
    envelope, registry = signed_permit()
    envelope['signatures'][0]['signature_hex'] = '0' * 128
    with pytest.raises(ValueError, match='signature'):
        verify_wp2_execution_permit(envelope, registry, expected=identity(), now=NOW)
    envelope, registry = signed_permit()
    envelope['payload']['expires_at_utc'] = '2026-10-21T12:00:00+00:00'
    with pytest.raises(ValueError, match='24-hour'):
        verify_wp2_execution_permit(envelope, registry, expected=identity(), now=NOW)


def test_repository_trust_registry_is_empty_and_cannot_authenticate_permits():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    envelope, _ = signed_permit()
    with pytest.raises(ValueError):
        claim_repository_wp2_execution_permit(root, envelope, expected=identity(),
            replay_directory=root.parent / 'wp2-empty-trust-ledger-fixture', now=NOW)
