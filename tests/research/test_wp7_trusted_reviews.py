from copy import deepcopy
from datetime import datetime, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.validation.wp7_trusted_reviews import review_message, verify_run_reviews

NOW = datetime(2026, 10, 16, tzinfo=timezone.utc)


def signed_review():
    payload = dict(schema_version='wp7_run_review_v1', purpose='scientific_run_review',
                   protocol_sha256='a'*64, run_manifest_sha256='b'*64,
                   issued_at_utc='2026-10-15T00:00:00+00:00', expires_at_utc='2026-10-17T00:00:00+00:00')
    registry = dict(schema_version='wp7_trusted_reviewers_v1', reviewers=[])
    envelope = dict(payload=payload, signatures=[])
    for role in ('owner', 'independent_reviewer'):
        key = Ed25519PrivateKey.generate()
        registry['reviewers'].append(dict(key_id=role, role=role,
            public_key_hex=key.public_key().public_bytes_raw().hex(), revoked=False,
            not_before_utc='2026-01-01T00:00:00+00:00', expires_at_utc='2027-01-01T00:00:00+00:00'))
        envelope['signatures'].append(dict(key_id=role, signature_hex=key.sign(review_message(payload)).hex()))
    return envelope, registry


def verify(envelope, registry):
    return verify_run_reviews(envelope, registry, protocol_sha256='a'*64,
                             run_manifest_sha256='b'*64, now=NOW)


def test_real_independent_signatures_verify_but_never_grant_permissions():
    envelope, registry = signed_review()
    result = verify(envelope, registry)
    assert result['state'] == 'trusted_reviews_verified'
    assert result['reviewer_ids'] == ['independent_reviewer', 'owner']
    assert not any(result['permissions'].values())


@pytest.mark.parametrize('mutation', ('payload', 'signature', 'unknown', 'revoked', 'expired',
    'future', 'wrong_protocol', 'wrong_run', 'purpose', 'unsigned', 'self_boolean', 'duplicate', 'same_key'))
def test_untrusted_tampered_expired_or_replayed_reviews_fail_closed(mutation):
    envelope, registry = signed_review()
    if mutation == 'payload':
        envelope['payload']['issued_at_utc'] = '2026-10-14T00:00:00+00:00'
    elif mutation == 'signature':
        envelope['signatures'][0]['signature_hex'] = '00'*64
    elif mutation == 'unknown':
        registry['reviewers'] = []
    elif mutation == 'revoked':
        registry['reviewers'][0]['revoked'] = True
    elif mutation == 'expired':
        registry['reviewers'][0]['expires_at_utc'] = '2026-10-15T00:00:00+00:00'
    elif mutation == 'future':
        registry['reviewers'][0]['not_before_utc'] = '2026-10-17T00:00:00+00:00'
    elif mutation == 'wrong_protocol':
        envelope['payload']['protocol_sha256'] = 'c'*64
    elif mutation == 'wrong_run':
        envelope['payload']['run_manifest_sha256'] = 'c'*64
    elif mutation == 'purpose':
        envelope['payload']['purpose'] = 'training_authorized'
    elif mutation == 'unsigned':
        envelope['signatures'].pop()
    elif mutation == 'self_boolean':
        envelope['training_permitted'] = True
    elif mutation == 'duplicate':
        envelope['signatures'][1] = deepcopy(envelope['signatures'][0])
    else:
        registry['reviewers'][1]['public_key_hex'] = registry['reviewers'][0]['public_key_hex']
    with pytest.raises(ValueError):
        verify(envelope, registry)


def test_expired_signed_payload_and_naive_verifier_clock_rejected():
    envelope, registry = signed_review()
    for now in (datetime(2026, 10, 18, tzinfo=timezone.utc), datetime(2026, 10, 16)):
        with pytest.raises(ValueError):
            verify_run_reviews(envelope, registry, protocol_sha256='a'*64,
                               run_manifest_sha256='b'*64, now=now)
