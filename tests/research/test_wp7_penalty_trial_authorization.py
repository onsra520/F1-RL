from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts.wp7_penalty_trial_authorization import authorization_message, canonical, verify_trial_authorization
from scripts.wp7_penalty_trial_protocol import build_proposal
from scripts import wp7_penalty_trial_authorization as verifier

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT.parent / 'slalom-fresh-audit.json').is_file() or not (ROOT.parent / 'TRAIN5-v4-internal-20261109' / 'evaluation-journal.jsonl').is_file(), reason='local trial seed history is not available')
NOW = datetime(2026, 11, 11, tzinfo=timezone.utc)


@pytest.fixture(scope='module')
def proposal():
    return build_proposal(ROOT)


def envelope(proposal, output, key):
    payload = {'schema': 'train5_penalty_trial_authorization_v1', 'purpose': 'paired_internal_train_only',
               'proposal_sha256': sha256(canonical(proposal)).hexdigest(), 'output_root': str(output.resolve()),
               'nonce': 'ab' * 16, 'issued_at_utc': NOW.isoformat(), 'expires_at_utc': (NOW + timedelta(hours=2)).isoformat(),
               'accept_missing_seed_history': True, 'accept_missing_c2_replay': True,
               'accept_internal_review_pending': True}
    return {'payload': payload, 'signature_hex': key.sign(authorization_message(payload)).hex()}


def test_fake_signature_checks_scope_without_granting_training(proposal, tmp_path, monkeypatch):
    key = Ed25519PrivateKey.generate()
    trusted = key.public_key().public_bytes_raw()
    monkeypatch.setattr(verifier, '_owner_public_key', lambda root, digest: trusted)
    output = tmp_path / 'new'
    auth = envelope(proposal, output, key)
    assert verify_trial_authorization(ROOT, proposal, output, auth, now=NOW)['training_permitted'] is False
    changed = deepcopy(proposal)
    changed['pairs'][0]['seeds']['ppo'] += 1
    with pytest.raises(ValueError):
        verify_trial_authorization(ROOT, changed, output, auth, now=NOW)
    for mutate in ('budget', 'source'):
        changed = deepcopy(proposal)
        if mutate == 'budget':
            changed['budget_per_run']['actual_final_steps'] += 1
        else:
            changed['source_bindings']['scripts/wp7_terminal_crash_penalty.py'] = '00' * 32
        with pytest.raises(ValueError):
            verify_trial_authorization(ROOT, changed, output, auth, now=NOW)
    altered = deepcopy(auth)
    altered['payload']['accept_missing_c2_replay'] = False
    with pytest.raises(ValueError):
        verify_trial_authorization(ROOT, proposal, output, altered, now=NOW)
    with pytest.raises(ValueError):
        verify_trial_authorization(ROOT, proposal, tmp_path / 'different', auth, now=NOW)
    monkeypatch.setattr(verifier, '_owner_public_key', lambda root, digest: Ed25519PrivateKey.generate().public_key().public_bytes_raw())
    with pytest.raises(ValueError):
        verify_trial_authorization(ROOT, proposal, output, auth, now=NOW)
    monkeypatch.setattr(verifier, '_owner_public_key', lambda root, digest: trusted)
    with pytest.raises(ValueError):
        verify_trial_authorization(ROOT, proposal, output, auth, now=NOW + timedelta(hours=3))
    output.mkdir()
    with pytest.raises(ValueError):
        verify_trial_authorization(ROOT, proposal, output, auth, now=NOW)
