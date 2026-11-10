from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts import wp7_penalty_trial_authorization as auth
from scripts import wp7_penalty_trial_evaluate as evaluation
from scripts.wp7_penalty_trial_protocol import build_proposal

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT.parent / 'slalom-fresh-audit.json').is_file() or not (ROOT.parent / 'TRAIN5-v4-internal-20261109' / 'evaluation-journal.jsonl').is_file(), reason='local trial seed history is not available')


def test_fake_paired_evaluation_uses_shared_seeds(monkeypatch, tmp_path):
    proposal = build_proposal(ROOT)
    proposal['evaluation_seed_rows'] = [row for row in proposal['evaluation_seed_rows'] if row[2] == proposal['tracks'][0] and row[3] == 'nominal' and row[4] == 0]
    proposal['evaluation_episodes'] = 12
    key, now = Ed25519PrivateKey.generate(), datetime.now(timezone.utc)
    monkeypatch.setattr(auth, '_owner_public_key', lambda root, digest: key.public_key().public_bytes_raw())
    monkeypatch.setattr(auth, 'validate_proposal', lambda root, p: None if p == proposal else (_ for _ in ()).throw(ValueError('proposal differs')))
    output = tmp_path / 'output'; output.mkdir()
    payload = {'schema': 'train5_penalty_trial_authorization_v1', 'purpose': 'paired_internal_train_only',
               'proposal_sha256': sha256(auth.canonical(proposal)).hexdigest(), 'output_root': str(output.resolve()),
               'nonce': 'cd' * 16, 'issued_at_utc': now.isoformat(), 'expires_at_utc': (now + timedelta(hours=1)).isoformat(),
               'accept_missing_seed_history': True, 'accept_missing_c2_replay': True, 'accept_internal_review_pending': True}
    envelope = {'payload': payload, 'signature_hex': key.sign(auth.authorization_message(payload)).hex()}
    runs = []
    for kind in proposal['interventions']:
        for pair in proposal['pairs']:
            path = output / f"{kind}-{pair['cohort']}-{pair['replicate']}.zip"; path.write_bytes(b'fake')
            runs.append({'intervention': kind, 'cohort': pair['cohort'], 'replicate': pair['replicate'],
                         'model': str(path), 'steps': 200704, 'sha256': sha256(b'fake').hexdigest()})
    (output / 'training-result.json').write_text(json.dumps({'runs': runs}))
    monkeypatch.setattr(evaluation, 'preview_registered_factories', lambda *a, **k: SimpleNamespace(
        scenarios=[SimpleNamespace(track_id=proposal['tracks'][0], factory=object())]))
    monkeypatch.setattr(evaluation.PPOModel, 'load', lambda *a, **k: object())
    monkeypatch.setattr(evaluation, 'validate_loaded_policy', lambda *a: None)
    monkeypatch.setattr(evaluation, 'evaluate_one', lambda **kw: {'outcome': 'finished', 'valid_physics': True, 'episode_return': 1.0})
    monkeypatch.setattr(evaluation.os, 'fsync', lambda fd: None)
    report = evaluation.evaluate_trial(root=ROOT, output=output, proposal=proposal, envelope=envelope, evidence_root=ROOT)
    assert report['episodes'] == 12 and report['heldout_evaluated'] is False
    rows = [json.loads(line) for line in (output / 'paired-evaluation-journal.jsonl').read_text().splitlines()]
    assert len(rows) == 12
    for i in range(0, 12, 2):
        assert rows[i]['environment_seed'] == rows[i+1]['environment_seed']
        assert rows[i]['policy_seed'] == rows[i+1]['policy_seed']
