from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from scripts import wp7_penalty_trial_authorization as auth
from scripts import wp7_penalty_trial_evaluate as evaluation
from scripts.wp7_penalty_trial_protocol import build_proposal

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT.parent / 'slalom-fresh-audit.json').is_file() or not (ROOT.parent / 'TRAIN5-v4-internal-20261109' / 'evaluation-journal.jsonl').is_file(), reason='local trial seed history is not available')


def _fake_output(tmp_path, name, proposal):
    output = tmp_path / name
    output.mkdir()
    runs = []
    for kind in proposal['interventions']:
        for pair in proposal['pairs']:
            tag = f"{kind}-{pair['cohort']}-{pair['replicate']}"
            path = output / f'{tag}.zip'
            path.write_bytes(tag.encode())
            runs.append({'intervention': kind, 'cohort': pair['cohort'], 'replicate': pair['replicate'],
                         'model': str(path), 'steps': proposal['budget_per_run']['actual_final_steps'],
                         'sha256': sha256(path.read_bytes()).hexdigest()})
    (output / 'training-result.json').write_text(json.dumps({'runs': runs}))
    return output


def _patch(monkeypatch, proposal):
    monkeypatch.setattr(auth, 'validate_proposal', lambda root, p: None if p == proposal else (_ for _ in ()).throw(ValueError('proposal differs')))
    monkeypatch.setattr(evaluation.PPOModel, 'load', lambda path, **kw: SimpleNamespace(tag=Path(path).name))
    monkeypatch.setattr(evaluation, 'validate_loaded_policy', lambda *a: None)
    monkeypatch.setattr(evaluation, 'preview_registered_factories', lambda *a, **k: SimpleNamespace(
        scenarios=[SimpleNamespace(track_id=t, factory=object()) for t in proposal['tracks']]))
    monkeypatch.setattr(evaluation.os, 'fsync', lambda fd: None)


def _fake_metrics(**kw):
    digest = sha256(f"{kw['policy'].tag}|{kw['track_id']}|{kw['condition']}|{kw['environment_seed']}|{kw['policy_seed']}".encode()).digest()
    outcome = 'certified_physical_crash' if digest[0] % 4 == 0 else 'finished'
    return {'outcome': outcome, 'valid_physics': True, 'episode_return': digest[1] / 255.0}


def test_fake_paired_evaluation_uses_shared_seeds(monkeypatch, tmp_path):
    proposal = build_proposal(ROOT)
    proposal['evaluation_seed_rows'] = [row for row in proposal['evaluation_seed_rows'] if row[2] == proposal['tracks'][0] and row[3] == 'nominal' and row[4] == 0]
    proposal['evaluation_episodes'] = 12
    _patch(monkeypatch, proposal)
    monkeypatch.setattr(evaluation, 'evaluate_one', lambda **kw: {'outcome': 'finished', 'valid_physics': True, 'episode_return': 1.0})
    output = _fake_output(tmp_path, 'output', proposal)
    report = evaluation.evaluate_trial(root=ROOT, output=output, proposal=proposal, evidence_root=ROOT)
    assert report['episodes'] == 12 and report['heldout_evaluated'] is False
    rows = [json.loads(line) for line in (output / 'paired-evaluation-journal.jsonl').read_text().splitlines()]
    assert len(rows) == 12
    for i in range(0, 12, 2):
        assert rows[i]['environment_seed'] == rows[i+1]['environment_seed']
        assert rows[i]['policy_seed'] == rows[i+1]['policy_seed']


def test_sharded_evaluation_matches_sequential_on_full_seed_table(monkeypatch, tmp_path):
    proposal = build_proposal(ROOT)
    _patch(monkeypatch, proposal)
    monkeypatch.setattr(evaluation, 'evaluate_one', lambda **kw: _fake_metrics(**kw))
    sequential = _fake_output(tmp_path, 'sequential', proposal)
    report_seq = evaluation.evaluate_trial(root=ROOT, output=sequential, proposal=proposal, evidence_root=ROOT)
    sharded = _fake_output(tmp_path, 'sharded', proposal)
    shards = sorted({(r[0], r[1], r[2]) for r in proposal['evaluation_seed_rows']})
    assert len(shards) == 30
    for cohort, rep, track in shards:
        evaluation.evaluate_shard(root=ROOT, output=sharded, proposal=proposal, evidence_root=ROOT,
                                  cohort=cohort, replicate=rep, track=track)
    report_shard = evaluation.merge_shards(root=ROOT, output=sharded, proposal=proposal)
    assert (sequential / 'paired-evaluation-journal.jsonl').read_bytes() == (sharded / 'paired-evaluation-journal.jsonl').read_bytes()
    assert report_seq == report_shard
    assert report_seq['episodes'] == 7200 and report_seq['heldout_evaluated'] is False
    assert (sharded / 'paired-evaluation-result.json').is_file()


def test_scope_rejects_output_inside_checkout_or_existing_directory(tmp_path):
    proposal = build_proposal(ROOT)
    with pytest.raises(ValueError, match='outside checkout'):
        auth.verify_trial_scope(ROOT, proposal, ROOT / 'inside-checkout')
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(ValueError, match='outside checkout'):
        auth.verify_trial_scope(ROOT, proposal, existing)
    assert auth.verify_trial_scope(ROOT, proposal, tmp_path / 'new')['training_permitted'] is False
