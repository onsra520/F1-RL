from pathlib import Path
import json
import sys
import pytest
from scripts import wp7_penalty_trial_cli as cli
from scripts import wp7_penalty_trial_runner as runner
from scripts.wp7_penalty_trial_protocol import build_proposal

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT.parent / 'slalom-fresh-audit.json').is_file() or not (ROOT.parent / 'TRAIN5-v4-internal-20261109' / 'evaluation-journal.jsonl').is_file(), reason='local trial seed history is not available')


def test_pool_runs_all_children_and_reports_only_failures(tmp_path):
    logs = tmp_path / 'logs'
    logs.mkdir()
    tasks = []
    for i in range(4):
        code = 'import sys; print("child", %d); sys.exit(%d)' % (i, 3 if i == 1 else 0)
        tasks.append((f'task-{i}', [sys.executable, '-B', '-c', code]))
    failed = cli._run_pool(tmp_path, 2, tasks, logs)
    assert failed == [('task-1', 3)]
    for i in range(4):
        assert 'child %d' % i in (logs / f'task-{i}.log').read_text(encoding='utf8')


def test_training_result_is_collected_in_proposal_order(tmp_path):
    proposal = build_proposal(ROOT)
    output = tmp_path / 'out'
    output.mkdir()
    for intervention in runner.INTERVENTIONS:
        for row in proposal['pairs']:
            run = output / runner.run_dir_name(intervention, row['cohort'], row['replicate'])
            run.mkdir()
            entry = {'intervention': intervention, 'cohort': row['cohort'], 'replicate': row['replicate'],
                     'steps': proposal['budget_per_run']['actual_final_steps'], 'model': str(run / 'final-model.zip'), 'sha256': '0' * 64}
            (run / 'run-result.json').write_text(json.dumps(entry))
    runs = None
    import scripts.wp7_penalty_trial_runner as r
    from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS
    from hashlib import sha256
    originals = {name: sha256((ROOT / 'models' / name / 'model-final.zip').read_bytes()).hexdigest() for name in FROZEN_CHECKPOINTS}
    assert originals == FROZEN_CHECKPOINTS
    runs = runner.write_training_result(root=ROOT, proposal=proposal, output=output)
    assert [(r_['intervention'], r_['cohort'], r_['replicate']) for r_ in runs] == [
        (iv, row['cohort'], row['replicate']) for iv in runner.INTERVENTIONS for row in proposal['pairs']]
    assert len(json.loads((output / 'training-result.json').read_text())['runs']) == 12
