from pathlib import Path
import pytest
from scripts.wp7_penalty_trial_protocol import build_proposal, require_execution_capability, validate_proposal

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT.parent / 'slalom-fresh-audit.json').is_file() or not (ROOT.parent / 'TRAIN5-v4-internal-20261109' / 'evaluation-journal.jsonl').is_file(), reason='local trial seed history is not available')


def test_paired_roster_budget_and_non_authorization():
    proposal = build_proposal(ROOT)
    assert proposal['execution_permitted'] is False and proposal['heldout_permitted'] is False
    assert proposal['training_runs'] == 12 and proposal['total_actual_training_steps'] == 2408448
    assert len(proposal['pairs']) == 6 and len(proposal['evaluation_seed_rows']) == 3600
    assert proposal['evaluation_episodes'] == 7200
    assert proposal['interventions'] == {'baseline': 0, 'terminal_crash_penalty': 5}
    assert proposal['track_sampling'] == 'unchanged_uniform'


def test_no_envelope_can_enable_unimplemented_runner():
    for envelope in (None, {}, {'signature_hex': '00' * 64}):
        with pytest.raises(ValueError, match='not implemented'):
            require_execution_capability(envelope)


@pytest.mark.parametrize('field,value', [('execution_permitted', True), ('training_runs', 6), ('heldout_permitted', True)])
def test_modified_scope_fails_closed(field, value):
    proposal = build_proposal(ROOT)
    proposal[field] = value
    with pytest.raises(ValueError, match='proposal or source bindings changed'):
        validate_proposal(ROOT, proposal)
