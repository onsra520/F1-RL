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


def test_new_seed_domain_does_not_reuse_interrupted_trial():
    from hashlib import sha256

    proposal = build_proposal(ROOT)
    def old_seed(*parts):
        value = '/'.join(('F1RL-TERMINAL-PENALTY-TRIAL-20261111-v1', *map(str, parts)))
        return int.from_bytes(sha256(value.encode('ascii')).digest()[:4], 'big')

    previous, current = set(), set()
    for pair in proposal['pairs']:
        old = {role: old_seed('train', pair['cohort'], pair['replicate'], role)
               for role in ('ppo', 'sampling', 'perturbation')}
        previous.update(old.values())
        previous.update(range(old['ppo'], old['ppo'] + 8))
        current.update(pair['seeds'].values())
        current.update(range(pair['seeds']['ppo'], pair['seeds']['ppo'] + 8))
    for row in proposal['evaluation_seed_rows']:
        previous.update((old_seed('env', *row[:-2]), old_seed('policy', *row[:-2])))
        current.update(row[-2:])
    assert previous.isdisjoint(current)


def test_v3_seeds_are_disjoint_from_retired_v1_and_v2_domains():
    from scripts.wp7_penalty_trial_protocol import RETIRED_DOMAINS, DOMAIN, _seed_in, retired_domain_seeds

    assert DOMAIN == 'F1RL-TERMINAL-PENALTY-TRIAL-20261111-v3'
    assert RETIRED_DOMAINS == ('F1RL-TERMINAL-PENALTY-TRIAL-20261111-v1', 'F1RL-TERMINAL-PENALTY-TRIAL-20261111-v2')
    proposal = build_proposal(ROOT)
    assert proposal['seed_domain'] == DOMAIN
    assert proposal['retired_seed_domains'] == list(RETIRED_DOMAINS)
    current = set()
    for pair in proposal['pairs']:
        current.update(pair['seeds'].values())
        current.update(range(pair['seeds']['ppo'], pair['seeds']['ppo'] + 8))
    for row in proposal['evaluation_seed_rows']:
        current.update(row[-2:])
    assert current.isdisjoint(retired_domain_seeds(proposal['tracks']))
    assert len(proposal['evaluation_seed_rows']) == 3600 and proposal['training_runs'] == 12
