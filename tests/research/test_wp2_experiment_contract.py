import json
from pathlib import Path

import pytest

from f1rl.exploratory.abcd import (build_candidate_run_manifest, load_abcd_manifest,
                                   validate_abcd_manifest, verify_candidate_run_manifest,
                                   authorize_candidate_run)
from f1rl.exploratory.reward import validate as validate_demo
from f1rl.research.wp2_statistics import holm_adjust


ROOT = Path(__file__).resolve().parents[2]


def test_registered_abcd_manifest_is_strict_draft_and_uses_versioned_corridors():
    manifest = load_abcd_manifest(ROOT)
    assert manifest['status'] == 'DRAFT — NOT PREREGISTERED — NOT AUTHORIZED'
    assert manifest['evaluation']['hard_route_boundary_m'] == 4.0
    assert [(row['candidate_id'], row['lateral_penalty_per_m2_s'], row['demo_training']['training_corridor_m'])
            for row in manifest['candidates']] == [
                ('A', 2.0, 2.0), ('B', 4.0, 2.0), ('C', 2.0, 3.0), ('D', 4.0, 3.0)]
    assert validate_demo(manifest['candidates'][2]['demo_training'])['version'] == 'straight_route_stability_demo_v2'


def test_abcd_manifest_rejects_unregistered_factor_or_permission_escalation():
    manifest = load_abcd_manifest(ROOT)
    bad = json.loads(json.dumps(manifest))
    bad['candidates'][2]['training_corridor_m'] = 4.0
    with pytest.raises(ValueError):
        validate_abcd_manifest(bad)
    bad = json.loads(json.dumps(manifest))
    bad['status'] = 'APPROVED'
    with pytest.raises(ValueError):
        validate_abcd_manifest(bad)


def test_holm_adjustment_matches_known_three_test_example():
    assert holm_adjust((0.01, 0.04, 0.03)) == pytest.approx((0.03, 0.06, 0.06))


def test_prepared_run_manifest_binds_clean_source_candidate_and_statistical_design():
    run = build_candidate_run_manifest(ROOT, candidate_id='C', run_id='c-seed-100',
        seed_bundle={'ppo':100, 'sampling':200, 'training_reset':1100})
    binding = verify_candidate_run_manifest(ROOT, run)
    assert binding['candidate_id'] == 'C' and binding['split'] == 'TRAIN'
    assert binding['experiment_manifest_sha256'] == load_abcd_manifest(ROOT)['sha256']
    assert binding['run_manifest_sha256'] == run['sha256']
    bad = dict(run)
    bad['training_authorized'] = True
    from f1rl.research.wp7_protocol import digest
    bad['sha256'] = digest({key:value for key,value in bad.items() if key != 'sha256'})
    with pytest.raises(ValueError):
        verify_candidate_run_manifest(ROOT, bad)


def test_draft_run_cannot_reach_worker_authorization_or_create_replay_claim(tmp_path, monkeypatch):
    import f1rl.research.wp2_execution_permit as permit_module

    ledger = tmp_path / 'replay-ledger'
    monkeypatch.setattr(permit_module, '_replay_ledger_directory', lambda: ledger)
    run = build_candidate_run_manifest(ROOT, candidate_id='A', run_id='a-seed-100',
        seed_bundle={'ppo':100, 'sampling':200, 'training_reset':1100})
    with pytest.raises(ValueError):
        authorize_candidate_run(ROOT, run, {'payload':{}, 'signatures':[]})
    assert not ledger.exists()


def test_draft_manifest_is_rejected_before_repository_permit_claim(tmp_path, monkeypatch):
    import f1rl.research.wp2_execution_permit as permit_module

    def unexpected_claim(*args, **kwargs):
        raise AssertionError('draft manifest reached permit claim boundary')

    monkeypatch.setattr(permit_module, 'claim_repository_wp2_execution_permit', unexpected_claim)
    run = build_candidate_run_manifest(ROOT, candidate_id='C', run_id='c-seed-100',
        seed_bundle={'ppo':100, 'sampling':200, 'training_reset':1100})
    with pytest.raises(ValueError, match='approved preregistration'):
        authorize_candidate_run(ROOT, run, {'payload':{}, 'signatures':[]})


@pytest.mark.parametrize('values', [(), (0.1, float('nan')), (1.1,), (-0.1,)])
def test_holm_rejects_empty_or_invalid_p_values(values):
    with pytest.raises(ValueError):
        holm_adjust(values)
