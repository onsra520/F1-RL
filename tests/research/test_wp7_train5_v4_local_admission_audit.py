"""Independent local safety audit of TRAIN-5 draft; no training execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from f1rl.research.wp7_train5_v4_local_audit import (
    AuditError, build_local_audit, load_local_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_local_audit_discloses_all_actual_remaining_blockers():
    sources = load_local_inputs(ROOT)
    r = build_local_audit(ROOT, sources)
    assert r["matrix"]["capability_total"] == 27
    assert r["matrix"]["implemented"] == 24
    assert r["matrix"]["blocked"] == 3
    assert r["matrix"]["compatibility_open"] == 9
    assert r["matrix"]["all_source_and_test_paths_exist"] is True
    assert r["seeds"]["registered_count"] == 18
    assert r["seeds"]["distinct_count"] == 18
    assert r["seeds"]["missing_gha_job_logs"] == 15
    assert r["seeds"]["accepted_exception"] is True
    assert r["seeds"]["absolute_nonuse_proven"] is False
    assert r["admission"]["ready"] is False
    assert r["admission"]["ppo_training_permitted"] is False
    assert r["admission"]["heldout_permitted"] is False


def test_source_paths_are_checked_not_assumed():
    p = load_local_inputs(ROOT)
    p["matrix"]["subsystems"][0]["capabilities"][0]["source"] = "f1rl/does-not-exist.py"
    r = build_local_audit(ROOT, p)
    assert r["matrix"]["all_source_and_test_paths_exist"] is False
    assert r["admission"]["ready"] is False


def test_duplicate_seed_fails_closed():
    p = load_local_inputs(ROOT)
    p["registry"]["candidate_seeds"][1] = p["registry"]["candidate_seeds"][0]
    with pytest.raises(AuditError, match="duplicate"):
        build_local_audit(ROOT, p)


def test_status_flip_without_evidence_fails_closed():
    p = load_local_inputs(ROOT)
    p["matrix"]["subsystems"][0]["compatibility"]["status"] = "pass"
    with pytest.raises(AuditError, match="unverified"):
        build_local_audit(ROOT, p)


def test_seed_exception_cannot_be_silently_removed():
    p = load_local_inputs(ROOT)
    p["registry"]["exception"]["job_logs_not_recovered"] = False
    with pytest.raises(AuditError, match="historical"):
        build_local_audit(ROOT, p)
