"""Stage-1 evidence admission: engineering audit only, NOT scientific training.

This test deliberately requires the authentic historical C2 DRAFT protocol
and the disclosed owner-accepted global seed evidence exception. A PASS is
permission to draft the Stage-2 protocol, never to train or read HELD-OUT.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import yaml

from f1rl.research.wp7_readiness import evaluate_readiness, load_readiness_matrix

ROOT = Path(__file__).resolve().parents[2]
C2 = ROOT / "configs/experiments/ppo-generalization-c2-v2.yml"
MATRIX = ROOT / "configs/research/wp7-core-readiness-matrix-v1.json"
REGISTRATION = ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
EXCEPTION = ROOT / "configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json"
# The exact Git blob from the authenticated Oct-13 C2 source history,
# SHA256 of LF bytes. CRLF checkout is normalised *only* for comparison.
HISTORICAL_C2_CANONICAL_SHA256 = (
    "a2cfa9bf4391069bc41ef391198da6a22dddece8bdd80b50a22a72c480ee4917"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_historical_c2_is_real_git_content_and_remains_draft_with_no_seeds():
    assert C2.is_file(), "Do not substitute a generated C2 contract"
    source = C2.read_bytes()
    assert sha256(source.replace(b"\r\n", b"\n")).hexdigest() == HISTORICAL_C2_CANONICAL_SHA256
    doc = yaml.safe_load(source)
    assert doc["experiment"] == {
        "id": "ppo-generalization-c2-v2", "protocol_version": 2, "status": "draft",
    }
    assert doc["training"]["protocol_status"] == "pilot_owned"
    assert doc["seeds"] == {"ppo": [], "sampling": []}
    assert doc["tracks"]["training_ids"] == [
        "straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"
    ]
    assert doc["tracks"]["heldout_ids"] == ["left-turn-v1", "offset-slalom-v1"]
    assert not any("evaluation/" in str(x) for x in doc["tracks"]["training_ids"])


def test_all_nine_subsystems_are_audited_without_fake_compatibility_pass():
    matrix = load_readiness_matrix(ROOT)
    assert MATRIX.is_file()
    assert len(matrix["subsystems"]) == 9
    capabilities = [c for s in matrix["subsystems"] for c in s["capabilities"]]
    assert len(capabilities) == 27
    assert sum(c["status"] == "implemented" for c in capabilities) == 24
    blocked = {c["id"] for c in capabilities if c["status"] == "blocked"}
    assert blocked == {
        "registered_wp7_training", "admitted_stochastic_ppo", "independent_approval_restore"
    }
    required = {
        "windows_linux_exact_head_ci",
        "historical_c2_replay",
        "cross_module_seed_and_json_schema",
    }
    for item in matrix["subsystems"]:
        assert item["compatibility"]["status"] == "in_progress"
        assert set(item["compatibility"]["required"]) == required
        assert sum(c["weight"] for c in item["capabilities"]) == 100
        assert all((ROOT / c["source"]).is_file() and (ROOT / c["test"]).is_file()
                   for c in item["capabilities"])
    readiness = evaluate_readiness(ROOT, verify=False)
    assert readiness["compatibility_complete"] is False
    assert readiness["scientific_acceptance"] == "NOT_EVALUATED"
    assert readiness["training_authorized"] is False


def test_signed_seed_facts_exception_is_valid_scope_not_absolute_nonuse():
    frozen = _read_json(REGISTRATION)
    accept = _read_json(EXCEPTION)
    cohorts = frozen["proposed_training_seeds"]
    values = [int(v) for group in cohorts.values() for triple in group for v in triple.values()]
    assert len(values) == 18 and len(set(values)) == 18
    assert sorted(values) == sorted(accept["candidate_seeds"])
    assert accept["seed_count"] == 18
    assert accept["registry_gate_status"] == "CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION"
    assert accept["global_seed_registry_accepted_with_exception"] is True
    assert accept["universal_global_seed_nonuse_independently_proven"] is False
    assert accept["exception"]["affected_run_count"] == 15
    assert accept["exception"]["job_logs_not_recovered"] is True
    assert accept["exception"]["all_15_runs_are_accepted_as_PASS"] is False
    owner = accept["owner_approval"]
    assert owner["owner_seed_facts_SSH_signed_and_verified"] is True
    assert owner["owner_no_offmachine_facts_SSH_signed_and_verified"] is True
    assert owner["decision_separately_SSH_signed"] is False


def test_stage1_is_strictly_non_execution_and_never_unlocks_stage2_training():
    proposal = _read_json(REGISTRATION)
    registry = _read_json(EXCEPTION)
    assert proposal["status"] == "draft_not_authorized"
    assert proposal["owner_approval"] is None
    assert proposal["seed_audit_status"] == "PROPOSED_UNVERIFIED_GLOBAL"
    assert all(proposal[k] is False for k in (
        "training_permission", "smoke_permission", "heldout_permission"
    ))
    assert all(registry[k] is False for k in (
        "ppo_training_permission", "ppo_smoke_permission",
        "heldout_policy_evaluation_permission", "scientific_release_permission",
        "full06_scientific_evaluation_permission"
    ))
    assert registry["training_scenario_and_protocol_owner_approval"] == (
        "NOT_GRANTED_SEPARATE_REVIEW_REQUIRED"
    )
