"""Owner-accepted GSR exception is not training or HELD-OUT admission."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from scripts.wp7_train5_handoff_preflight import audit as source_preflight

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
ACCEPTANCE = ROOT / "configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json"
BRIDGE = ROOT / "configs/research/wp7-train5-paired-v2-git-blob-lock.json"
HEAD_ANCHOR = "e8c15965431465e8c97b73bfe2be30fd261875ff"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def test_exception_close_is_explicit_and_not_a_claim_of_recovered_logs():
    report = load(ACCEPTANCE)
    assert report["schema"] == "f1rl_train5_global_seed_registry_owner_exception_acceptance_v1"
    assert report["source_anchor_commit"] == HEAD_ANCHOR
    assert report["registry_gate_status"] == "CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION"
    assert report["global_seed_registry_accepted_with_exception"] is True
    assert report["universal_global_seed_nonuse_independently_proven"] is False
    exception = report["exception"]
    assert exception["id"] == "F1RL-TRAIN5-GSR-EX-001"
    assert exception["affected_run_count"] == 15
    assert exception["job_logs_not_recovered"] is True
    assert exception["all_15_runs_are_accepted_as_PASS"] is False
    assert exception["archived_workflow_sources_verified_count"] == 15
    assert exception["workflow_source_unique_commit_count"] == 8
    assert exception["workflow_definitions_unique_hashes"] == 2
    assert exception["workflows_containing_any_proposed_seed_literal"] == 0


def test_owner_decision_is_explicit_but_not_falsely_ssh_signed():
    report = load(ACCEPTANCE)
    decision = report["owner_approval"]
    assert decision["source"] == "EXPLICIT_OWNER_APPROVAL_IN_CURRENT_CHAT_2026-11-07"
    assert decision["decision_separately_SSH_signed"] is False
    assert decision["owner_seed_facts_SSH_signed_and_verified"] is True
    assert decision["owner_no_offmachine_facts_SSH_signed_and_verified"] is True
    text = decision["verbatim_text"]
    assert "15 log GitHub Actions" in text
    assert "không đồng nghĩa" in text
    assert "phải được phê duyệt riêng" in text
    assert sha256(text.encode("utf8")).hexdigest() == decision["verbatim_utf8_sha256"]


def test_all_18_seeds_match_frozen_draft_registration():
    report = load(ACCEPTANCE)
    draft = load(REGISTRATION)
    values = sorted({
        value
        for cohort in draft["proposed_training_seeds"].values()
        for bundle in cohort
        for value in bundle.values()
    })
    assert len(values) == len(set(values)) == 18
    assert all(type(x) is int and 0 <= x < 2**32 for x in values)
    assert report["candidate_seeds"] == values
    assert report["seed_count"] == 18
    bridge = load(BRIDGE)
    assert bridge["registration_sha256"] == report["source_provenance"]["registration_sha256"]
    assert report["source_provenance"]["registration_status_at_acceptance"] == "draft_not_authorized"
    assert report["source_provenance"]["global_seed_registry_acceptance_file_is_separate_from_frozen_registration"] is True


def test_acceptance_does_not_authorize_any_experiment_or_release():
    report = load(ACCEPTANCE)
    draft = load(REGISTRATION)
    for field in (
        "ppo_smoke_permission",
        "ppo_training_permission",
        "heldout_policy_evaluation_permission",
        "full06_scientific_evaluation_permission",
        "scientific_release_permission",
    ):
        assert report[field] is False
    assert report["training_scenario_and_protocol_owner_approval"] == "NOT_GRANTED_SEPARATE_REVIEW_REQUIRED"
    assert draft["status"] == "draft_not_authorized"
    assert draft["seed_audit_status"] == "PROPOSED_UNVERIFIED_GLOBAL"
    assert draft["global_seed_registry_receipt"] is None
    assert draft["owner_approval"] is None
    assert draft["training_permission"] is False
    assert draft["smoke_permission"] is False
    assert draft["heldout_permission"] is False


def test_original_frozen_source_only_preflight_still_passes():
    result = source_preflight(ROOT)
    assert result["result"] == "PASS_SOURCE_LOCK_NO_TRAINING"
    assert result["source_file_git_blob_hashes_verified"] == 44
    assert result["proposed_seed_count"] == 18
    assert result["training_permission"] is False
    assert result["scientific_release_permission"] is False
