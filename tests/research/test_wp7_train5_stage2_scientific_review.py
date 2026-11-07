"""Read-only Stage-2 scientific proposal review; NOT a training admission."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import math

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]
PROPOSAL=ROOT/"configs/research/wp7-train5-stage2-science-proposal-v1.json"
REG=ROOT/"configs/research/wp7-train5-paired-v2-registration.json"
METRICS=ROOT/"configs/research/wp7-train5-v3-fixture-and-development-criteria.json"
PILOT=ROOT/"configs/recovery/wp7-pilot-200k.yml"
DEV_EVALUATOR=ROOT/"f1rl/research/wp7_train5_v3_dev_metrics.py"
SOURCE_ANCHOR="64b22bcf90d684836da30d2237ad1cefc94150ce"


def data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def test_stage2_is_source_bound_and_explicitly_not_an_approval():
    p=data(PROPOSAL)
    assert p["schema"]=="f1rl_wp7_train5_stage2_scientific_protocol_draft_and_critical_review_v1"
    assert p["source_anchor_commit"]==SOURCE_ANCHOR
    assert p["preparation_permission"]=="STAGE2_DOCUMENT_AND_TEST_REVIEW_ONLY"
    assert p["scientific_status"]=="DRAFT_CRITICAL_REVIEW_COMPLETE_NOT_OWNER_APPROVED"
    assert p["review_verdict"]=="PASS_DOCUMENT_COMPLETENESS_FAIL_SCIENTIFIC_TRAINING_ADMISSION"
    assert p["owner_protocol_signed"] is False
    assert p["independent_scientific_reviewer_signed"] is False
    assert p["training_permission"] is False
    assert p["heldout_permission"] is False
    assert p["scientific_release_permission"] is False
    assert p["admission_requirements"]["stage3_execution_admission"]=="NOT_GRANTED"
    assert all(p["permissions"][action] is False for action in (
        "ppo_training","ppo_finetuning","ppo_smoke_replay","heldout",
        "full06","scientific_release"
    ))


def test_all_science_source_bindings_use_canonical_git_contents():
    p=data(PROPOSAL)
    assert p["source_hash_contract"]=="SHA256_OF_LOCKED_GIT_BLOB_BYTES_CANONICAL_LF_CROSS_PLATFORM"
    references=p["source_bindings"]
    assert len(references)==10
    for path,digest in references.items():
        candidate=ROOT/path
        assert candidate.is_file()
        assert candidate.resolve().is_relative_to(ROOT)
        assert sha256(candidate.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==digest
    assert "f1rl/learning/algorithms/ppo.py" in references
    assert "f1rl/learning/training/runner.py" in references


def test_training_budget_rounding_conflict_is_detected_not_silently_accepted():
    p=data(PROPOSAL)["training_design"]
    cfg=yaml.safe_load(PILOT.read_text(encoding="utf8"))
    quantum=cfg["ppo"]["n_steps"]*cfg["vector_env"]["n_envs"]
    wanted=cfg["training"]["total_timesteps"]
    actual=math.ceil(wanted/quantum)*quantum
    assert (quantum,wanted,actual)==(2048,200000,200704)
    assert cfg["ppo"]["batch_size"]==64
    assert quantum % cfg["ppo"]["batch_size"]==0
    assert wanted % quantum!=0
    assert cfg["training"]["checkpoint_every_steps"] % quantum!=0
    assert p["rollout_quantum_steps"]==quantum
    assert p["rollouts_for_at_least_proposed_budget"]==98
    assert p["expected_realized_steps_if_full_rollouts"]==actual
    assert p["overshoot_per_policy"]==704
    assert p["overshoot_all_six"]==4224
    assert p["current_dev_metric_code_final_checkpoint_step"]==200000
    assert p["resolution_status"]=="OWNER_DECISION_AND_FUTURE_VERSIONED_IMPLEMENTATION_REQUIRED"
    assert p["step_budget_consistency"].startswith("BLOCKER_")
    assert 'row["checkpoint_step"]!=200000' in DEV_EVALUATOR.read_text("utf8")
    assert len(p["resolution_options"])==2


def test_development_thresholds_and_integer_budget_preserved_without_retuning():
    p=data(PROPOSAL)["development_design"]
    previous=data(METRICS)["scientific_quality_criteria_proposal"]
    original=previous["preproposal_thresholds"]
    assert previous["decision"]=="PROPOSED_NOT_APPROVED"
    assert p["preproposal_thresholds_unmodified"]==original
    assert p["track_ids"]==data(REG)["train_ids"]
    assert p["heldout_ids_metadata_only"]==data(REG)["heldout_ids"]
    assert len(p["track_ids"])==5
    assert len(p["conditions"])==2
    assert p["policy_count"]==6
    assert p["cell_count"]==60
    assert p["episodes_per_cell"]==60
    assert p["total_episodes"]==3600
    assert p["nominal_episodes"]==1800
    assert p["per_track_episodes"]==720
    assert p["per_cohort_episodes"]==1800
    assert p["integer_counts_for_exact_3600"]==[
      {"metric":"nominal_finish_min","minimum_successes":1620,"denominator":1800},
      {"metric":"per_track_finish_min","minimum_successes":576,"denominator":720},
      {"metric":"hard_route_violations_max","maximum_events":72,"denominator":3600},
      {"metric":"certified_physical_crashes_max","maximum_events":36,"denominator":3600},
      {"metric":"invalid_or_uncertified_physics_max","maximum_events":0,"denominator":3600},
    ]
    assert p["threshold_status"]=="EXPLORATORY_PROPOSAL_NOT_OWNER_SIGNED"
    assert p["power_justification"]=="MISSING_BEFORE_OWNER_ACCEPTANCE"
    assert p["dev_episode_execution"]=="NOT_EXECUTED"
    assert "NO_INDEPENDENCE_ASSUMPTION" in p["uncertainty_assessment"]
    assert len(p["required_preregistered_statistics"])>=6


def test_18_proposed_training_seeds_remain_distinct_untouched_and_exception_disclosed():
    d=data(PROPOSAL)["training_design"]
    reg=data(REG)
    roster=[seed for bundle in reg["proposed_training_seeds"].values() for row in bundle for seed in row.values()]
    assert len(roster)==len(set(roster))==18
    assert d["reserved_training_seed_count"]==18
    assert d["reserved_seeds_consumed"]==0
    assert d["missing_historical_GHA_logs"]==15
    assert d["global_nonuse_absolute_proof"] is False
    assert d["global_seed_registry_status"]=="CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION"
    assert reg["owner_approval"] is None
    assert reg["status"]=="draft_not_authorized"
    assert all(reg[k] is False for k in ("training_permission","smoke_permission","heldout_permission"))
    assert d["actual_training_run_status"]=="NOT_EXECUTED"


@pytest.mark.parametrize("key",(
    "registered_wp7_training","admitted_stochastic_ppo","independent_approval_restore"
))
def test_science_capability_does_not_unlock_from_documentation(key):
    p=data(PROPOSAL)
    assert key in p["admission_requirements"]["scientific_capabilities_blocked"]
    assert p["admission_requirements"]["compatibility_subsystems_open"]==9
    assert p["admission_requirements"]["scientific_training_permission"] is False
    assert p["admission_requirements"]["current_source_v2_admission_guard_unconditionally_closed"] is True


def test_synthetic_numeric_fixtures_cannot_impersonate_physical_calibration():
    p=data(PROPOSAL)
    f=p["fixture_provenance"]
    assert f["completed_resets"]==30 and f["simulation_steps"]==360
    assert f["physics_substeps"]==5760
    assert f["reserved_training_seeds_consumed"]==0
    assert f["heldout_accessed"] is False
    assert f["external_empirical_calibration_passed"] is False
    assert p["simulation_scope"]["owner_approved"] is False
    assert p["simulation_scope"]["empirical_calibration_measurements_available"] is False
    assert p["simulation_scope"]["preferred_claim"].startswith("SIMULATION_ONLY")
    assert p["training_reset"]["training_distribution_empirically_calibrated"] is False
    assert p["training_reset"]["training_ready_factory_activated"] is False


def test_review_traces_all_critical_risks_and_denies_new_training_permission():
    p=data(PROPOSAL)
    issues=p["risk_register"]
    assert [item["id"] for item in issues]==[
        "R1","R2","R3","R4","R5","R6","R7","R8"
    ]
    assert sum(item["severity"]=="BLOCKING" for item in issues)==5
    assert p["admission_requirements"]["owner_scientific_protocol_signature"] is False
    assert p["admission_requirements"]["registered_training_runner_verified"] is False
    assert p["admission_requirements"]["ppo_backend_verified_for_admitted_new_science"] is False
    for filename in (
        "WP7-TRAIN5-STAGE2-SCIENTIFIC-PROTOCOL-PROPOSAL-20261107.md",
        "WP7-TRAIN5-STAGE2-INDEPENDENT-DESK-REVIEW-20261107.md",
    ):
        page=(ROOT/"docs/.superpowers"/filename).read_text(encoding="utf8")
        assert "200,704" in page
        assert "scientific" in page.lower()
        assert "not" in page.lower()
