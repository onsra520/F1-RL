"""Portable, no-training checks for the owner-approved v2 main task closeout.

This verifies truthful owner *in-chat* authorization and documents missing
wall-clock duration; it never converts chat approval to signed/scientific
approval or admits HELD-OUT/PPO training.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
CLOSE=ROOT/"docs/.superpowers/physics-v4-final-closeout-20261107"
P=ROOT/"docs/.superpowers/PHYSICS-V4-LEFT-TURN-PLAN.md"
REG=ROOT/"configs/research/wp7-train5-paired-v2-registration.json"


def load(name):
    return json.loads((CLOSE/name).read_text(encoding="utf8"))


def test_owner_approval_is_scoped_to_versioned_v2_not_a_signed_science_gate():
    obj=load("OWNER-SCOPE-APPROVAL.json")
    assert obj["schema"]=="f1rl_physics_v4_main_task_owner_chat_scope_amendment_v1"
    assert obj["owner_direction"].startswith("Owner explicitly instructed")
    assert obj["scope_change_owner_approved_in_chat"] is True
    assert obj["authorization_channel"]=="IN_CHAT_OWNER_INSTRUCTION"
    assert obj["authorization_scope"]=="ENGINEERING_MAIN_TASK_0_TO_6_ONLY"
    assert obj["cryptographic_owner_signature_present"] is False
    assert obj["independent_reviewer_signature_present"] is False
    assert obj["scientific_training_admission_approved"] is False
    assert obj["historical_v1_acceptance_relabelled_as_pass"] is False
    assert obj["preexisting_original_v1_literal_completion"] is False
    assert obj["original_plan_historical_status"]=="DRAFT_AWAITING_APPROVAL"
    # The archived plan is Windows-CRLF in the owner receipt, but Git checks it
    # out as LF on Linux. Require BOTH identities, never edit the old plan.
    canonical=P.read_bytes().replace(b"\r\n",b"\n")
    assert sha256(canonical).hexdigest()=="7bad13b62174c5e8c632f24ee524df8ebb3277bc12e0427a3489eb5474354769"
    assert sha256(canonical.replace(b"\n",b"\r\n")).hexdigest()==obj["original_plan_sha256"]


def test_six_locked_smoke_entries_normalized_without_invented_duration():
    src=load("SMOKE-DURATION-DISCLOSURE.json")
    assert src["schema"]=="f1rl_physics_v4_frozen_ppo_smoke_duration_disclosure_v1"
    assert src["original_one_shot_source_commit"]=="01ef1e3e75de8df60fc2159ed70b38d0ee777253"
    assert src["current_closeout_parent_source_commit"]=="d0e61d54232c6e4d92a7ab2484ebf7a8f1c5040d"
    assert src["reset_seed"]==20261106
    assert src["finished"]==6
    assert src["total_steps"]==1117
    assert src["time_measurement"]=="NOT_RECORDED_FOR_ALL_SIX_EPISODES"
    assert src["duration_values_invented"] is False
    assert src["duration_from_simulated_steps_inferred"] is False
    assert src["smoke_rerun"] is False
    cases=src["episodes"]
    assert len(cases)==6
    expected={
      ("C-seed0","right-turn-train-v2"):203,
      ("C-seed0","left-turn-train-v2"):205,
      ("B-seed1","right-turn-train-v2"):177,
      ("B-seed1","left-turn-train-v2"):177,
      ("A-seed2","right-turn-train-v2"):179,
      ("A-seed2","left-turn-train-v2"):176,
    }
    assert {(x["model"],x["track"]):x["steps"] for x in cases}==expected
    assert all(x["duration_seconds"] is None and
               x["duration_status"]=="NOT_RECORDED" and
               x["duration_is_measured"] is False
               for x in cases)
    assert all(x["checkpoint_order"]==["checkpoint-0","checkpoint-1","finish"]
               and x["result"]=="FINISH" for x in cases)
    assert all(len(x["source_log_sha256"])==64 and
               set(x["source_log_sha256"])<=(set("0123456789abcdef"))
               for x in cases)


def test_exact_final_report_filename_is_present_in_new_source_subdirectory():
    report=(CLOSE/"FINAL-REPORT.md").read_text(encoding="utf8")
    assert report.startswith("# FINAL-REPORT")
    assert "Owner-authorized paired TRAIN v2" in report
    assert "NOT_RECORDED" in report
    assert "original literal v1" in report
    assert "not" in report.lower()
    assert "no cryptographic owner signature" in report.lower()
    assert "independent" in report.lower()
    assert "1,570 distinct" in report
    assert "123/123 PASS" in report
    for model in ("C-seed0","B-seed1","A-seed2"):
        assert model in report
    assert "training" in report.lower() and "not" in report.lower()
    assert "held-out" in report.lower()


def test_original_draft_remains_historical_and_source_gates_closed():
    plan=P.read_text(encoding="utf8")
    assert "Status: DRAFT" in plan
    assert "left-turn-train-v1" in plan
    approval=load("OWNER-SCOPE-APPROVAL.json")
    for key in ("training_permission","heldout_permission","full06_permission","science_release_permission"):
        assert approval[key] is False
    reg=json.loads(REG.read_text("utf8"))
    assert reg["status"]=="draft_not_authorized"
    assert reg["training_permission"] is False
    assert reg["heldout_permission"] is False
    assert reg["smoke_permission"] is False


def test_decision_separates_amendment_from_missing_measurement_and_v1_block():
    approval=load("OWNER-SCOPE-APPROVAL.json")
    changes=approval["changes"]
    assert all(k in changes for k in ("task2","task3","task5","task6"))
    assert "HELD-OUT" in changes["task2"]
    assert "NOT_RECORDED" in changes["task5"]
    assert "no" in changes["task5"].lower()
    assert "historical file" in changes["task6"]
    assert "2026-11-06" in (CLOSE/"FINAL-REPORT.md").read_text(encoding="utf8")
    assert approval["missing_duration_measured_or_backfilled"] is False
    amendment=(CLOSE/"OWNER-SCOPE-AMENDMENT.md").read_text("utf8")
    assert "owner" in amendment.lower() and "engineering" in amendment.lower()
    assert "not a cryptographically signed" in amendment
    assert "NOT_RECORDED" in amendment
    assert "retraining" in amendment.lower()
