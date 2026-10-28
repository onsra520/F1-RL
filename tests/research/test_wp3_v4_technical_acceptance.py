"""Adversarial tests for owner-UNSIGNED physics v4 technical acceptance."""
import json
from pathlib import Path
import os
import pytest

from f1rl.research.wp3_v4_technical_acceptance import (
    source_bound_acceptance,SCHEMA,VERSION
)

ROOT=Path(__file__).resolve().parents[2]
HISTORY=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
REPLAY=ROOT/"docs/research/WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json"
RECEIPT=ROOT/"docs/research/WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt"


def check(replay=REPLAY,receipt=RECEIPT):
    return source_bound_acceptance(ROOT,HISTORY,replay,test_receipt_file=receipt)


def test_technical_evidence_can_be_source_bound_without_fake_reviewer():
    from f1rl.research.historical_source_records import verify_historical_records
    assert verify_historical_records(ROOT, HISTORY)["status"] == "HISTORICAL_RECORD_INTEGRITY_VERIFIED"
    with pytest.raises(ValueError, match="historical source identity"):
        check()
    p=json.loads((ROOT/"docs/research/WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json").read_bytes())
    assert p["schema_version"]==SCHEMA
    assert p["engineering_test_status"]=="PASS"
    assert p["focused_physics_evaluator_tests_passed"]==300
    assert p["focused_physics_evaluator_tests_failed"]==0
    assert p["registered_final_model_count"]==12
    assert p["engineering_replay_3_of_3_completed"] is True
    assert len(p["three_exact_replays"])==3
    assert [x["classification"] for x in p["three_exact_replays"]]==[
        "DEV_VALID_FAILED_CRASH","DEV_FINISH","DEV_VALID_FAILED_CRASH"]
    assert p["physics_contract"]==VERSION
    assert all(p["version_guards"].values())
    assert p["source_tree_clean"] is False
    assert p["independent_domain_physics_review_completed"] is False
    assert p["independent_reviewer_signature"] is None
    assert p["owner_scientific_physics_signoff"] is False
    assert p["approved_physical_scoring_contract"] is False
    assert p["production_and_registered_eval_admitted"] is False
    assert p["full_5760_eval_started"] is False
    assert p["overall_status"]=="TECHNICAL_GATE_PASS_SCIENTIFIC_SIGNOFF_PENDING"


@pytest.mark.parametrize(("index","field","value"),[
    (0,"model_final_sha256","a"*64),
    (0,"env_seed",-1),
    (1,"policy_seed",-1),
    (1,"success_flag_unaccepted",False),
    (1,"failure_reason","physical_crash_development"),
    (2,"status","VALID_OFFICIAL_EVALUATION_EPISODE"),
    (2,"scientific_evidence",True),
    (2,"model_id","C-seed1"),
])
def test_forged_model_seed_or_scientific_result_is_rejected(tmp_path,index,field,value):
    report=json.loads(REPLAY.read_text("utf8"))
    report["cases"][index][field]=value
    path=tmp_path/"modified-replay.json"
    path.write_text(json.dumps(report),encoding="utf8")
    with pytest.raises(ValueError):
        check(replay=path)


@pytest.mark.parametrize(("field","value"),[
    ("full_5760_eval_admission","PASS"),
    ("all_required_replays_scientifically_certified",True),
    ("registered_evaluation_executed",True),
    ("heldout_access",True),
    ("physics_profile","spatial_rotating_owner_v3"),
    ("roster_sha256","0"*64),
])
def test_science_registration_and_replay_provenance_escalations_rejected(tmp_path,field,value):
    report=json.loads(REPLAY.read_text("utf8"))
    report[field]=value
    path=tmp_path/"modified-replay.json"
    path.write_text(json.dumps(report),encoding="utf8")
    with pytest.raises(ValueError):
        check(replay=path)


@pytest.mark.parametrize("content",[
    "300 passed, 2 failed in 32s\n",
    "299 passed in 32s\n",
    "300 passed in 32s\nTraceback (most recent call last):\n",
    "302 passed in 32s\nERROR: unresolved physics\n",
    "no test report\n"
])
def test_forged_failed_or_partial_test_receipt_cannot_trigger_acceptance(tmp_path,content):
    path=tmp_path/"pytest.txt"
    path.write_text(content,encoding="utf8")
    with pytest.raises(ValueError):
        check(receipt=path)
