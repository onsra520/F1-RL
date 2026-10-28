"""WP3 frozen registry manifest cannot be edited or used as reviewer signature."""
from copy import deepcopy
from pathlib import Path
import os
import json
import pytest

from src.validation.wp3_crash_contract_registration import assemble_registration
from src.rl.crash_terminal_registered import _sha

ROOT=Path(__file__).resolve().parents[2]
HISTORY=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
REPLAY=ROOT/"docs/research/WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json"
RECEIPT=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt"
REGISTRY=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json"

def test_historical_registry_remains_valid_but_not_current_source_approval():
    from src.validation.historical_source_records import verify_historical_records
    assert verify_historical_records(ROOT, HISTORY)["status"] == "HISTORICAL_RECORD_INTEGRITY_VERIFIED"
    with pytest.raises(ValueError, match="historical source identity"):
        assemble_registration(ROOT,HISTORY,REPLAY,RECEIPT)
    historical=json.loads(REGISTRY.read_text("utf8"))
    observed=historical
    # Historical contract receipt is immutable. The new v4 EDGE contact is
    # separately versioned and intentionally changes source bundle hashes.
    # Do not edit the historical receipt or fake current independent approval.
    assert historical["record_sha256"]==_sha({
        k:v for k,v in historical.items() if k!="record_sha256"})
    assert observed["source_bundle_sha256"]=="681482bc1a2817a439a0eb70f3bb51cc03129d5e212497a1ee433c67a7b0627e"
    assert observed["registered_original_checkpoint_count"]==12
    assert observed["engineering_regression_passed"]==352
    assert [r["registered_technical_result"] for r in observed["same_build_replay_cases"]]==[
       "valid_failed_crash","valid_finish","valid_failed_crash"]
    assert observed["software_contract_implemented"] is True
    assert observed["scientific_scoring_approved"] is False
    assert historical["scientific_scoring_approved"] is False
    assert observed["independent_physics_review_signature"] is None
    assert observed["owner_scientific_signoff"] is False
    assert observed["full_5760_episode_evaluation_authorized"] is False
    assert observed["source_tree_clean"] is False

@pytest.mark.parametrize(("index","key","value"),[
    (0,"failure_reason",None),
    (0,"success_flag_unaccepted",True),
    (0,"model_final_sha256","a"*64),
    (0,"scientific_evidence",True),
    (1,"env_seed",0),
    (1,"policy_seed",0),
    (1,"status","REGISTERED_SCIENCE_PASS"),
    (2,"failure_reason","physical_crash_development"),
])
def test_forged_scientific_or_model_seed_claim_rejected(tmp_path,index,key,value):
    entry=json.loads(REPLAY.read_text("utf8"))
    entry["cases"][index][key]=value
    path=tmp_path/"forged.json"
    path.write_text(json.dumps(entry),encoding="utf8")
    with pytest.raises(ValueError):
        assemble_registration(ROOT,HISTORY,path,RECEIPT)

@pytest.mark.parametrize(("key","value"),[
    ("scientific_signoff",True),
    ("full_5760_eval_admission","PASS"),
    ("registered_evaluation_executed",True),
    ("heldout_access",True),
    ("crash_scoring_contract","unapproved"),
    ("roster_sha256","f"*64),
])
def test_forged_registration_metadata_rejected(tmp_path,key,value):
    entry=json.loads(REPLAY.read_text("utf8"))
    entry[key]=value
    path=tmp_path/"forged.json"
    path.write_text(json.dumps(entry),encoding="utf8")
    with pytest.raises(ValueError):
        assemble_registration(ROOT,HISTORY,path,RECEIPT)

@pytest.mark.parametrize("content",[
    "352 passed, 1 failed in 20s\n",
    "1 passed in 20s\n",
    "352 passed in 20s\nTraceback\n",
    "352 passed in 20s\nERROR\n",
])
def test_partial_failed_or_forged_pytest_receipt_rejected(tmp_path,content):
    path=tmp_path/"test.txt"
    path.write_text(content,encoding="utf8")
    with pytest.raises(ValueError):
        assemble_registration(ROOT,HISTORY,REPLAY,path)

def test_registry_sha_covers_all_metadata():
    report=json.loads(REGISTRY.read_text("utf8"))
    assert report["record_sha256"]==_sha({k:v for k,v in report.items()
                                          if k!="record_sha256"})
    report["scientific_scoring_approved"]=True
    assert report["record_sha256"]!=_sha({k:v for k,v in report.items()
                                          if k!="record_sha256"})
