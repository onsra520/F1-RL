"""WP3 frozen registry manifest cannot be edited or used as reviewer signature."""
from copy import deepcopy
from pathlib import Path
import json
import pytest

from f1rl.research.wp3_crash_contract_registration import assemble_registration
from f1rl.learning.crash_terminal_registered import _sha

ROOT=Path(__file__).resolve().parents[2]
HISTORY=Path(r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1")
REPLAY=ROOT/"docs/research/WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json"
RECEIPT=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt"
REGISTRY=ROOT/"docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json"


@pytest.fixture
def registry_payload_validation_only(monkeypatch):
    """Test-only lineage stub to reach all payload/seed/roster validations.

    Never used by training or production. The first test independently proves
    the *real* unrelated-source ancestry check remains fail-closed.
    """
    import subprocess
    from f1rl.research import wp3_crash_contract_registration as module

    real_run=module.subprocess.run

    def allow_only_unit_lineage(cmd,*args,**kwargs):
        if (
            isinstance(cmd,(tuple,list))
            and len(cmd)>=7
            and cmd[0]=="git"
            and cmd[3:5]==["merge-base","--is-ancestor"]
            and Path(cmd[2]).resolve()==ROOT.resolve()
        ):
            return subprocess.CompletedProcess(cmd,0,stdout=b"",stderr=b"")
        return real_run(cmd,*args,**kwargs)

    monkeypatch.setattr(module.subprocess,"run",allow_only_unit_lineage)


def test_unaltered_historical_payload_is_valid_only_in_test_scaffold(
    registry_payload_validation_only,
):
    observed=assemble_registration(ROOT,HISTORY,REPLAY,RECEIPT)
    historical=json.loads(REGISTRY.read_text("utf8"))
    assert observed["source_bundle_sha256"]!=historical["source_bundle_sha256"]
    assert observed["registered_original_checkpoint_count"]==12
    assert observed["engineering_regression_passed"]==352
    assert [r["registered_technical_result"] for r in observed["same_build_replay_cases"]]==[
        "valid_failed_crash","valid_finish","valid_failed_crash"]
    assert observed["software_contract_implemented"] is True
    assert observed["scientific_scoring_approved"] is False
    assert observed["independent_physics_review_signature"] is None
    assert observed["owner_scientific_signoff"] is False
    assert observed["full_5760_episode_evaluation_authorized"] is False
    assert observed["source_tree_clean"] is False

def test_historical_registry_remains_valid_but_not_current_source_approval():
    historical=json.loads(REGISTRY.read_text("utf8"))
    # Preserve the exact historical checksum and fail-closed scientific status.
    # The present owner handoff branch is NOT a descendant of the archived
    # WP3 authoring commit ce0e55d. It must not claim a valid registration
    # assembled against an unrelated source, even when all six original
    # historical evidence files have been restored byte-for-byte.
    assert historical["record_sha256"]==_sha({
        k:v for k,v in historical.items() if k!="record_sha256"})
    assert historical["scientific_scoring_approved"] is False
    assert historical["owner_scientific_signoff"] is False
    assert historical["full_5760_episode_evaluation_authorized"] is False
    with pytest.raises(ValueError,match="archived handoff source commit is not the current source ancestor"):
        assemble_registration(ROOT,HISTORY,REPLAY,RECEIPT)

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
def test_forged_scientific_or_model_seed_claim_rejected(tmp_path,registry_payload_validation_only,index,key,value):
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
def test_forged_registration_metadata_rejected(tmp_path,registry_payload_validation_only,key,value):
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
def test_partial_failed_or_forged_pytest_receipt_rejected(tmp_path,registry_payload_validation_only,content):
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
