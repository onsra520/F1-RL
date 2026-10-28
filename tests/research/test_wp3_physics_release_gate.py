"""Physical observation invalid ≠ officially certified success/terminal."""
from copy import deepcopy

import pytest

from src.research.wp3_physics_release_gate import (
    check_snapshot,demand_full_evaluation_authorization
)

def test_all_three_real_frozen_cases_remain_invalid():
    report=check_snapshot()
    assert report["status"]=="BLOCKED_UNCERTIFIED_PHYSICS"
    assert report["replay_cases_certified"]==0
    assert report["replay_cases_required"]==3
    assert len(report["observed_invalid_scratch_cases"])==3
    assert {r["error_type"] for r in report["observed_invalid_scratch_cases"]}=={
      "SuspensionResolutionError","GroundedBodyCollisionUnscored"}
    assert all(r["certified_valid"] is False for r in report["observed_invalid_scratch_cases"])
    assert report["registered_eval_started"] is False
    assert report["new_eval_episodes"]==0
    assert report["scientific_evidence"] is False

def test_unapproved_5760_episode_eval_cannot_be_authorized():
    with pytest.raises(RuntimeError,match="FULL_5760_EVAL_BLOCKED"):
        demand_full_evaluation_authorization(check_snapshot())

@pytest.mark.parametrize(("field","value"),[
    ("status","PASS"),("replay_cases_certified",3),
    ("registered_eval_started",True),("release_allowed",True),
    ("scientific_evidence",True),
])
def test_forged_scoring_permission_remains_blocked(field,value):
    report=deepcopy(check_snapshot())
    report[field]=value
    with pytest.raises(RuntimeError,match="FULL_5760_EVAL_BLOCKED"):
        demand_full_evaluation_authorization(report)
