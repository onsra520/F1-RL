"""Current V4V replay admission separates engineering crash from Eval success."""
import json
import pytest
from f1rl.research.wp3_issue51_latest_gate import LATEST,OLD,source_bound_dev_crash_gate


def test_latest_replay_does_not_mint_scientific_approval():
    x=source_bound_dev_crash_gate()
    assert x["status"]=="BLOCKED"
    assert x["real_frozen_replay_cases"]==3
    assert x["development_only_completed_crash"]==1
    assert x["invalid_road_or_ramp_cases"]==2
    assert x["scientifically_certified_cases"]==0
    assert x["full_eval_episodes_started"]==0

@pytest.mark.parametrize("section,index,field,value",[
    ("row",0,"status","ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED"),
    ("row",1,"env_seed",0),
    ("row",2,"scientific_evidence",True),
    ("row",2,"failure_reason","success"),
    ("row",2,"model_final_sha256","f"*64),
    ("document",None,"full_5760_eval_admission","PASS"),
    ("document",None,"registered_evaluation_executed",True),
    ("document",None,"heldout_access",True),
])
def test_modified_scientific_identity_cannot_pass(section,index,field,value,tmp_path):
    raw=json.loads(LATEST.read_text("utf8"))
    target=raw if section=="document" else raw["cases"][index]
    target[field]=value
    path=tmp_path/"forged.json"
    path.write_text(json.dumps(raw),encoding="utf8")
    with pytest.raises(ValueError):
        source_bound_dev_crash_gate(path,OLD)
