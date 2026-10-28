"""Current real frozen-case evidence and fail-closed scientific admission."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from src.research.wp3_current_eval_gate import (
    CURRENT, OLD, EXPECTED, require_latest_replay_evidence,
    require_registered_full_evaluation,
)


OLD = Path(os.environ.get("F1RL_WP3_EVIDENCE", str(OLD)))

@pytest.fixture(autouse=True)
def configured_replay_origin(monkeypatch):
    import src.research.wp3_current_eval_gate as gate_module
    original = gate_module.require_latest_replay_evidence
    monkeypatch.setattr(gate_module, 'require_latest_replay_evidence',
                        lambda: original(CURRENT, OLD))

def test_current_three_frozen_model_seed_witnesses_remain_invalid():
    gate=require_latest_replay_evidence(CURRENT, OLD)
    assert gate["status"]=="BLOCKED_UNCERTIFIED_PHYSICS"
    assert gate["frozen_model_seed_replays"]==3
    assert gate["physically_valid_replays"]==0
    assert gate["allowed_new_eval_episodes"]==0
    assert gate["registered_eval_approved"] is False
    with pytest.raises(RuntimeError,match="5760 EVAL BLOCKED"):
        require_registered_full_evaluation(gate)


@pytest.mark.parametrize("mutation",[
    lambda r:r["cases"][0].__setitem__("status","VALID_EPISODE"),
    lambda r:r["cases"][1].__setitem__("model_final_sha256","f"*64),
    lambda r:r["cases"][2].__setitem__("env_seed",0),
    lambda r:r["cases"][0].__setitem__("scientific_evidence",True),
    lambda r:r["cases"][0].__setitem__("error_type","CrashTerminal"),
    lambda r:r["cases"].reverse(),
    lambda r:r["cases"].pop(),
    lambda r:r.__setitem__("heldout_access",True),
    lambda r:r.__setitem__("registered_evaluation_executed",True),
    lambda r:r.__setitem__("full_5760_eval_admission","PASS"),
    lambda r:r.__setitem__("roster_sha256","0"*64),
    lambda r:r.__setitem__("physics_profile","spatial_rotating_owner_v3"),
])
def test_forged_outcome_model_seed_or_eval_claim_rejected(mutation,tmp_path):
    original=json.loads(CURRENT.read_text("utf8"))
    mutation(original)
    copy=tmp_path/"forged-unscored-replay.json"
    copy.write_text(json.dumps(original),encoding="utf8")
    with pytest.raises(ValueError):
        require_latest_replay_evidence(copy,OLD)


def test_cannot_promote_unchanged_blocked_replay_document_to_full_eval():
    result=require_latest_replay_evidence(CURRENT, OLD)
    with pytest.raises(ValueError):
        require_registered_full_evaluation({**result,"status":"PASS"})
