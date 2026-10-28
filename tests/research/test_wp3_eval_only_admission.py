"""Issue #51 Eval-only immutable input admission and negative tests."""
from copy import deepcopy
from pathlib import Path
import os
import json

import pytest

from f1rl.research.wp3_eval_only_admission import (
    _roster_check,build_eval_only_preflight,digest,refuse_full_eval,
    MODELS,TRACKS,STRATA,REPLAY_CASES,
)

ROOT=Path(__file__).resolve().parents[2]
HISTORY=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
REPLAY=ROOT/"docs/research/WP3-CE0E55D-THREE-REPLAY-PREFLIGHT-20261024.json"
GATE=ROOT/"docs/research/WP3-EVAL-ONLY-COMPLETION-GATE-20261024.json"


def originals():
    return (json.loads((HISTORY/"evaluation/evaluation-roster.json").read_text("utf8")),
            json.loads((HISTORY/"WP3-EVAL-SOURCE-PROVENANCE.json").read_text("utf8")))


def rehash(roster):
    roster["sha256"]=digest({k:v for k,v in roster.items() if k!="sha256"})
    return roster


def test_original_paired_96_cell_roster_is_valid_exact_60_slots():
    r,p=originals()
    _roster_check(r,p)
    assert r["expected_cells"]==12*4*2
    assert r["expected_episodes"]==5760
    assert len(MODELS)==12 and len(TRACKS)==4 and len(STRATA)==2
    assert len(r["seed_rows"])==96


@pytest.mark.parametrize("change",[
    lambda r:r["seed_rows"][0]["episodes"][0].__setitem__("environment_seed",4),
    lambda r:r["seed_rows"][0]["episodes"][0].__setitem__("episode",59),
    lambda r:r["seed_rows"][0].__setitem__("stratum","heldout"),
    lambda r:r["seed_rows"][0].__setitem__("seed_index",999),
    lambda r:r["seed_rows"][0].__setitem__("model_id","A-seed0"),
    lambda r:r["seed_rows"].pop(),
    lambda r:r["seed_rows"][0]["episodes"].pop(),
    lambda r:r["evaluation"].__setitem__("hard_route_boundary_m",5.),
    lambda r:r["evaluation"].__setitem__("policy_actions","deterministic"),
    lambda r:r["evaluation"].__setitem__("heldout_access",True),
    lambda r:r["evaluation"].__setitem__("checkpoint_selection","best_seen"),
])
def test_recomputed_roster_sha_still_rejects_unsafe_changes(change):
    r,p=originals()
    change(r)
    rehash(r)
    p["evaluation_roster_sha256"]=r["sha256"]
    with pytest.raises(ValueError):
        _roster_check(r,p)


def test_replayed_three_frozen_episodes_still_physically_invalid():
    x=json.loads(REPLAY.read_text("utf8"))
    assert x["full_eval_gate"]=="BLOCKED"
    found={(r["model_id"],r["track"],r["stratum"],r["episode"]):r for r in x["results"]}
    assert set(found)==set(REPLAY_CASES)
    assert [found[c]["error_type"] for c in REPLAY_CASES]==[
        "SuspensionResolutionError","SuspensionResolutionError","FlightPhysicsError"
    ]
    assert x["registered_eval_started"] is False if "registered_eval_started" in x else x["new_registered_eval_runs"]==0


def test_12_real_frozen_final_zip_and_telemetry_sha_verified():
    result=build_eval_only_preflight(ROOT,HISTORY,REPLAY)
    assert result["frozen_final_checkpoint_count"]==12
    assert len(result["verified_models"])==12
    assert result["checkpoint_validation"]=="12_MODELS_ALL_SHA256_MATCH"
    # This historical v3 preflight is read-only; the source can now be a
    # locked clean descendant of the original ce0e55d authoring commit.
    from subprocess import check_output
    currently_dirty=bool(check_output(["git","-C",str(ROOT),"status","--porcelain"],
                                      text=True).strip())
    assert result["source_dirty"] is currently_dirty
    assert result["handoff_source_commit"]=="ce0e55d7a428826b2d4ed3218641105bd0206f4b"
    assert result["expected_cells"]==96 and result["expected_episodes"]==5760
    assert result["registered_eval_started"] is False
    assert result["full_eval_execution_authorized"] is False
    assert result["new_episodes_executed_by_preflight"]==0
    assert result["mandatory_replay_gate"]=="BLOCKED"
    assert len(result["mandatory_replay_cases"])==3
    with pytest.raises(RuntimeError,match="FULL_EVAL_BLOCKED"):
        refuse_full_eval(result)


def test_preflight_sha_and_finish_gates_are_fail_closed():
    plan=json.loads(GATE.read_text("utf8"))
    assert plan["sha256"]==digest({k:v for k,v in plan.items() if k!="sha256"})
    with pytest.raises(RuntimeError,match="FULL_EVAL_BLOCKED"):
        refuse_full_eval(plan)
    plan["sha256"]="0"*64
    with pytest.raises(ValueError,match="checksum"):
        refuse_full_eval(plan)


def test_historical_source_files_remain_identical_to_referenced_hashes():
    from f1rl.research.wp3_eval_only_admission import hash_file
    p=json.loads(GATE.read_text("utf8"))
    assert hash_file(HISTORY/"evaluation/evaluation-roster.json")==p["historical_roster_file_sha256"]
    assert hash_file(HISTORY/"WP3-EVAL-SOURCE-PROVENANCE.json")==p["historical_eval_provenance_sha256"]
