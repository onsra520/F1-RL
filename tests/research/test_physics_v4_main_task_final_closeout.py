"""Evidence-preserving closeout checks for original Physics-v4 Task 0-6.

No PPO execution, training, HELD-OUT geometry or original source mutation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from f1rl.research.wp7_train5_v2_protocol import require_training_admission

ROOT=Path(__file__).resolve().parents[2]
VERDICT=ROOT/"docs/.superpowers/PHYSICS-V4-MAIN-TASK-FINAL-VERDICT-20261107.json"
REPORT=ROOT/"docs/.superpowers/PHYSICS-V4-MAIN-TASK-FINAL-ACCEPTANCE-20261107.md"
DRAFT=ROOT/"configs/research/wp7-train5-paired-v2-registration.json"
REGISTRY=ROOT/"configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def test_seven_tasks_recorded_with_real_v1_collision_not_false_100_percent():
    data=read(VERDICT)
    assert data["schema"]=="f1rl_physics_v4_main_task_final_scope_disposition_v1"
    assert data["tested_source_anchor"]=="f1b9ae4d2a2dc16a7ecd9698ff31705f01c97932"
    assert data["historical_baseline"]=="2417570f32f6879c2651bce37d88fd6ac83a2ea4"
    assert data["overall"]=="SEVEN_ENGINEERING_TASKS_DISPOSITIONED_ON_VERSIONED_V2_AMENDMENT"
    assert data["literal_original_mirror_v1_100_percent"] is False
    assert data["v1_heldout_metadata_collision"] is True
    assert data["versioned_v2_replacement_accepted_engineering"] is True
    assert set(data["tasks"])=={str(x) for x in range(7)}
    assert all(v.startswith("PASS") for v in data["tasks"].values())


def test_two_independent_nonppo_witnesses_and_six_frozen_smoke_results():
    data=read(VERDICT)
    for side in ("right_v2_nonppo","left_v2_nonppo"):
        witness=data[side]
        assert witness=={"steps":497,"gates":[113,311,497],"replay":True}
    assert len(data["smoke_cases"])==6
    assert sum(row["steps"] for row in data["smoke_cases"].values())==1117
    for model in ("C-seed0","B-seed1","A-seed2"):
        for track in ("right-turn-train-v2","left-turn-train-v2"):
            row=data["smoke_cases"][f"{model}/{track}"]
            assert row["terminal"]=="finished"
            assert len(row["sha256"])==64
    assert data["direction_retraining"]["old_right_v1"].startswith("BLOCKED")
    assert data["direction_retraining"]["mirror_left_v1"].startswith("NOT_ADMISSIBLE")
    assert data["direction_retraining"]["right_v2"].startswith("NO_AUTOMATIC_RETRAINING")
    assert data["direction_retraining"]["left_v2"].startswith("NO_AUTOMATIC_RETRAINING")


def test_original_train4_evidence_and_scoped_regression_are_precise():
    data=read(VERDICT)
    assert data["frozen_source_asset_model_count"]==19
    assert data["historical_handoff_sha256_entries_verified"]==30
    assert data["historical_full06"]=={"tracks":4,"cells":96,"episodes":5760,"rerun":False}
    observed=data["test_groups"]
    for k,n in (("original",73),("targeted_and_legacy",85),
                ("scoped_local",217),("scoped_ci",108)):
        assert observed[k]=={"tests":n,"failures":0,"errors":0,"skipped":0}
    assert len(data["evidence_sha256"])==13
    assert all(len(x)==64 and all(v in "0123456789abcdef" for v in x)
               for x in data["evidence_sha256"].values())


def test_readable_final_report_does_not_claim_wrong_v1_or_train_admission():
    text=REPORT.read_text(encoding="utf8")
    data=read(VERDICT)
    assert "NOT literal v1" in text
    assert "NOT 100% PASS" in text
    assert "left-turn-train-v1" in text
    assert "held-out" in text.lower()
    assert "6/6 FINISH" in text
    assert "NO automatic retraining" in text
    assert "Training" in text or "training" in text
    assert data["original_historical_report_preserved"] is True


def test_end_gate_is_independently_fail_closed():
    data=read(VERDICT)
    draft=read(DRAFT)
    registry=read(REGISTRY)
    assert data["new_ppo_training_performed"] is False
    assert data["new_training_seeds_consumed"]==0
    assert data["heldout_geometry_read"] is False
    assert data["training_permission"] is False
    assert data["scientific_release_permission"] is False
    assert draft["status"]=="draft_not_authorized"
    assert draft["training_permission"] is False
    assert draft["heldout_permission"] is False
    assert registry["registry_gate_status"]=="CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION"
    assert registry["exception"]["all_15_runs_are_accepted_as_PASS"] is False
    with pytest.raises(PermissionError,match="CLOSED"):
        require_training_admission(True)
