"""Regression for fail-closed WP7 retrospective evidence and freeze gate.

Metadata-only tests: no PPO training or HELD-OUT asset access.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from f1rl.learning.freeze_readiness import (
    BLOCKERS,
    validate_completed_seed_state,
    validate_linked_audits,
)
from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.learning.training.config import TrainingRunConfig


ROOT=Path(__file__).resolve().parents[2]
PROFILE=TrainingRunConfig.from_yaml(ROOT/"configs/recovery/wp7-pilot-200k.yml")


def valid_state(seed=0):
    state={
        "run_id":f"2026-10-11_205037_ppo-multitrack-seed{seed}-sample{seed}",
        "status":"completed",
        "mode":"multitrack",
        "requested_total_timesteps":200_000,
        "actual_total_timesteps":200_704,
        "checkpoint_every_steps":25_000,
        "latest_checkpoint":f"checkpoints/ppo-multitrack-seed{seed}-sample{seed}-step175000.zip",
        "final_model":f"ppo-multitrack-seed{seed}-sample{seed}-final-model.zip",
        "training_tracks":list(TRAIN_IDS),
        "sampling_seed":seed,
        "ppo":{**asdict(PROFILE.ppo),"seed":seed},
        "vector_env":asdict(PROFILE.vector_env),
        "provenance_file":None,
    }
    return state


def verify_state(state,seed=0):
    validate_completed_seed_state(
        state,
        expected_run_id=f"2026-10-11_205037_ppo-multitrack-seed{seed}-sample{seed}",
        expected_ppo_seed=seed,expected_sampling_seed=seed,
        expected_profile=PROFILE,
    )


@pytest.mark.parametrize("seed",[0,1])
def test_actual_seed_configs_satisfy_metadata_contract(seed):
    verify_state(valid_state(seed),seed)


@pytest.mark.parametrize("field,value",[
    ("status","running"),
    ("mode","singletrack"),
    ("requested_total_timesteps",199_999),
    ("actual_total_timesteps",199_999),
    ("actual_total_timesteps",225_000),
    ("actual_total_timesteps",True),
    ("checkpoint_every_steps",None),
    ("training_tracks",list(TRAIN_IDS[::-1])),
    ("sampling_seed",1),
    ("final_model","incorrect.zip"),
    ("latest_checkpoint","checkpoints/step50000.zip"),
])
def test_corrupt_finished_run_identity_is_rejected(field,value):
    state=valid_state()
    state[field]=value
    with pytest.raises(ValueError,match="run identity/state"):
        verify_state(state)


def test_silent_config_drift_rejected():
    state=valid_state()
    state["ppo"]["learning_rate"]=.0004
    with pytest.raises(ValueError,match="PPO/vector config"):
        verify_state(state)
    state=valid_state()
    state["vector_env"]["n_envs"]=4
    with pytest.raises(ValueError,match="PPO/vector config"):
        verify_state(state)


def fake_records():
    return [
        {"run_id":f"r{i}","ppo_seed":i,"source_git_revision":"commit1",
         "model_sha256":f"model{i}","manifest_sha256":f"manifest{i}"}
        for i in range(2)
    ]


def fake_rows():
    return [
        {"track_id":key,"success":True,"simulator_obstacle_hit_count":0,
         "max_horizontal_route_deviation_m":2. if key!="right-turn-v1" else 4.,
         "min_continuous_xz_point_to_obstacle_m":2. if key=="slalom-v1" else None}
        for key in TRAIN_IDS
    ]


def fake_xz():
    return {
        "schema_version":"wp7-recovery-dual-seed-trajectory-audit-v1",
        "scientific_evidence":False,"heldout_rollouts":0,
        "audits":[{
            "run_id":f"r{i}","source_git_revision":"commit1",
            "final_model_sha256":f"model{i}",
            "evidence_manifest_sha256":f"manifest{i}",
            "scientific_evidence":False,"heldout_rollouts":0,
            "results":fake_rows(),
        } for i in range(2)]
    }


def fake_3d():
    names=[
        "sensitivity_small_3p2x1p4x1p0",
        "sensitivity_medium_4p0x1p8x1p2",
        "sensitivity_large_5p0x2p2x1p4",
    ]
    return {
        "schema_version":"wp7-recovery-dual-seed-3d-body-proxy-audit-v1",
        "scientific_evidence":False,"heldout_rollouts":0,
        "audits":[{
            "run_id":f"r{i}","source_git_revision":"commit1",
            "final_model_sha256":f"model{i}","manifest_sha256":f"manifest{i}",
            "scientific_evidence":False,"heldout_rollouts":0,
            "track_results":[{
                "track_id":key,"success":True,"simulator_obstacle_hit_count":0,
                "profiles":[{
                    "name":name,
                    "conservative_clearance_pass":(j!=2 or i==1),
                    "min_conservative_gap_m":(-.04 if j==2 and i==0 else .4),
                } for j,name in enumerate(names)],
            } for key in TRAIN_IDS],
        } for i in range(2)],
    }


def test_both_linked_geometry_reports_match_expected_models_and_splits():
    a=validate_linked_audits(audit_type="trajectory_xz",document=fake_xz(),records=fake_records())
    assert len(a["runs"])==2
    assert a["runs"][0]["right_turn_max_deviation_m"]==4.
    b=validate_linked_audits(audit_type="swept_3d_proxy",document=fake_3d(),records=fake_records())
    assert b["runs"][0]["large_proxy_result"]=="inconclusive_not_collision_proof"
    assert b["runs"][1]["large_proxy_result"]=="clearance_pass_for_assumed_geometry"


@pytest.mark.parametrize("kind",[
    "wrong_seed_model",
    "wrong_manifest",
    "heldout_rollouts",
    "track_swap",
    "false_success",
])
def test_modified_xz_evidence_fails_closed(kind):
    doc=fake_xz()
    item=doc["audits"][0]
    if kind=="wrong_seed_model": item["final_model_sha256"]="model1"
    if kind=="wrong_manifest": item["evidence_manifest_sha256"]="manifest1"
    if kind=="heldout_rollouts": doc["heldout_rollouts"]=1
    if kind=="track_swap": item["results"][1]["track_id"]="left-turn-v1"
    if kind=="false_success": item["results"][1]["success"]=False
    with pytest.raises(ValueError):
        validate_linked_audits(audit_type="trajectory_xz",document=doc,records=fake_records())


def test_proxy_negative_is_inconclusive_not_collisions():
    doc=fake_3d()
    doc["audits"][0]["track_results"][1]["profiles"][2]["conservative_clearance_pass"]=True
    with pytest.raises(ValueError,match="large proxy"):
        validate_linked_audits(audit_type="swept_3d_proxy",document=doc,records=fake_records())


def test_scientific_gates_cannot_disappear_due_to_two_train_successes():
    assert len(BLOCKERS)==6
    assert any("geometry" in x for x in BLOCKERS)
    assert any("provenance" in x for x in BLOCKERS)
    assert any("backup" in x for x in BLOCKERS)


def test_only_published_audits_may_change_since_original_train_commit():
    from f1rl.learning.freeze_readiness import verify_training_source_diff
    assert verify_training_source_diff((
        "f1rl/learning/trajectory_audit.py",
        "f1rl/learning/swept_volume_audit.py",
    ))==(
        "f1rl/learning/swept_volume_audit.py",
        "f1rl/learning/trajectory_audit.py",
    )
    for path in (
        "f1rl/sandbox/world.py",
        "f1rl/learning/reconstruction.py",
        "assets/tracks/training/slalom-v1.json",
        "config.yml",
    ):
        with pytest.raises(ValueError,match="changed"):
            verify_training_source_diff((path,))
