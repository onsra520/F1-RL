"""WP7 geometry contract and registered TRAIN stochastic-action screening invariants."""
from pathlib import Path
from math import isclose
import pytest
from src.rl.geometry_review import load_geometry_review
from src.rl.robustness_screen import (
    seed_for,one_sided_clopper_pearson_lower,EPISODES_PER_TRACK,SCREEN_LOWER_BOUND,
    PROTOCOL_ID,
)
from src.rl.pilot_preflight import TRAIN_IDS

ROOT=Path(__file__).resolve().parents[2]


def test_original_physics_referenced_without_approving_body():
    item=load_geometry_review(ROOT/"configs/recovery/wp7-vehicle-geometry-review.yml")
    assert item.status=="unapproved"
    assert item.baseline_wheelbase_m==3.
    assert item.baseline_ride_height_m==0.5
    assert not item.heldout_evaluation_authorized
    assert set(item.missing_body_fields)=={
        "length_m","width_m","height_m","local_center_xyz_m",
        "collision_hull","tire_footprint",
    }


@pytest.mark.parametrize("change",[
    "status: approved",
    "training_change_authorized: true",
    "heldout_evaluation_authorized: true",
    "max_lateral_deviation_m: 5.6",
    "length_m: 4.0",
])
def test_geometry_assumptions_cannot_be_silently_approved(tmp_path,change):
    original=(ROOT/"configs/recovery/wp7-vehicle-geometry-review.yml").read_text("utf-8")
    key=change.split(":",1)[0]+":"
    rows=original.splitlines()
    idx=next(i for i,s in enumerate(rows) if s.strip().startswith(key))
    rows[idx]=rows[idx].split(key)[0]+change
    path=tmp_path/"geometry.yml"
    path.write_text("\n".join(rows),encoding="utf-8")
    with pytest.raises(ValueError):
        load_geometry_review(path)


def test_registered_stochastic_seed_is_stable_and_distinct():
    assert EPISODES_PER_TRACK==16
    assert PROTOCOL_ID=="wp7-train-stochastic-action-sensitivity-v1"
    seeds={seed_for(run_id="pilot0",track_id=track,index=i) for track in TRAIN_IDS for i in range(16)}
    assert len(seeds)==64
    assert seed_for(run_id="pilot0",track_id=TRAIN_IDS[0],index=1)!=seed_for(
        run_id="pilot1",track_id=TRAIN_IDS[0],index=1)
    with pytest.raises(ValueError):
        seed_for(run_id="pilot0",track_id="left-turn-v1",index=0)


@pytest.mark.parametrize("successes,trials,expected",[
    (0,16,0.),
    (16,16,0.05**(1/16)),
    (1,1,0.05),
    (1,2,1-(.95)**.5),
])
def test_exact_one_sided_binomial_lower_bound(successes,trials,expected):
    value=one_sided_clopper_pearson_lower(successes,trials)
    assert isclose(value,expected,abs_tol=1e-7)
    assert 0<=value<=1


@pytest.mark.parametrize("successes,trials",[(17,16),(-1,16),(4,0),(True,16)])
def test_invalid_binomial_screen_fails_closed(successes,trials):
    with pytest.raises(ValueError):
        one_sided_clopper_pearson_lower(successes,trials)


def test_registered_screen_is_not_automatic_freeze():
    assert SCREEN_LOWER_BOUND==0.80
    assert 0.05**(1/16)>SCREEN_LOWER_BOUND


def fake_screen_document():
    from src.rl.robustness_screen import seed_for,one_sided_clopper_pearson_lower
    run_ids=("r0","r1")
    screens=[]
    for i,run_id in enumerate(run_ids):
        tracks=[]
        for track_id in TRAIN_IDS:
            fail={
                (0,"right-turn-v1"):{0,1,2,3},
                (1,"slalom-v1"):{0},
                (1,"right-turn-v1"):{0},
            }.get((i,track_id),set())
            episodes=[{
                "index":j,
                "torch_action_seed":seed_for(run_id=run_id,track_id=track_id,index=j),
                "success":j not in fail,
            } for j in range(16)]
            successes=16-len(fail)
            lower=one_sided_clopper_pearson_lower(successes,16)
            tracks.append({
                "track_id":track_id,
                "episodes":episodes,
                "episode_count":16,
                "success_count":successes,
                "success_rate":successes/16,
                "one_sided_95pct_cp_lower":round(lower,8),
                "screen_lower_bound_ge_0p80":lower>=.8,
            })
        screens.append({
            "schema_version":"wp7-recovery-train-stochastic-screen-v1",
            "run_id":run_id,
            "model_sha256":f"model{i}",
            "original_manifest_sha256":f"manifest{i}",
            "original_training_git_commit":"origin",
            "protocol_id":PROTOCOL_ID,
            "geometry_review_sha256":"geom",
            "geometry_status":"unapproved",
            "environment_variation":"none_fixed_TRAIN_road_initial_state",
            "sampling_mode":"stochastic_ppo_policy_actions_torch_seeded_independently_per_episode",
            "episodes_per_train_track":16,
            "confidence_level":.95,
            "predeclared_screen_lower_bound":.8,
            "scientific_evidence":False,
            "heldout_rollouts":0,
            "scientific_freeze_permitted":False,
            "heldout_evaluation_permitted":False,
            "all_train_track_screens_pass":False,
            "train_tracks":tracks,
        })
    return {
        "schema_version":"wp7-recovery-dual-seed-action-sensitivity-v1",
        "status":"exploratory_not_scientific",
        "scientific_evidence":False,
        "heldout_rollouts":0,
        "scientific_freeze_permitted":False,
        "heldout_evaluation_permitted":False,
        "geometry_review_status":"unapproved",
        "geometry_review_sha256":"geom",
        "screens":screens,
    }


def fake_archive_records():
    return [{
        "run_id":f"r{i}","model_sha256":f"model{i}",
        "manifest_sha256":f"manifest{i}",
        "source_git_revision":"origin",
    } for i in range(2)]


def test_freeze_gate_links_real_stochastic_protocol_and_keeps_blocked():
    from src.rl.freeze_readiness import validate_stochastic_train_screen
    r=validate_stochastic_train_screen(
        document=fake_screen_document(),records=fake_archive_records(),
        geometry_sha256="geom",
    )
    assert len(r["runs"])==2
    assert not r["environment_generalization_established"]
    assert r["runs"][0]["tracks"][2]["success_count"]==12
    assert not r["runs"][1]["all_screen_pass"]


@pytest.mark.parametrize("corruption",[
    "heldout","wrong_model","wrong_seed","wrong_track","wrong_ci",
    "wrong_pass","wrong_geometry","fake_approval","wrong_count",
])
def test_freeze_gate_rejects_stochastic_evidence_tampering(corruption):
    from src.rl.freeze_readiness import validate_stochastic_train_screen
    d=fake_screen_document()
    s=d["screens"][0]
    t=s["train_tracks"][2]
    if corruption=="heldout": d["heldout_rollouts"]=1
    if corruption=="wrong_model": s["model_sha256"]="model1"
    if corruption=="wrong_seed": t["episodes"][1]["torch_action_seed"]+=1
    if corruption=="wrong_track": t["track_id"]="left-turn-v1"
    if corruption=="wrong_ci": t["one_sided_95pct_cp_lower"]=1.
    if corruption=="wrong_pass": t["screen_lower_bound_ge_0p80"]=True
    if corruption=="wrong_geometry": d["geometry_review_sha256"]="other"
    if corruption=="fake_approval": s["scientific_freeze_permitted"]=True
    if corruption=="wrong_count": t["episode_count"]=15
    with pytest.raises(ValueError,match="stochastic screen"):
        validate_stochastic_train_screen(
            document=d,records=fake_archive_records(),geometry_sha256="geom"
        )
