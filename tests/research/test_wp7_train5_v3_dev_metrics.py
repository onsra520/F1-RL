"""Pure TRAIN-only 3600 episode metric gate, test-first synthetic fixtures."""
from functools import lru_cache
from hashlib import sha256
from itertools import product
from pathlib import Path

import pytest

from f1rl.research.wp7_train5_v3_dev_metrics import (
    COHORTS,CONDITIONS,REPLICATES,DevelopmentMetricError,
    evaluate_dev_metrics,load_registered_development_contract,
)

ROOT=Path(__file__).resolve().parents[2]
S=load_registered_development_contract(ROOT)
TRACKS=S["track_ids"]


@lru_cache(maxsize=1)
def baseline_rows():
    rows=[]
    for cohort,replica,track,condition in product(COHORTS,REPLICATES,TRACKS,CONDITIONS):
        identity=sha256(f"{cohort}/{replica}".encode()).hexdigest()
        for index in range(60):
            rows.append({
                "cohort":cohort,"replicate":replica,
                "track_id":track,"condition":condition,"episode_index":index,
                "checkpoint_step":200000,"checkpoint_sha256":identity,
                "outcome":"finished","valid_physics":True,"episode_return":100.0,
            })
    return tuple(rows)


def clone_rows():
    return [r.copy() for r in baseline_rows()]


def test_full_3600_cells_finish_is_metric_pass_but_not_scientific_admission():
    rows=clone_rows()
    r=evaluate_dev_metrics(rows,S)
    assert r["input_episode_count"]==3600
    assert r["complete_cell_count"]==60
    assert r["distinct_final_checkpoint_count"]==6
    assert r["nominal_finish_rate"]==1
    assert r["finish_rate_all"]==1
    assert all(x==1 for x in r["finish_rates_per_track"].values())
    assert r["mean_episode_return"]==100
    assert r["development_quality_result"]=="METRICS_PASS_NOT_TRAINING_ADMISSION"
    assert r["scientific_training_permission"] is False
    assert r["heldout_permission"] is False


def test_invalid_physics_is_rejected_by_metrics_even_if_every_episode_finishes():
    rows=clone_rows()
    rows[0]["valid_physics"]=False
    r=evaluate_dev_metrics(rows,S)
    assert r["invalid_or_uncertified_physics_count"]==1
    assert "invalid_or_uncertified_physics" in r["failed_metrics"]
    assert r["development_quality_result"]=="DEVELOPMENT_METRICS_FAIL"


def test_hard_route_violation_threshold_is_enforced():
    rows=clone_rows()
    for row in rows[:80]:
        row["outcome"]="hard_route_violation"
    result=evaluate_dev_metrics(rows,S)
    assert result["hard_route_violation_rate"]>0.02
    assert "hard_route_violation_rate" in result["failed_metrics"]


def test_certified_crash_threshold_is_enforced():
    rows=clone_rows()
    for row in rows[:38]:
        row["outcome"]="certified_physical_crash"
    result=evaluate_dev_metrics(rows,S)
    assert result["certified_physical_crash_rate"]>0.01
    assert "certified_physical_crash_rate" in result["failed_metrics"]


def test_per_track_finish_rate_threshold_is_enforced():
    rows=clone_rows()
    changed=0
    for row in rows:
        if row["track_id"]==TRACKS[0] and changed<150:
            row["outcome"]="valid_step_limit"
            changed+=1
    result=evaluate_dev_metrics(rows,S)
    assert result["finish_rates_per_track"][TRACKS[0]]<.8
    assert "per_track_finish_rate" in result["failed_metrics"]


def test_nominal_finish_threshold_is_enforced():
    rows=clone_rows()
    modified=0
    for row in rows:
        if row["condition"]=="nominal" and modified<200:
            row["outcome"]="valid_step_limit"
            modified+=1
    result=evaluate_dev_metrics(rows,S)
    assert result["nominal_finish_rate"]<.9
    assert "nominal_finish_rate" in result["failed_metrics"]


@pytest.mark.parametrize("mutation",(
    "heldout_track","missing_episode","duplicate_episode_index",
    "wrong_checkpoint","wrong_step","NaN_return","invalid_outcome",
    "wrong_condition","boolean_index","invalid_sha",
))
def test_incomplete_or_corrupt_development_fail_closed(mutation):
    rows=clone_rows()
    if mutation=="heldout_track":rows[0]["track_id"]="left-turn-v1"
    elif mutation=="missing_episode":rows.pop()
    elif mutation=="duplicate_episode_index":rows[1]["episode_index"]=0
    elif mutation=="wrong_checkpoint":rows[0]["checkpoint_sha256"]="0"*64
    elif mutation=="wrong_step":rows[0]["checkpoint_step"]=175000
    elif mutation=="NaN_return":rows[0]["episode_return"]=float("nan")
    elif mutation=="invalid_outcome":rows[0]["outcome"]="unrecognized_crash"
    elif mutation=="wrong_condition":rows[0]["condition"]="test"
    elif mutation=="boolean_index":rows[0]["episode_index"]=True
    elif mutation=="invalid_sha":rows[0]["checkpoint_sha256"]="garbage"
    with pytest.raises(DevelopmentMetricError):
        evaluate_dev_metrics(rows,S)


def test_unapproved_or_changed_thresholds_refuse_metric_evaluation():
    rows=clone_rows()
    changed=S.copy()
    changed["scientific_quality_criteria_proposal"]={
        **S["scientific_quality_criteria_proposal"],
        "preproposal_thresholds":{
            **S["scientific_quality_criteria_proposal"]["preproposal_thresholds"],
            "nominal_finish_rate_min":.6,
        },
    }
    with pytest.raises(DevelopmentMetricError,match="threshold"):
        evaluate_dev_metrics(rows,changed)


def test_bogus_approval_flip_cannot_make_unsigned_thresholds_admissible():
    rows=clone_rows()
    changed={
        **S,"scientific_admission_approved":True,
    }
    with pytest.raises(DevelopmentMetricError,match="cannot confer"):
        evaluate_dev_metrics(rows,changed)
