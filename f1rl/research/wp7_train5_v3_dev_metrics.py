"""Pre-registered TRAIN-5 v3 development metrics; no training admission.

This module aggregates *future* exact final-checkpoint TRAIN-only episodes.
Even a numerically passing evaluation cannot authorize PPO or HELD-OUT.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import re

SCHEMA="f1rl_train5_v3_dev_metrics_v1"
COHORTS=("scratch","frozen_initialization")
CONDITIONS=("nominal","perturbation")
REPLICATES=(0,1,2)
EPISODES_PER_CELL=60
OUTCOMES=frozenset({
    "finished", "valid_step_limit", "hard_route_violation",
    "certified_physical_crash", "invalid_physics",
})
SHA=re.compile(r"[0-9a-f]{64}\Z")


class DevelopmentMetricError(ValueError):
    pass


def evaluate_dev_metrics(records: list[dict], contract: dict) -> dict:
    """Validate complete 3600 TRAIN episodes then measure fixed thresholds.

    Produces a metric verdict, *not* a scientific training admission or
    research held-out release. Never alters data, seeds or checkpoint files.
    """
    if not isinstance(records,list) or not isinstance(contract,dict):
        raise DevelopmentMetricError("records list and frozen contract required")
    spec=contract.get("scientific_quality_criteria_proposal")
    if not isinstance(spec,dict) or spec.get("decision")!="PROPOSED_NOT_APPROVED":
        raise DevelopmentMetricError("invalid unapproved metric schema")
    expected_tracks=contract.get("track_ids")
    if (not isinstance(expected_tracks,list) or len(expected_tracks)!=5 or
        len(set(expected_tracks))!=5 or
        expected_tracks!=[
            "straight-sprint-v1","slalom-v1","right-turn-train-v2",
            "ramp-v1","left-turn-train-v2",
        ]):
        raise DevelopmentMetricError("track roster not exactly registered five TRAIN tracks")
    budget=spec.get("preexisting_dev_design")
    if budget!={"cells":60,"episodes_per_cell":60,"total":3600}:
        raise DevelopmentMetricError("development budget changed")
    if (spec.get("thresholds_are_exploratory_not_owner_signed") is not True or
        contract.get("scientific_admission_approved") is not False or
        contract.get("permissions",{}).get("ppo_training") is not False):
        raise DevelopmentMetricError("metric gate cannot confer scientific authorization")
    thresholds=spec["preproposal_thresholds"]
    required={
        "uncertified_crash_or_invalid_physics_count_max":0,
        "nominal_finish_rate_min":.9,
        "per_track_finish_rate_min":.8,
        "hard_route_violation_rate_max":.02,
        "certified_physical_crash_rate_max":.01,
    }
    if thresholds!=required:
        raise DevelopmentMetricError("metric thresholds modified")
    if len(records)!=3600:
        raise DevelopmentMetricError("exactly 3600 development episodes required")
    expected_keys=set(product(COHORTS,REPLICATES,expected_tracks,CONDITIONS))
    cells=defaultdict(set)
    counts=Counter()
    model_hashes={}
    per_track=Counter()
    nominal=Counter()
    for row in records:
        if not isinstance(row,dict) or set(row)!={
            "cohort","replicate","track_id","condition","episode_index",
            "checkpoint_step","checkpoint_sha256","outcome","valid_physics",
            "episode_return",
        }:
            raise DevelopmentMetricError("malformed development record")
        cohort,replica,track,condition=row["cohort"],row["replicate"],row["track_id"],row["condition"]
        index=row["episode_index"]
        key=(cohort,replica,track,condition)
        if key not in expected_keys:
            raise DevelopmentMetricError("unregistered or HELD-OUT cell")
        if type(index) is not int or not 0<=index<60 or index in cells[key]:
            raise DevelopmentMetricError("missing/duplicate/noncanonical episode index")
        cells[key].add(index)
        if row["checkpoint_step"]!=200000 or type(row["checkpoint_step"]) is not int:
            raise DevelopmentMetricError("development must use frozen FINAL 200k checkpoint")
        fingerprint=row["checkpoint_sha256"]
        if not isinstance(fingerprint,str) or SHA.fullmatch(fingerprint) is None:
            raise DevelopmentMetricError("checkpoint SHA256 required")
        model_key=(cohort,replica)
        if model_key in model_hashes and model_hashes[model_key]!=fingerprint:
            raise DevelopmentMetricError("checkpoint identity changed across development")
        model_hashes[model_key]=fingerprint
        outcome=row["outcome"]
        if outcome not in OUTCOMES:
            raise DevelopmentMetricError("unrecognized outcome")
        if type(row["valid_physics"]) is not bool:
            raise DevelopmentMetricError("valid_physics must be boolean")
        value=row["episode_return"]
        if type(value) not in (int,float) or not math.isfinite(value):
            raise DevelopmentMetricError("episode return must be finite")
        if outcome=="invalid_physics" or not row["valid_physics"]:
            counts["invalid"]+=1
        counts[outcome]+=1
        per_track[(track,"total")]+=1
        if outcome=="finished":per_track[(track,"finished")]+=1
        if condition=="nominal":
            nominal["total"]+=1
            if outcome=="finished":nominal["finished"]+=1
        counts["return_sum"]+=value
    if (set(cells)!=expected_keys or any(cells[k]!=set(range(60)) for k in expected_keys)
        or len(model_hashes)!=6 or len(set(model_hashes.values()))!=6):
        raise DevelopmentMetricError("incomplete cells or nonindependent checkpoint identities")
    n=len(records)
    rate=lambda a,b: a/b
    score={
        "schema":SCHEMA,
        "input_episode_count":n,"complete_cell_count":len(cells),
        "distinct_final_checkpoint_count":len(model_hashes),
        "nominal_finish_rate":rate(nominal["finished"],nominal["total"]),
        "finish_rate_all":rate(counts["finished"],n),
        "finish_rates_per_track":{
            track:rate(per_track[(track,"finished")],per_track[(track,"total")])
            for track in expected_tracks
        },
        "hard_route_violation_rate":rate(counts["hard_route_violation"],n),
        "certified_physical_crash_rate":rate(counts["certified_physical_crash"],n),
        "invalid_or_uncertified_physics_count":counts["invalid"],
        "mean_episode_return":counts["return_sum"]/n,
        "only_train_tracks":True,
        "no_checkpoint_selection":True,
        "scientific_training_permission":False,
        "heldout_permission":False,
    }
    failures=[]
    if score["nominal_finish_rate"]<thresholds["nominal_finish_rate_min"]:
        failures.append("nominal_finish_rate")
    if any(v<thresholds["per_track_finish_rate_min"] for v in score["finish_rates_per_track"].values()):
        failures.append("per_track_finish_rate")
    if score["hard_route_violation_rate"]>thresholds["hard_route_violation_rate_max"]:
        failures.append("hard_route_violation_rate")
    if score["certified_physical_crash_rate"]>thresholds["certified_physical_crash_rate_max"]:
        failures.append("certified_physical_crash_rate")
    if score["invalid_or_uncertified_physics_count"]>thresholds["uncertified_crash_or_invalid_physics_count_max"]:
        failures.append("invalid_or_uncertified_physics")
    score["failed_metrics"]=failures
    score["development_quality_result"]=(
        "METRICS_PASS_NOT_TRAINING_ADMISSION" if not failures
        else "DEVELOPMENT_METRICS_FAIL"
    )
    return score


def load_registered_development_contract(root:Path)->dict:
    filename=Path(root)/"configs/research/wp7-train5-v3-fixture-and-development-criteria.json"
    return json.loads(filename.read_text("utf8"))
