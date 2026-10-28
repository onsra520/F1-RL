"""WP3 current three-case physics admission, pinned to original model/seed bytes.

No approval is inferred from a replay with zero Python exceptions; acceptance
requires a NEW independent physical certification and evaluator source
registration. Current V4S records remain explicitly INVALID and unscored.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json

BASE=Path(__file__).resolve().parents[2]
import os
OLD=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
CURRENT=BASE/"docs/research/WP3-V4S-THREE-REPLAY-IMPULSE-AUDIT-20261024.json"
EXPECTED=(
    ("C-seed0","right-turn-v1","nominal",0,"SuspensionResolutionError"),
    ("A-seed1","ramp-v1","randomized_train",17,"SuspensionResolutionError"),
    ("C-seed0","slalom-v1","nominal",2,"GroundedBodyCollisionUnscored"),
)

def _json(path:Path)->dict:
    def no_duplicates(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError("duplicate field in physics admission source")
            value[key]=item
        return value
    doc=json.loads(path.read_text("utf8"),object_pairs_hook=no_duplicates)
    if type(doc) is not dict:raise ValueError("invalid replay report")
    return doc

def require_latest_replay_evidence(
    file:Path=CURRENT,origin:Path=OLD,
)->dict:
    origin=Path(origin)
    file=Path(file)
    roster_path=origin/"evaluation/evaluation-roster.json"
    provenance_path=origin/"WP3-EVAL-SOURCE-PROVENANCE.json"
    roster,prov=_json(roster_path),_json(provenance_path)
    latest=_json(file)
    if (latest.get("schema_version")!="wp3_v4_unregistered_three_case_scratch_v1" or
        latest.get("source_base_commit")!="ce0e55d7a428826b2d4ed3218641105bd0206f4b" or
        latest.get("frozen_train_sha")!="fe83f517479f3b990028d6409304fae83b8199a9" or
        latest.get("roster_sha256")!=roster["sha256"] or
        latest.get("roster_file_sha256")!=sha256(roster_path.read_bytes()).hexdigest() or
        latest.get("registered_evaluation_executed") is not False or
        latest.get("full_5760_eval_admission")!="BLOCKED" or
        latest.get("heldout_access") is not False or
        latest.get("original_evidence_mutated") is not False or
        latest.get("experiment_status")!="SCRATCH_NOT_REGISTERED" or
        latest.get("physics_profile")!="spatial_road_edge_partial_lifecycle_v4"):
        raise ValueError("frozen source/roster/origin/approval mismatch")
    if prov.get("evaluation_roster_sha256")!=roster["sha256"]:
        raise ValueError("original frozen roster provenance mismatch")
    models={row["model_id"]:row for row in prov["models"]}
    rows=latest["cases"]
    if type(rows) is not list or len(rows)!=len(EXPECTED):
        raise ValueError("required three real replay cases missing")
    for row,expected in zip(rows,EXPECTED,strict=True):
        model,track,stratum,episode,error_name=expected
        if (type(row) is not dict or
            (row.get("model_id"),row.get("track"),row.get("stratum"),
             row.get("episode"),row.get("error_type"))!=expected or
            row.get("status")!="INVALID_UNCERTIFIED_PHYSICS_OR_EVAL" or
            row.get("scientific_evidence") is not False or
            type(row.get("error")) is not str or not row["error"]):
            raise ValueError("current replay cannot be treated as valid or reordered")
        cell=next((x for x in roster["seed_rows"] if
            (x["model_id"],x["track_id"],x["stratum"])==(model,track,stratum)),None)
        if cell is None:raise ValueError("no source roster record")
        seeds=cell["episodes"][episode]
        if (row.get("env_seed")!=seeds["environment_seed"] or
            row.get("policy_seed")!=seeds["policy_seed"] or
            row.get("model_final_sha256")!=models[model]["model_final_sha256"]):
            raise ValueError("paired seeds or frozen model SHA forged")
    return {
        "schema_version":"wp3_three_real_physics_cases_gate_v1",
        "status":"BLOCKED_UNCERTIFIED_PHYSICS",
        "current_evidence_sha256":sha256(file.read_bytes()).hexdigest(),
        "roster_sha256":roster["sha256"],
        "frozen_model_seed_replays":3,
        "physically_valid_replays":0,
        "registered_eval_approved":False,
        "allowed_new_eval_episodes":0,
        "reason":"Road contact, wheel travel stop and obstacle crash terminal remain unintegrated to the accepted simulator; three episodes invalid",
    }

def require_registered_full_evaluation(document:dict)->None:
    # Hard gate for this unapproved physics source; a future signed successor
    # MUST be a different validator, not a mutation of this V4S source.
    latest=require_latest_replay_evidence()
    if document!=latest:
        raise ValueError("three-case original model/seed physics witness mismatch")
    raise RuntimeError("WP3 5760 EVAL BLOCKED: 0/3 source-validated real replay cases certified")
