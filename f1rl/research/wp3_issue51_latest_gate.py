"""Live WP3 Issue51 admission: latest development crash is not scientific Eval.

Validates exact frozen model/seed SHA for the three original cases while
preserving the distinction between development-only completed crash and
a certified evaluation outcome. Never authorizes any new episode.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import json
from .wp3_current_eval_gate import OLD, EXPECTED

ROOT=Path(__file__).resolve().parents[2]
LATEST=ROOT/"docs/research/WP3-V4V-THREE-FROZEN-REPLAY-DEV-CRASH-20261024.json"

def source_bound_dev_crash_gate(path:Path=LATEST,origin:Path=OLD)->dict:
    report=json.loads(path.read_text("utf8"))
    roster=json.loads((origin/"evaluation/evaluation-roster.json").read_text("utf8"))
    provenance=json.loads((origin/"WP3-EVAL-SOURCE-PROVENANCE.json").read_text("utf8"))
    if (report.get("roster_sha256")!=roster["sha256"] or
        report.get("roster_file_sha256")!=sha256((origin/"evaluation/evaluation-roster.json").read_bytes()).hexdigest() or
        report.get("source_base_commit")!="ce0e55d7a428826b2d4ed3218641105bd0206f4b" or
        report.get("experiment_status")!="SCRATCH_NOT_REGISTERED" or
        report.get("registered_evaluation_executed") is not False or
        report.get("full_5760_eval_admission")!="BLOCKED" or
        report.get("heldout_access") is not False or
        report.get("all_required_replays_scientifically_certified") is not False or
        provenance["evaluation_roster_sha256"]!=roster["sha256"]):
        raise ValueError("unregistered developmental replay cannot be promoted")
    cases=report.get("cases")
    if not isinstance(cases,list) or len(cases)!=3:raise ValueError("three exact frozen cases required")
    models={x["model_id"]:x for x in provenance["models"]}
    accepted_dev=[]
    for row,expected in zip(cases,EXPECTED,strict=True):
        model,track,stratum,ep,original_error=expected
        if (row.get("model_id"),row.get("track"),row.get("stratum"),row.get("episode"))!=(model,track,stratum,ep):
            raise ValueError("replay source case ordering mismatch")
        roster_row=next((x for x in roster["seed_rows"] if (x["model_id"],x["track_id"],x["stratum"])==(model,track,stratum)),None)
        if roster_row is None:raise ValueError("missing original source cell")
        original_seed=roster_row["episodes"][ep]
        if (row.get("env_seed")!=original_seed["environment_seed"]
            or row.get("policy_seed")!=original_seed["policy_seed"]
            or row.get("model_final_sha256")!=models[model]["model_final_sha256"]
            or row.get("scientific_evidence") is not False):
            raise ValueError("source model/seed/science evidence mismatch")
        if model=="C-seed0" and track=="slalom-v1":
            if (row.get("status")!="ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED"
                or row.get("failure_reason")!="physical_crash_development" or row.get("success_flag_unaccepted") is not False):
                raise ValueError("must retain development-only crash status")
            accepted_dev.append((model,track))
        elif row.get("status")!="INVALID_UNCERTIFIED_PHYSICS_OR_EVAL" or row.get("error_type")!=original_error:
            raise ValueError("road or ramp physics failure altered or concealed")
    if accepted_dev!=[("C-seed0","slalom-v1")]:
        raise ValueError("development crash check missing")
    return {
        "status":"BLOCKED",
        "source":"WP3-V4V_THREE_FROZEN_REPLAY_DEVELOPMENT",
        "model_seed_sha256_binding":"PASS",
        "real_frozen_replay_cases":3,
        "development_only_completed_crash":1,
        "invalid_road_or_ramp_cases":2,
        "scientifically_certified_cases":0,
        "registered_evaluation_started":False,
        "full_eval_episodes_started":0,
        "report_sha256":sha256(path.read_bytes()).hexdigest(),
    }
