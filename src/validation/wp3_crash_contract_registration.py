"""Evidence-bound WP3 crash terminal v1 engineering contract registry.

NO independent physical owner/reviewer signature is asserted here. Software
registration defines replayable scoring semantics; scientific release still
requires explicit independent physics acceptance and an immutable clean source
version with a distinct full-Eval coordinator.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json,re,subprocess

from ..rl.crash_terminal_registered import (
    RACE_V4,CRASH_V1,PHYSICS_V4,descriptor,_sha
)
from .wp3_v4_technical_acceptance import digest_file

EXPECTED=(
    ("C-seed0","right-turn-v1","nominal",0,False,"physical_crash"),
    ("A-seed1","ramp-v1","randomized_train",17,True,None),
    ("C-seed0","slalom-v1","nominal",2,False,"physical_crash"),
)
REQUIRED_SOURCES=(
    "f1rl/learning/crash_terminal_registered.py",
    "f1rl/learning/evaluation/crash_contract.py",
    "f1rl/learning/evaluation/evaluator.py",
    "f1rl/learning/gym_environment.py",
    "f1rl/learning/wp7_race.py",
    "f1rl/runtime/physics.py",
    "f1rl/runtime/simulation.py",
    "f1rl/runtime/road_crash_validation.py",
    "f1rl/sandbox/physical_crash.py",
    "f1rl/sandbox/accepted_contact_transition.py",
    "f1rl/sandbox/accepted_road_body_contact.py",
    "f1rl/sandbox/spatial_contact_dynamics.py",
)
SRC_HEAD="ce0e55d7a428826b2d4ed3218641105bd0206f4b"
TRAIN_SHA="fe83f517479f3b990028d6409304fae83b8199a9"


def _read(path:Path)->dict:
    def no_dupes(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError("duplicate evidence key")
            result[key]=value
        return result
    d=json.loads(path.read_text("utf8"),object_pairs_hook=no_dupes)
    if type(d) is not dict:raise ValueError("evidence root not JSON object")
    return d


def assemble_registration(
    root:Path,history:Path,replay_file:Path,receipt:Path
)->dict:
    root=Path(root).resolve(strict=True)
    history=Path(history).resolve(strict=True)
    replay_file=Path(replay_file).resolve(strict=True)
    receipt=Path(receipt).resolve(strict=True)
    actual_head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
        text=True).strip()
    ancestry=subprocess.run(["git","-C",str(root),"merge-base","--is-ancestor",
                             SRC_HEAD,actual_head],capture_output=True)
    if ancestry.returncode!=0:
        raise ValueError("archived handoff source commit is not the current source ancestor")
    # Historical engineering registry always describes its authoring state.
    current_head=SRC_HEAD
    roster_path=history/"evaluation/evaluation-roster.json"
    prov_path=history/"WP3-EVAL-SOURCE-PROVENANCE.json"
    roster=_read(roster_path)
    prov=_read(prov_path)
    replay=_read(replay_file)
    if (roster.get("expected_cells")!=96
        or roster.get("expected_episodes")!=5760
        or prov.get("training_source_commit")!=TRAIN_SHA
        or prov.get("evaluation_roster_sha256")!=roster.get("sha256")
        or len(prov.get("models",[]))!=12
        or replay.get("schema_version")!="wp3_v4_unregistered_three_case_scratch_v1"
        or replay.get("source_base_commit")!=SRC_HEAD
        or replay.get("frozen_train_sha")!=TRAIN_SHA
        or replay.get("roster_sha256")!=roster["sha256"]
        or replay.get("roster_file_sha256")!=digest_file(roster_path)
        or replay.get("physics_profile")!=PHYSICS_V4
        or replay.get("crash_scoring_contract")!=CRASH_V1
        or replay.get("scientific_signoff") is not False
        or replay.get("experiment_status")!="SCRATCH_NOT_REGISTERED"
        or replay.get("full_5760_eval_admission")!="BLOCKED"
        or replay.get("registered_evaluation_executed") is not False
        or replay.get("all_required_replays_scientifically_certified") is not False
        or replay.get("heldout_access") is not False
        or replay.get("original_evidence_mutated") is not False):
        raise ValueError("registered crash V1 source, seed roster or scientific gate mismatch")
    models={}
    for model in prov["models"]:
        name=model["model_id"]
        if name in models or digest_file(history/"runs"/name/"model-final.zip")!=model["model_final_sha256"]:
            raise ValueError("original model final checkpoint hash mismatch")
        models[name]=model["model_final_sha256"]
    if len(models)!=12:raise ValueError("12 frozen models required")
    rows=replay.get("cases")
    if type(rows) is not list or len(rows)!=3:raise ValueError("three source-bound replays required")
    results=[]
    for row,(mid,track,stratum,episode,success,reason) in zip(rows,EXPECTED,strict=True):
        original=next((x for x in roster["seed_rows"] if
            (x["model_id"],x["track_id"],x["stratum"])==(mid,track,stratum)),None)
        if original is None:raise ValueError("paired source row not found")
        seeds=original["episodes"][episode]
        if (
           (row.get("model_id"),row.get("track"),row.get("stratum"),row.get("episode"))
           !=(mid,track,stratum,episode)
           or row.get("model_final_sha256")!=models[mid]
           or row.get("env_seed")!=seeds["environment_seed"]
           or row.get("policy_seed")!=seeds["policy_seed"]
           or row.get("scratch_physics_version")!=PHYSICS_V4
           or row.get("status")!="ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED"
           or row.get("scientific_evidence") is not False
           or row.get("success_flag_unaccepted") is not success
           or row.get("failure_reason")!=reason
        ):raise ValueError("tampered/crash source identity or outcome")
        results.append({"model_id":mid,"track":track,"stratum":stratum,
                        "episode":episode,"registered_technical_result":
                        "valid_failed_crash" if reason else "valid_finish",
                        "scientifically_admitted":False})
    text=receipt.read_text("utf-8-sig")
    match=re.search(r"(?m)^([0-9]+) passed in [0-9]+(?:\.[0-9]+)?s\s*$",text)
    if match is None or int(match.group(1))<330 or any(
        marker in text for marker in ("FAILED","ERROR","Traceback")):
        raise ValueError("test receipt invalid for versioned crash contract")
    source={name:digest_file(root/name.replace("f1rl/learning/evaluation/", "src/evaluation/", 1).replace("f1rl/learning/", "src/rl/", 1).replace("f1rl/sandbox/", "src/simulation/", 1).replace("f1rl/runtime/", "src/core/", 1).replace("f1rl/research/", "src/validation/", 1).replace("f1rl/local_demo/", "src/interfaces/", 1).replace("f1rl/", "src/", 1)) for name in REQUIRED_SOURCES}
    result={
        "schema_version":"wp3_crash_terminal_v1_technical_registry",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_base_commit":current_head,"source_tree_clean":False,
        "frozen_training_commit":TRAIN_SHA,"physics_version":PHYSICS_V4,
        "race_version":RACE_V4,"crash_contract_version":CRASH_V1,
        "engineering_scoring_contract":descriptor(),
        "engineering_scoring_contract_sha256":_sha(descriptor()),
        "source_files_sha256":source,"source_bundle_sha256":_sha(source),
        "paired_roster_sha256":roster["sha256"],
        "paired_roster_file_sha256":digest_file(roster_path),
        "training_provenance_sha256":digest_file(prov_path),
        "verified_original_model_final_sha256":models,
        "registered_original_checkpoint_count":12,
        "replay_source_file_sha256":digest_file(replay_file),
        "same_build_replay_cases":results,
        "engineering_regression_passed":int(match.group(1)),
        "engineering_regression_failed":0,
        "engineering_regression_receipt_sha256":digest_file(receipt),
        "terminal_outcome":"valid_failed_crash_not_finish",
        "reward_event_prefix":"only_physics_verified_interval_to_first_impact",
        "scope":"ENGINEERING_SCORING_REGISTRATION_ONLY",
        "software_contract_implemented":True,
        "independent_physics_review_signature":None,
        "owner_scientific_signoff":False,
        "scientific_scoring_approved":False,
        "full_5760_episode_evaluation_authorized":False,
        "old_evidence_mutated":False,
        "heldout_access":False,
        "status":"ENGINEERING_CONTRACT_REGISTERED_INDEPENDENT_SIGNOFF_PENDING",
    }
    from .historical_source_records import verify_historical_records
    verify_historical_records(root, history)
    original = _read(root/"docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json")
    if (source != original["source_files_sha256"] or
        digest_file(replay_file) != original["replay_source_file_sha256"] or
        digest_file(receipt) != original["engineering_regression_receipt_sha256"]):
        raise ValueError("historical source identity does not match this source tree")
    raise ValueError("historical source identity cannot authorize receipt reissuance")
