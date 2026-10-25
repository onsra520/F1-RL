"""WP3 Issue #51: standalone evaluation-only source/physics admission.

This module contains no training, episode execution, or untrusted resume paths.
It binds the *frozen 12 final PPO models* and original paired seed roster
to the current source, then refuses a new 5,760-slot Eval until three
required real scratch replays are certified under a new registered physics
profile. Historical evaluation roots are never mutated by this module.
"""
from __future__ import annotations

from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import subprocess

FROZEN_TRAIN="fe83f517479f3b990028d6409304fae83b8199a9"
HANDOFF_SOURCE="ce0e55d7a428826b2d4ed3218641105bd0206f4b"
TRACKS=("straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1")
STRATA=("nominal","randomized_train")
MODELS=tuple(f"{candidate}-seed{seed}" for candidate in "ABCD" for seed in range(3))
REPLAY_CASES=(("C-seed0","right-turn-v1","nominal",0),
              ("A-seed1","ramp-v1","randomized_train",17),
              ("C-seed0","slalom-v1","nominal",2))
SCHEMA="wp3_eval_only_physics_preflight_v1"


def canonical(document:object)->bytes:
    return json.dumps(document,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False).encode("utf8")


def digest(document:object)->str:
    return sha256(canonical(document)).hexdigest()


def hash_file(path:Path)->str:
    hasher=sha256()
    with Path(path).open("rb") as input:
        for chunk in iter(lambda:input.read(1024*1024),b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json(path:Path)->dict:
    def strict(pairs):
        item={}
        for key,val in pairs:
            if key in item:
                raise ValueError(f"duplicate JSON field {key}")
            item[key]=val
        return item
    obj=json.loads(Path(path).read_text("utf8"),object_pairs_hook=strict)
    if type(obj) is not dict:raise ValueError("JSON root must be object")
    return obj


def _git(root:Path,*args:str)->str:
    return subprocess.check_output(["git","-C",str(root),*args],
                                   text=True,stderr=subprocess.PIPE).strip()


def _roster_check(roster:dict, provenance:dict)->None:
    original={key:val for key,val in roster.items() if key!="sha256"}
    if (roster.get("sha256")!=digest(original) or
        provenance.get("evaluation_roster_sha256")!=roster["sha256"] or
        provenance.get("training_source_commit")!=FROZEN_TRAIN):
        raise ValueError("registered historical roster/source SHA mismatch")
    e=roster["evaluation"]
    if (e["tracks"]!=list(TRACKS) or e["strata"]!=list(STRATA) or
        e["hard_route_boundary_m"]!=4.0 or
        e["episodes_per_track_stratum_model"]!=60 or
        e["checkpoint_selection"]!="final_checkpoint_only" or
        e["policy_actions"]!="stochastic_with_paired_candidate_independent_rng_seeds" or
        e["heldout_access"] is not False or
        roster["expected_cells"]!=96 or roster["expected_episodes"]!=5760):
        raise ValueError("evaluation would drift from canonical original task")
    rows=roster["seed_rows"]
    if type(rows) is not list or len(rows)!=96:
        raise ValueError("evaluation requires exactly 96 scheduled cells")
    observed=[]
    seed_pairs={}
    for row in rows:
        key=(row["model_id"],row["track_id"],row["stratum"])
        observed.append(key)
        model=row["model_id"]
        if (model not in MODELS or
            row["candidate_id"]!=model[0] or
            type(row["seed_index"]) is not int or
            row["seed_index"]!=int(model[-1]) or
            row["track_id"] not in TRACKS or row["stratum"] not in STRATA):
            raise ValueError("roster model identity/track/stratum mismatch")
        episodes=row["episodes"]
        if type(episodes) is not list or len(episodes)!=60:
            raise ValueError("wrong episodes per cell")
        for n,item in enumerate(episodes):
            if (set(item)!={"episode","environment_seed","policy_seed"} or
                type(item["episode"]) is not int or item["episode"]!=n or
                any(type(item[x]) is not int or not 0<=item[x]<2**32
                    for x in ("environment_seed","policy_seed"))):
                raise ValueError("roster episode order or RNG seed invalid")
            pair=(item["environment_seed"],item["policy_seed"])
            index=(row["track_id"],row["stratum"],n)
            if index in seed_pairs and seed_pairs[index]!=pair:
                raise ValueError("paired environment/policy seed mismatch")
            seed_pairs[index]=pair
    if len(set(observed))!=96 or set(observed)!={
        (model,track,stratum) for model in MODELS for track in TRACKS for stratum in STRATA
    } or len(seed_pairs)!=4*2*60:
        raise ValueError("roster cell enumeration mismatch")
    if len({v[0] for v in seed_pairs.values()})!=480 or len({v[1] for v in seed_pairs.values()})!=480:
        raise ValueError("seed collision / candidate pairing invalid")


def build_eval_only_preflight(root:Path,history:Path,three_replays:Path)->dict:
    root=Path(root).resolve(strict=True)
    history=Path(history).resolve(strict=True)
    source=_git(root,"rev-parse","HEAD")
    ancestry=subprocess.run(["git","-C",str(root),"merge-base","--is-ancestor",
                             HANDOFF_SOURCE,source],capture_output=True)
    if ancestry.returncode!=0:
        raise ValueError("original handoff base must be ancestor of evaluation-only source")
    # This read-only forensic preflight describes v3 and its ORIGINAL
    # handoff, never an approval to score the new v4 implementation.
    original=_json(history/"WP3-EVAL-SOURCE-PROVENANCE.json")
    roster=_json(history/"evaluation/evaluation-roster.json")
    _roster_check(roster,original)
    if (len(original["models"])!=12 or
        {row["model_id"] for row in original["models"]}!=set(MODELS)):
        raise ValueError("frozen 12-model provenance roster invalid")
    weights=[]
    for model in sorted(original["models"],key=lambda x:x["model_id"]):
        mid=model["model_id"]
        folder=history/"runs"/mid
        values={}
        for title,filename,expected in (
            ("checkpoint","model-final.zip","model_final_sha256"),
            ("telemetry","training-telemetry.jsonl","training_telemetry_sha256"),
            ("validation","engineering-validation.json","engineering_validation_sha256"),
        ):
            actual=hash_file(folder/filename)
            if actual!=model[expected]:
                raise ValueError(f"frozen {mid} {title} SHA-256 mismatch")
            values[title+"_sha256"]=actual
        state=_json(folder/"state.json")
        if state.get("final_model",{}).get("sha256")!=values["checkpoint_sha256"]:
            raise ValueError(f"frozen {mid} state model provenance mismatch")
        weights.append({"model_id":mid,**values})
    replay=_json(three_replays)
    got={(r["model_id"],r["track"],r["stratum"],r["episode"]):r for r in replay["results"]}
    if (set(got)!=set(REPLAY_CASES) or len(replay["results"])!=3 or
        replay["original_roster_sha256"]!=roster["sha256"] or
        replay["training_commit"]!=FROZEN_TRAIN or
        replay["source_commit"]!=HANDOFF_SOURCE or
        replay["physics_profile"]!="spatial_rotating_owner_v3"):
        raise ValueError("three frozen model and seed replay provenance mismatch")
    # All original cases are still blocked. A new opt-in physics profile
    # would require its own source/seed-bound replay report, not this old one.
    unresolved=[{"model_id":k[0],"track":k[1],"stratum":k[2],
                 "episode":k[3],"error_type":got[k].get("error_type"),
                 "status":got[k]["status"]}
                for k in REPLAY_CASES
                if got[k]["status"]!="VALID_EPISODE"]
    if replay["full_eval_gate"]!="BLOCKED" or len(unresolved)!=3:
        raise ValueError("original physics gate replays unexpectedly modified")
    assets={name:hash_file(root/"assets/tracks/training"/(name+".json")) for name in TRACKS}
    plan={
        "schema_version":SCHEMA,"evaluated_at_utc":datetime.now(timezone.utc).isoformat(),
        "handoff_source_commit":HANDOFF_SOURCE,
        "source_dirty":bool(_git(root,"status","--porcelain")),
        "frozen_train_commit":FROZEN_TRAIN,"historical_roster_sha256":roster["sha256"],
        "historical_roster_file_sha256":hash_file(history/"evaluation/evaluation-roster.json"),
        "historical_eval_provenance_sha256":hash_file(history/"WP3-EVAL-SOURCE-PROVENANCE.json"),
        "historical_evaluation_root_mutated":False,
        "frozen_final_checkpoint_count":12,"checkpoint_validation":"12_MODELS_ALL_SHA256_MATCH",
        "verified_models":weights,"training_track_sha256":assets,
        "expected_cells":96,"expected_episodes":5760,
        "episode_order":"12_models_x_4_train_tracks_x_2_train_strata_x_60_frozen_seed_pairs",
        "evaluation_inference":"stochastic_deterministic_false_cpu",
        "original_boundary_m":4.0,"curriculum_applied":False,
        "new_run_identity":None,"new_physics_profile":None,
        "current_physics_profile":"spatial_rotating_owner_v3_UNCHANGED",
        "mandatory_replay_cases":unresolved,
        "mandatory_replay_gate":"BLOCKED",
        "standalone_evaluator_new_run_admitted":False,
        "full_eval_execution_authorized":False,
        "new_episodes_executed_by_preflight":0,
        "registered_eval_started":False,
        "heldout_access":False,
        "scientific_evidence":False,
        "blocked_reason":"Two SuspensionResolutionError and one FlightPhysicsError remain in exact frozen model/seed scratch replays; separately versioned partial-wheel-release and scored ballistic impact response are not installed/certified.",
    }
    plan["sha256"]=digest(plan)
    return plan


def refuse_full_eval(plan:dict)->None:
    if plan.get("sha256")!=digest({k:v for k,v in plan.items() if k!="sha256"}):
        raise ValueError("preflight checksum mismatch")
    # Intentionally never enable run on the old physics profile. The future
    # owner-approved evaluator is to have a separate registration path.
    raise RuntimeError("FULL_EVAL_BLOCKED: physics not completed; no registered new evaluation ID. Preserve all 5,760 original slots and old evidence.")
