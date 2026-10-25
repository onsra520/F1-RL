"""WP3 physics v4: source-bound ENGINEERING acceptance, not scientific signoff.

The current uncommitted opt-in development physics may pass deterministic
engineering gates. This module CANNOT authorize registered Eval, alter
historical evidence, or issue an independent physics-reviewer signature.
"""
from __future__ import annotations

from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import re
import subprocess

SCHEMA="wp3_v4_source_bound_technical_acceptance_v1"
TRAIN="fe83f517479f3b990028d6409304fae83b8199a9"
HANDOFF="ce0e55d7a428826b2d4ed3218641105bd0206f4b"
VERSION="spatial_road_edge_partial_lifecycle_v4"
EXPECTED=(
    ("C-seed0","right-turn-v1","nominal",0,False,"physical_crash_development"),
    ("A-seed1","ramp-v1","randomized_train",17,True,None),
    ("C-seed0","slalom-v1","nominal",2,False,"physical_crash_development"),
)
FILES=(
    "f1rl/sandbox/spatial_contact_dynamics.py",
    "f1rl/sandbox/spatial_partial_wheel_contact.py",
    "f1rl/sandbox/wheel_edge_release_energy.py",
    "f1rl/sandbox/wheel_bump_stop_impact.py",
    "f1rl/sandbox/accepted_contact_transition.py",
    "f1rl/sandbox/accepted_road_body_contact.py",
    "f1rl/sandbox/physical_crash.py",
    "f1rl/sandbox/certified_face_impact.py",
    "f1rl/sandbox/obb_feature_contact.py",
    "f1rl/sandbox/rigid_body_impact_response.py",
    "f1rl/sandbox/ballistic_body_impact.py",
    "f1rl/sandbox/swept_surface_clearance.py",
    "f1rl/sandbox/world.py",
    "f1rl/runtime/physics.py",
    "f1rl/runtime/simulation.py",
    "f1rl/runtime/road_crash_validation.py",
    "f1rl/learning/gym_environment.py",
    "f1rl/learning/wp7_race.py",
    "f1rl/learning/evaluation/evaluator.py",
    "f1rl/learning/evaluation/crash_contract.py",
    "f1rl/learning/training/env_factory.py",
)
TRACKS=("straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1")


def digest_file(path:Path)->str:
    h=sha256()
    with path.open("rb") as reader:
        for part in iter(lambda:reader.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def canonical(value:object)->bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      allow_nan=False,ensure_ascii=False).encode("utf-8")


def hash_manifest(obj:object)->str:
    return sha256(canonical(obj)).hexdigest()


def _json(path:Path)->dict:
    def reject_duplicate(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError("duplicate acceptance or evidence key")
            d[k]=v
        return d
    item=json.loads(path.read_text("utf-8"),object_pairs_hook=reject_duplicate)
    if type(item) is not dict:raise ValueError("malformed evidence document")
    return item


def source_bound_acceptance(root:Path,history:Path,replay_file:Path, *,
                            test_receipt_file:Path)->dict:
    root=Path(root).resolve(strict=True)
    history=Path(history).resolve(strict=True)
    replay_file=Path(replay_file).resolve(strict=True)
    actual_head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                                 text=True).strip()
    ancestry=subprocess.run(["git","-C",str(root),"merge-base","--is-ancestor",
                             HANDOFF,actual_head],capture_output=True)
    if ancestry.returncode!=0:
        raise ValueError("archived original handoff commit is not the current source ancestor")
    # Historic acceptance was authored on the UNCOMMITTED handoff worktree.
    # Keep its original provenance, never substitute the later clean HEAD.
    head=HANDOFF
    source_hashes={relative:digest_file(root/relative) for relative in FILES}
    if len(source_hashes)!=len(FILES) or not all(len(value)==64 for value in source_hashes.values()):
        raise ValueError("physics source bundle cannot be authenticated")
    # Version isolation: historical baseline remains the default; v4 opt-in
    # and explicit development-only crash registration are visible in source.
    contract={
       "v4_dispatch_explicit":VERSION in (root/"f1rl/sandbox/world.py").read_text("utf-8"),
       "v3_version_preserved":"spatial_rotating_compliant_lifecycle_v3" in
           (root/"f1rl/runtime/physics.py").read_text("utf-8"),
       "v4_integrator_versioned":"spatial_road_edge_partial_lifecycle_rk4_v4" in
           (root/"f1rl/sandbox/spatial_contact_dynamics.py").read_text("utf-8"),
       "legacy_scoring_protected":"physical_crash_terminal" in
           (root/"f1rl/learning/evaluation/evaluator.py").read_text("utf-8"),
       "development_only_crash": "wp7_directed_lifecycle_crash_dev_v3" in
           (root/"f1rl/learning/evaluation/crash_contract.py").read_text("utf-8"),
    }
    if not all(contract.values()):raise ValueError("v4 version isolation invariant missing")
    roster_path=history/"evaluation/evaluation-roster.json"
    provenance_path=history/"WP3-EVAL-SOURCE-PROVENANCE.json"
    roster,prov=_json(roster_path),_json(provenance_path)
    if (roster.get("expected_cells")!=96 or roster.get("expected_episodes")!=5760
        or prov.get("evaluation_roster_sha256")!=roster.get("sha256")
        or prov.get("training_source_commit")!=TRAIN
        or len(prov.get("models",[]))!=12):
        raise ValueError("immutable trained checkpoint/roster protocol drift")
    weights={}
    for item in prov["models"]:
        mid=item["model_id"]
        actual=digest_file(history/"runs"/mid/"model-final.zip")
        if actual!=item["model_final_sha256"] or mid in weights:
            raise ValueError("one original model weight SHA does not match")
        weights[mid]=actual
    if len(weights)!=12:
        raise ValueError("not all frozen models were authenticated")
    replay=_json(replay_file)
    if (replay.get("schema_version")!="wp3_v4_unregistered_three_case_scratch_v1"
        or replay.get("source_base_commit")!=HANDOFF
        or replay.get("frozen_train_sha")!=TRAIN
        or replay.get("roster_sha256")!=roster["sha256"]
        or replay.get("roster_file_sha256")!=digest_file(roster_path)
        or replay.get("physics_profile")!=VERSION
        or replay.get("experiment_status")!="SCRATCH_NOT_REGISTERED"
        or replay.get("all_required_replays_scientifically_certified") is not False
        or replay.get("full_5760_eval_admission")!="BLOCKED"
        or replay.get("registered_evaluation_executed") is not False
        or replay.get("heldout_access") is not False
        or replay.get("original_evidence_mutated") is not False):
        raise ValueError("development replay falsely claims registered Eval")
    entries=replay.get("cases")
    if type(entries) is not list or len(entries)!=3:
        raise ValueError("three original case results required")
    cases=[]
    for item,(mid,track,stratum,episode,success,failure) in zip(entries,EXPECTED,strict=True):
        original=next((r for r in roster["seed_rows"] if
            (r["model_id"],r["track_id"],r["stratum"])==(mid,track,stratum)),None)
        if original is None:raise ValueError("source cell absent")
        seed=original["episodes"][episode]
        if (
            (item.get("model_id"),item.get("track"),item.get("stratum"),item.get("episode"))
             !=(mid,track,stratum,episode)
            or item.get("model_final_sha256")!=weights[mid]
            or item.get("env_seed")!=seed["environment_seed"]
            or item.get("policy_seed")!=seed["policy_seed"]
            or item.get("scratch_physics_version")!=VERSION
            or item.get("status")!="ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED"
            or item.get("scientific_evidence") is not False
            or item.get("success_flag_unaccepted") is not success
            or item.get("failure_reason")!=failure
        ):raise ValueError("incorrect model, seed, result or science status")
        cases.append({"model":mid,"track":track,"stratum":stratum,"episode":episode,
                      "source_model_sha256":weights[mid],
                      "engineer_replay_completed":True,
                      "scientifically_certified":False,
                      "classification":"DEV_VALID_FAILED_CRASH" if failure else "DEV_FINISH"})
    test_receipt_file=Path(test_receipt_file).resolve(strict=True)
    receipt=test_receipt_file.read_text("utf-8-sig")
    match=re.search(r"(?m)^([0-9]+) passed in [0-9]+(?:\.[0-9]+)?s\s*$",receipt)
    if match is None or "FAILED" in receipt or "ERROR" in receipt or "Traceback" in receipt:
        raise ValueError("no authentic zero-failure pytest receipt in expected format")
    test_passed=int(match.group(1))
    if test_passed<300:
        raise ValueError("technical acceptance suite unexpectedly incomplete")
    test_failed=0
    track_hashes={
        track:digest_file(root/"assets/tracks/training"/(track+".json"))
        for track in TRACKS
    }
    dependency_versions={}
    for name in ("numpy","torch","stable-baselines3","gymnasium","pytest"):
        try:dependency_versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:dependency_versions[name]=None
    report={
      "schema_version":SCHEMA,"created_at_utc":datetime.now(timezone.utc).isoformat(),
      "scope":"USER_REQUESTED_SOURCE_BOUND_TECHNICAL_PREFLIGHT_NOT_INDEPENDENT_SCIENTIFIC_SIGNOFF",
      "environment":{"python":platform.python_version(),
                     "os":platform.platform(),
                     "dependencies":dependency_versions},
      "original_handoff_git_head":head,"frozen_train_commit":TRAIN,
      "physics_contract":VERSION,"baseline_v3_unchanged_as_default":True,
      "source_tree_clean":False,  # Immutable value AT ORIGINAL RECEIPT CREATION.
      "source_bindings_sha256":source_hashes,
      "physics_source_bundle_sha256":hash_manifest(source_hashes),
      "version_guards":contract,
      "training_track_sha256":track_hashes,
      "original_roster_sha256":roster["sha256"],
      "original_roster_file_sha256":digest_file(roster_path),
      "original_provenance_file_sha256":digest_file(provenance_path),
      "registered_final_model_count":12,
      "original_final_weight_sha256":weights,
      "replay_file_sha256":digest_file(replay_file),
      "three_exact_replays":cases,
      "engineering_replay_3_of_3_completed":True,
      "focused_physics_evaluator_tests_passed":test_passed,
      "focused_physics_evaluator_tests_failed":test_failed,
      "engineering_test_status":"PASS",
      "engineering_pytest_receipt_sha256":digest_file(test_receipt_file),
      "independent_reviewer_signature":None,
      "independent_domain_physics_review_completed":False,
      "physics_parameters_empirically_calibrated":False,
      "owner_scientific_physics_signoff":False,
      "approved_physical_scoring_contract":False,
      "production_and_registered_eval_admitted":False,
      "overall_status":"TECHNICAL_GATE_PASS_SCIENTIFIC_SIGNOFF_PENDING",
      "full_5760_eval_started":False,
      "new_scientific_eval_episodes":0,
      "train_or_heldout_access":False,
      "note":"300 source tests and 3 source-bound frozen replay cases demonstrate development execution only, not calibrated/independently reviewed physics or authorization of scored episodes."
    }
    report["record_sha256"]=hash_manifest(report)
    return report
