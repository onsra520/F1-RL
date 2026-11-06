"""Source-bound TRAIN-5 v2 *proposal* and non-training preview (fail-closed).

This is an independent versioned protocol distinct from historical TRAIN-4.
No code path here grants training/evaluation admission. It reads historical
HELD-OUT *manifest metadata* ONLY; never evaluation/*.json geometry.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from ..learning.pilot_preflight import TRAIN_IDS
from ..learning.reconstruction import RecoveryCandidate, build_training_scenarios
from ..learning.training.env_factory import TrainingScenario, TrainingScenarioSet
from ..sandbox.geometry.vector import Vec3
from ..sandbox.ballistic_flight import FlightState
from ..sandbox.independent_suspension_dynamics import SuspensionDynamicsState
from ..track_assets import load_track_manifest
from .wp3_eval_only_coordinator import apply_runtime_profile, FULL06_RUNTIME_PROFILE

SCHEMA="wp7_train5_paired_v2_source_bound_proposal_v1"
REGISTRATION="configs/research/wp7-train5-paired-v2-registration.json"
MANIFEST="assets/tracks/manifest-five-track-paired-v2.json"
CANDIDATE="configs/recovery/wp7-c2-five-track-paired-v2.yml"
EXPERIMENT="configs/experiments/ppo-generalization-five-track-paired-v2.yml"
TRAIN5_IDS=("straight-sprint-v1","slalom-v1","right-turn-train-v2",
            "ramp-v1","left-turn-train-v2")
HELDOUT_IDS=("left-turn-v1","offset-slalom-v1")
HISTORICAL_IDS=tuple(TRAIN_IDS)
MODEL_IDS=tuple(f"{cohort}-seed{i}" for cohort in "ABCD" for i in range(3))
PROPOSED={
    "train5_from_scratch": (
      {"ppo":2026111001,"sampling":2026111101,"perturbation":2026111201},
      {"ppo":2026111002,"sampling":2026111102,"perturbation":2026111202},
      {"ppo":2026111003,"sampling":2026111103,"perturbation":2026111203},
    ),
    "train5_from_frozen_checkpoint": (
      {"ppo":2026112001,"sampling":2026112101,"perturbation":2026112201},
      {"ppo":2026112002,"sampling":2026112102,"perturbation":2026112202},
      {"ppo":2026112003,"sampling":2026112103,"perturbation":2026112203},
    ),
}
# Bind both training assets, frozen reference and the runtime implementation.
# Adding a new dependency requires a NEW registration revision and digest.
SOURCE_FILES=(
 "assets/tracks/manifest.json",
 "assets/tracks/manifest-five-track-paired-v2.json",
 "assets/tracks/training/straight-sprint-v1.json",
 "assets/tracks/training/slalom-v1.json",
 "assets/tracks/training/right-turn-v1.json",
 "assets/tracks/training/right-turn-train-v2.json",
 "assets/tracks/training/ramp-v1.json",
 "assets/tracks/training/left-turn-train-v2.json",
 "configs/experiments/ppo-generalization-v1.yml",
 "configs/experiments/ppo-generalization-five-track-paired-v2.yml",
 "configs/recovery/wp7-c2-candidate.yml",
 "configs/recovery/wp7-c2-five-track-paired-v2.yml",
 "configs/research/wp7-spatial-ground-protocol-v1.json",
 "configs/research/wp7-rotating-owner-protocol-v3.json",
 "configs/research/wp7-hard-route-outcome-v2.json",
 "configs/recovery/wp7-pilot-200k.yml",
 "f1rl/learning/pilot_preflight.py",
 "f1rl/learning/reconstruction.py",
 "f1rl/learning/experiments/split.py",
 "f1rl/learning/training/env_factory.py",
 "f1rl/learning/gym_environment.py",
 "f1rl/learning/wp7_race.py",
 "f1rl/learning/crash_terminal_registered.py",
 "f1rl/runtime/physics.py",
 "f1rl/research/wp7_track_feasibility.py",
 "f1rl/research/wp7_protocol.py",
 "f1rl/research/wp7_cell_evaluation.py",
 "f1rl/cli/diagnostics/vector_benchmark_v4.py",
 "src/cli/diagnostics/diagnose_driving_v4.py",
 "src/cli/evaluation/compare_driving_v4.py",
 "f1rl/cli/training/exploratory_demo_launcher_v4.py",
 "f1rl/cli/evaluation/wp3_eval_only_coordinator_v4.py",
 "f1rl/cli/evaluation/wp3_owner_train_dev_eval_v4.py",
 "f1rl/cli/diagnostics/wp7_certify_planar_execution_v4.py",
 "src/cli/internal/wp7_ppo_fixture_worker_v4.py",
 "f1rl/cli/evidence/wp7_train5_ci_preflight.py",
 "f1rl/cli/evidence/wp7_training_admission_v4.py",
 "f1rl/cli/training/wp7_train_acceptance_v4.py",
 "f1rl/cli/evidence/wp7_verify_run_evidence_v4.py",
 "docs/.superpowers/handoff/evidence/step3-original-wp7-protocol.py.txt",
 "f1rl/research/wp3_eval_only_coordinator.py",
 "f1rl/research/wp7_executed_feasibility.py",
 "f1rl/research/wp7_train5_v2_protocol.py",
 "docs/.superpowers/handoff/evidence/step3-original-dual-turn-runner.py.txt",
 "f1rl/cli/evidence/wp7_dual_turn_v4_feasibility.py",
 "docs/.superpowers/handoff/evidence/wp7-historical-gpu-probe.py.txt",
 "src/cli/diagnostics/wp7_capture_gpu_v1.py",
 "pyproject.toml",
)+tuple(f"models/{m}/model-final.zip" for m in MODEL_IDS)

EVIDENCE_FILES=(
 "STEP3-PHYSICS-V4-DUAL-TURN-ACCEPTANCE.json",
 "STEP3-DUAL-TURN-PHYSICS-V4-PORTABLE-WITNESS.json",
 "STEP3-INDEPENDENT-V4-CONTROLS/right-v2-portable-attempt02/summary.json",
 "STEP3-INDEPENDENT-V4-CONTROLS/left-v2-portable-attempt02/summary.json",
)
RUNTIME_SPEC={
 "physics_version":"spatial_road_edge_partial_lifecycle_v4",
 "race_version":"wp7_directed_lifecycle_crash_contract_v4",
 "terminal_contract":"wp3_physical_crash_terminal_v1",
 "max_abs_tilt_rad":.7,
}

class Train5PreflightError(ValueError):
    """No geometry, reset, training, smoke or HELD-OUT execution is authorized."""


def _hash_bytes(data:bytes)->str:
    return sha256(data).hexdigest()


def _digest(doc:object)->str:
    return _hash_bytes(json.dumps(doc,sort_keys=True,allow_nan=False,
                                 separators=(",",":")).encode("utf-8"))


def _unique(pairs):
    obj={}
    for name,value in pairs:
        if name in obj:raise Train5PreflightError("duplicate protocol key")
        obj[name]=value
    return obj


def _in_root(root:Path, name:str)->Path:
    if type(name) is not str or not name or name.startswith(("/", "\\")) or ".." in Path(name).parts:
        raise Train5PreflightError("invalid relative source name")
    p=root/name
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
        raise Train5PreflightError("missing, symlinked, or escaping source: "+name)
    return p


def _assert(cond:bool, text:str):
    if not cond:raise Train5PreflightError(text)


def _manifest_contract(root:Path):
    old=json.loads(_in_root(root,"assets/tracks/manifest.json").read_text("utf8"),
                   object_pairs_hook=_unique)
    new=json.loads(_in_root(root,MANIFEST).read_text("utf8"),object_pairs_hook=_unique)
    _assert(type(old) is dict and type(new) is dict
        and set(old)==set(new)=={"schema_version","tracks"}
        and old["schema_version"]==new["schema_version"]==1,
        "historical/paired manifest schema changed")
    oldrows=old["tracks"]
    rows=new["tracks"]
    _assert(type(rows) is list and type(oldrows) is list
        and len(oldrows)==6 and len(rows)==7,
        "unexpected historic/paired manifest length")
    _assert(rows[:2]==oldrows[:2] and rows[3]==oldrows[3]
        and rows[5:]==oldrows[4:],
        "historical TRAIN-3/HELD-OUT metadata changed")
    _assert(tuple(x["track_id"] for x in rows[:5])==TRAIN5_IDS
            and tuple(x["track_id"] for x in rows[5:])==HELDOUT_IDS,
            "TRAIN-5 ordering or HELD-OUT IDs altered")
    for row,expected in ((rows[2],"training/right-turn-train-v2.json"),
                         (rows[4],"training/left-turn-train-v2.json")):
        _assert(set(row)=={"track_id","path","world_sha256","content_sha256"}
                and row["path"]==expected,
                "TRAIN-v2 asset identity/path mismatch")
    for field in ("world_sha256","content_sha256"):
        vals=[row[field] for row in rows]
        _assert(len(vals)==len(set(vals)),
                "TRAIN-5 must not collide with original TRAIN or HELD-OUT metadata")
    return rows


def _seed_contract(reg:dict, previous:dict):
    _assert(reg.get("seed_audit_status")=="PROPOSED_UNVERIFIED_GLOBAL"
            and reg.get("training_permission") is False
            and reg.get("smoke_permission") is False
            and reg.get("heldout_permission") is False
            and reg.get("owner_approval") is None
            and reg.get("global_seed_registry_receipt") is None,
            "seed provenance or owner approval not in fail-closed draft state")
    seeds=reg.get("proposed_training_seeds")
    _assert(type(seeds) is dict and set(seeds)==set(PROPOSED),
            "proposed seed cohorts changed")
    allvalues=[]
    oldvalues=[int(v) for row in previous["seed_roster"] for v in row.values()]
    for cohort,expected in PROPOSED.items():
        roster=seeds[cohort]
        _assert(type(roster) is list and len(roster)==len(expected),
                "train seed roster length changed")
        for i,row in enumerate(roster):
            _assert(type(row) is dict and row==expected[i],
                    "train seed identity differs from proposal")
            allvalues.extend(row.values())
    _assert(all(type(x) is int and 0<=x<2**32 for x in allvalues)
            and len(set(allvalues))==len(allvalues)
            and not set(allvalues).intersection(oldvalues),
            "seed reused, malformed or out of uint32 range")
    # The global audit is NOT inferred from repository-local metadata.


def _evidence_contract(repo_root:Path,evidence_root:Path, hashes:dict, rows:list,source_hashes:dict):
    root=evidence_root.resolve(strict=True)
    _assert(not root.is_symlink() and type(hashes) is dict
            and set(hashes)==set(EVIDENCE_FILES),
            "missing external Step3 evidence hash bindings")
    for name in EVIDENCE_FILES:
        file=_in_root(root,name)
        _assert(_hash_bytes(file.read_bytes())==hashes[name],
                "Step3 evidence digest mismatch: "+name)
    receipt=json.loads((root/EVIDENCE_FILES[0]).read_text("utf-8"),
                       object_pairs_hook=_unique)
    witness=json.loads((root/EVIDENCE_FILES[1]).read_text("utf-8"),
                       object_pairs_hook=_unique)
    _assert(receipt.get("success_in_both_nominal_cases") is True
            and receipt.get("zero_crash_and_no_physics_exceptions") is True
            and receipt.get("portable_replay_verified") is True
            and receipt.get("models_loaded_for_step3") is False
            and receipt.get("heldout_geometry_read") is False
            and receipt.get("ppo_training_run") is False
            and witness.get("verdict")=="PASS_TWO_REAL_INDEPENDENT_PHYSICS_V4_FINISHES",
            "Step3 source/witness did not certify required real v4 finishes")
    _assert(receipt.get("runner_source_script_sha256")==
                source_hashes["docs/.superpowers/handoff/evidence/step3-original-dual-turn-runner.py.txt"]
            and receipt.get("runtime_profile")==RUNTIME_SPEC["physics_version"]
            and receipt.get("race_version")==RUNTIME_SPEC["race_version"]
            and receipt.get("terminal_contract")==RUNTIME_SPEC["terminal_contract"]
            and receipt.get("scoped_pytest")=={
                "tests":169,"failures":0,"errors":0,"skipped":0},
            "Step3 runner/source/runtime or regression proof mismatch")
    for side,index in (("right",2),("left",4)):
        c=receipt["independent_controllers"][side]
        raw=json.loads((root/EVIDENCE_FILES[2 if side=="right" else 3]).read_text("utf8"),
                       object_pairs_hook=_unique)
        ep=raw["controller_episode"]
        replay=raw["replay_episode"]
        track_file="assets/tracks/training/"+rows[index]["track_id"]+".json"
        _assert(c["track_id"]==rows[index]["track_id"]
                and raw["track_id"]==rows[index]["track_id"]
                and raw["source_bindings"].get(track_file)==source_hashes[track_file]
                and c["source_summary_sha256"]==hashes[EVIDENCE_FILES[2 if side=="right" else 3]],
                "Step3 witness track/source binding mismatch")
        # The historical Step3 witness used the original WP7 protocol.
        # Compare to an immutable exact archived copy, not the later migrated
        # source that has a different GPU-probe path.
        step3_bindings=dict(raw["source_bindings"])
        original_protocol=step3_bindings.pop("f1rl/research/wp7_protocol.py",None)
        _assert(original_protocol==source_hashes[
                    "docs/.superpowers/handoff/evidence/step3-original-wp7-protocol.py.txt"],
                "Step3 original protocol source differs from archived input")
        _assert(all(name in source_hashes and value==source_hashes[name]
                    for name,value in step3_bindings.items()),
                "Step3 controller used unregistered/stale source")
        _assert(ep["finished"] is True and ep["exception"] is None
                and ep["checkpoints_completed"]==["checkpoint-0","checkpoint-1"]
                and ep["research_outcome"]["kind"]=="successful_finish"
                and ep["research_outcome"]["valid_sample"] is True
                and raw["replay_exact_match"] is True
                and ep["tick_rows_sha256"]==replay["tick_rows_sha256"]
                and raw["heldout_geometry_file_opened"] is False
                and raw["ppo_models_loaded"] is False,
                "Step3 real finish/replay does not match certified source")
    return receipt


def validate_preview(root:Path, *, registration_file:Path, evidence_root:Path,
                     check_evidence:bool=True)->dict:
    """Validate all source-bound inputs BEFORE touching any TRAIN geometry.

    This authorizes metadata inspection / engineering factory preview only.
    Rejection never invokes the simulator or a learner.
    """
    root=Path(root).resolve(strict=True)
    regpath=Path(registration_file).resolve(strict=True)
    _assert(regpath==(root/REGISTRATION).resolve(),
            "unexpected protocol location (not source-bound)")
    reg=json.loads(regpath.read_text(encoding="utf8"),object_pairs_hook=_unique)
    required={"schema_version","status","baseline_head","split_identity","train_ids",
              "heldout_ids","candidate","experiment","manifest",
              "runtime_profile","source_sha256","evidence_sha256",
              "seed_audit_status","training_permission","smoke_permission",
              "heldout_permission","owner_approval","global_seed_registry_receipt",
              "proposed_training_seeds","proposed_design","preflight_digest"}
    _assert(set(reg)==required and reg["schema_version"]==SCHEMA
            and reg["status"]=="draft_not_authorized"
            and reg["baseline_head"]=="1de279c37e62a27531d661480d20fa6e095c4e84"
            and reg["split_identity"]=="five_train_paired_v2_heldout_metadata_only"
            and reg["candidate"]==CANDIDATE and reg["experiment"]==EXPERIMENT
            and reg["manifest"]==MANIFEST
            and reg["train_ids"]==list(TRAIN5_IDS)
            and reg["heldout_ids"]==list(HELDOUT_IDS)
            and reg["runtime_profile"]==RUNTIME_SPEC,
            "TRAIN-5 v2 registration fields/roster/mode drifted")
    _assert(reg["preflight_digest"]==_digest({k:v for k,v in reg.items()
                                           if k!="preflight_digest"}),
            "registration digest invalid")
    design=reg["proposed_design"]
    _assert(type(design) is dict
            and design=={"cohorts":["train5_from_scratch","train5_from_frozen_checkpoint"],
              "replicates_per_cohort":3,"strata":["nominal","randomized_train"],
              "development_episodes_per_cell":60,"proposed_development_cells":60,
              "proposed_development_episodes":3600,
              "checkpoint_selection":"final_model_only",
              "evaluation_seeds":"sha256(versioned_proposal,policy,track,stratum,episode)",
              "full06_and_heldout":"not_part_of_training_registration",
              "status":"proposal_only_no_episodes_executed"},
            "train/evaluation design needs a new source-bound revision")
    _assert(type(reg["source_sha256"]) is dict
            and set(reg["source_sha256"])==set(SOURCE_FILES),
            "source lock missing or contains extra dependencies")
    for name in SOURCE_FILES:
        p=_in_root(root,name)
        _assert(_hash_bytes(p.read_bytes())==reg["source_sha256"][name],
                "locked source hash mismatch: "+name)
    _manifest_contract(root)
    old_reg=json.loads(_in_root(root,"configs/research/wp7-rotating-owner-protocol-v3.json").read_text("utf8"),
                       object_pairs_hook=_unique)
    _seed_contract(reg,old_reg)
    if check_evidence:
        _evidence_contract(root,Path(evidence_root),reg["evidence_sha256"],
                           _manifest_contract(root),reg["source_sha256"])
    return reg


def preview_train5_v2(root:Path, *, registration_file:Path, evidence_root:Path)->TrainingScenarioSet:
    """Construct five TRAIN engineering factories. NO train reset/admission."""
    from .wp7_protocol import load_registration,prepare_scenarios
    registration=validate_preview(root,registration_file=registration_file,
                                  evidence_root=evidence_root)
    root=Path(root).resolve(strict=True)
    candidate=RecoveryCandidate.from_yaml(_in_root(root,CANDIDATE))
    exp=candidate.experiment
    _assert(exp.metadata.id=="ppo-generalization-five-track-paired-v2"
            and exp.metadata.status=="draft"
            and exp.training.protocol_status=="pilot_owned"
            and exp.tracks.training_ids==TRAIN5_IDS
            and exp.tracks.heldout_ids==HELDOUT_IDS
            and exp.tracks.manifest.resolve()==(root/MANIFEST).resolve(),
            "candidate/experiment/manifest not matching registered v2 values")
    raw=build_training_scenarios(candidate)
    old_owner=load_registration(root,profile="spatial_rotating_owner_v3")
    template=next(x.factory for x in prepare_scenarios(
        root,old_owner,seed_index=0).scenarios if x.track_id=="right-turn-v1")
    height=template.vehicle_parameters.ride_height
    scenarios=[]
    for row in raw.scenarios:
        initial=replace(row.factory.initial_state,
          vehicle=replace(row.factory.initial_state.vehicle,
             position=row.factory.initial_state.vehicle.position+Vec3(0,height,0)),
          suspension=SuspensionDynamicsState(
             row.factory.initial_state.vehicle.position.y+height),
          flight=FlightState("grounded"))
        factory=replace(template,track=row.factory.track,navigation=row.factory.navigation,
            initial_state=initial,reset_config=None,reset_track_id=None)
        factory,provenance=apply_runtime_profile(factory,FULL06_RUNTIME_PROFILE)
        _assert(provenance["effective"]==RUNTIME_SPEC
                and factory.requires_scientific_admission
                and factory.reset_config is None,
                "nominal v4 engineering factory mismatch")
        scenarios.append(TrainingScenario(row.track_id,factory))
    _assert(tuple(x.track_id for x in scenarios)==TRAIN5_IDS,
            "five TRAIN-only source-bound scenario identities required")
    return TrainingScenarioSet(tuple(scenarios))


def require_training_admission(*args,**kwargs):
    """No draft proposal can authorize full PPO training, even with forged flags."""
    raise PermissionError("TRAIN-5 v2 scientific admission CLOSED; owner/seed/global audit pending")


def require_smoke_admission(*args,**kwargs):
    """No automatic six-case PPO smoke from a draft source registration."""
    raise PermissionError("TRAIN-5 v2 PPO smoke CLOSED pending explicit approval")


def require_heldout_admission(*args,**kwargs):
    raise PermissionError("HELD-OUT remains CLOSED; not part of TRAIN-5 preview")
