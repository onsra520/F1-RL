"""WP7 recovery acceptance audit; read-only retrospective evidence, fail-closed.

This is NOT experiment freezing, cannot authorize HELD-OUT, and does not
invent historical training provenance. Existing source/model/audit artifacts
remain immutable. Any evidence mismatch aborts report construction.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from .pilot_milestones import verify_pilot_evidence_manifest
from .pilot_preflight import PILOT_MILESTONES, TRAIN_IDS, SEED_PAIRS, build_pilot_preflight
from src.training.config import TrainingRunConfig


KNOWN_AUDIT_SHA256 = {
    "trajectory_xz": "0e44bcaf738f5c4eadb2f726b3a8108af64d1690f7a9fbf28ccbdabcd771d188",
    "swept_3d_proxy": "da1accdd0d0f5bdafb7e790466f8c30a6a054eb637b9c1dc6820b2114e2bfed8",
    "right_turn_road_union": "cdf22fc26b9fdee85d82c56dea0d26c900cc5e318010d989df3fca51f1049a62",
    "stochastic_train_screen": "ce7b86c1869aa60766eb3565f343b967c5a76285cc90a22b355fc1dc367b196b",
}

AUDIT_ONLY_PATHS = frozenset({
    "docs/research/wp7-dual-seed-train-trajectory-audit-2026-10-12.md",
    "docs/research/wp7-dual-seed-3d-swept-body-proxy-2026-10-12.md",
    "docs/research/wp7-retrospective-freeze-readiness-2026-10-13.md",
    "f1rl/learning/trajectory_audit.py",
    "f1rl/learning/swept_volume_audit.py",
    "f1rl/learning/freeze_readiness.py",
    "f1rl/learning/corridor_audit.py",
    "scripts/wp7_audit_train_trajectories.py",
    "scripts/wp7_audit_3d_swept_volume.py",
    "scripts/wp7_freeze_readiness.py",
    "scripts/wp7_audit_right_turn_corridor.py",
    "tests/learning/test_trajectory_audit_reconstruction.py",
    "tests/learning/test_swept_volume_audit_reconstruction.py",
    "tests/learning/test_freeze_readiness_reconstruction.py",
    "tests/learning/test_corridor_audit_reconstruction.py",
    "docs/research/wp7-right-turn-centerline-corridor-2026-10-13.md",
    "docs/research/wp7-geometry-and-stochastic-screen-2026-10-13.md",
    "docs/research/wp0-wp7-deep-audit-2026-10-13.md",
    "configs/recovery/wp7-vehicle-geometry-review.yml",
    "f1rl/learning/geometry_review.py",
    "f1rl/learning/robustness_screen.py",
    "scripts/wp7_screen_train_action_sensitivity.py",
    "tests/learning/test_wp7_robustness_screen_reconstruction.py",
})


def verify_training_source_diff(changed_files: tuple[str, ...]) -> tuple[str, ...]:
    """Allow only audit additions since training; never silently re-evaluate changed physics."""
    unknown=sorted(set(changed_files)-AUDIT_ONLY_PATHS)
    if unknown:
        raise ValueError("training/simulator source changed after archived run: "+", ".join(unknown))
    return tuple(sorted(set(changed_files)))


def _require_clean_git(root: Path) -> None:
    """Reject dirty inspection sources; no hidden local simulator modifications."""
    try:
        status=subprocess.run(
            ["git","-C",str(root),"status","--porcelain"],
            capture_output=True,text=True,encoding="utf-8",check=True,
        )
    except (OSError,subprocess.CalledProcessError) as exc:
        raise ValueError("cannot inspect source Git cleanliness") from exc
    if status.stdout.strip():
        raise ValueError("readiness audit requires a clean Git worktree")


def _diff_since_training(root: Path, training_commit: str) -> tuple[str, ...]:
    try:
        result=subprocess.run(
            ["git","-C",str(root),"diff","--name-only",training_commit,"HEAD"],
            capture_output=True,text=True,encoding="utf-8",check=True,
        )
    except (OSError,subprocess.CalledProcessError) as exc:
        raise ValueError("archived training source Git commit is unavailable") from exc
    return verify_training_source_diff(tuple(result.stdout.splitlines()))


BLOCKERS = (
    "physical_body_geometry_not_registered: original vehicle state lacks a declared 3D body collision hull",
    "point_only_collision_engine: existing 4/4 success uses post-step centre occupancy instead of body-aware continuous collision",
    "right_turn_corridor_not_registered: 5.511796m calibration and 4.415475m confirmation max deviation have no pre-approved corner-cutting tolerance",
    "training_time_environment_provenance_missing: both state.json provenance_file fields are null; inspection-time dependency versions cannot reconstruct history",
    "stochastic_action_screen_failed: 16-episode TRAIN sensitivity produced 12/16 Right Turn for calibration and 15/16 Slalom/Right Turn for confirmation; not environment perturbation robustness",
    "independent_backup_not_verified: audits and training archives share a local drive, off-device backup unproven",
)


def _obj(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be JSON object")
    return value


def _json(path: Path) -> dict[str, Any]:
    return _obj(json.loads(path.read_text(encoding="utf-8")), str(path))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate_completed_seed_state(
    state: dict[str, Any],
    *,
    expected_run_id: str,
    expected_ppo_seed: int,
    expected_sampling_seed: int,
    expected_profile: TrainingRunConfig,
) -> None:
    """Strict run identity/config check without needing model or external files."""
    if (
        state.get("run_id") != expected_run_id
        or state.get("status") != "completed"
        or state.get("mode") != "multitrack"
        or state.get("requested_total_timesteps") != 200_000
        or not isinstance(state.get("actual_total_timesteps"), int)
        or isinstance(state["actual_total_timesteps"], bool)
        or not 200_000 <= state["actual_total_timesteps"] < 225_000
        or state.get("checkpoint_every_steps") != 25_000
        or state.get("training_tracks") != list(TRAIN_IDS)
        or state.get("sampling_seed") != expected_sampling_seed
        or state.get("latest_checkpoint") !=
           f"checkpoints/ppo-multitrack-seed{expected_ppo_seed}-sample{expected_sampling_seed}-step175000.zip"
        or state.get("final_model") !=
           f"ppo-multitrack-seed{expected_ppo_seed}-sample{expected_sampling_seed}-final-model.zip"
    ):
        raise ValueError("completed pilot run identity/state does not match registered protocol")
    expected_ppo=asdict(expected_profile.ppo)
    expected_ppo["seed"]=expected_ppo_seed
    if (state.get("ppo") != expected_ppo
        or state.get("vector_env") != asdict(expected_profile.vector_env)):
        raise ValueError("archived PPO/vector config differs from registered pilot profile")


def _check_source_fingerprints(root:Path,manifest:dict[str,Any],expected:list[dict[str,str]])->None:
    entries=manifest.get("source_fingerprints")
    if entries!=expected:
        raise ValueError("archived source fingerprint declarations drifted")
    for item in expected:
        file=(root/item["path"]).resolve(strict=True)
        if not file.is_relative_to(root) or _sha(file)!=item["sha256"]:
            raise ValueError("source fingerprint no longer matches "+item["path"])


def _verify_single(
    *,root:Path, run:Path,seed:int,role:str,
    profile:TrainingRunConfig,expected_fingerprints:list[dict[str,str]],
)->dict[str,Any]:
    integrity=verify_pilot_evidence_manifest(run)
    if (integrity.get("status")!="verified_exploratory_evidence_not_scientific"
        or integrity.get("verified_milestones")!=len(PILOT_MILESTONES)
        or integrity.get("heldout_rollouts")!=0
        or integrity.get("scientific_evidence") is not False):
        raise ValueError("pilot run integrity must verify all eight exploratory milestones")
    state=_json(run/"state.json")
    validate_completed_seed_state(
        state,expected_run_id=run.name,expected_ppo_seed=seed,
        expected_sampling_seed=seed,expected_profile=profile,
    )
    manifest=_json(run/"evaluations"/"manifest.json")
    _check_source_fingerprints(root,manifest,expected_fingerprints)
    last=_json(run/"evaluations"/"milestone-200000.json")
    evaluator=_obj(last.get("evaluation"),"checkpoint evaluator")
    if (last.get("scientific_evidence") is not False
        or last.get("status")!="exploratory_unreviewed_not_scientific"
        or last.get("milestone_timesteps")!=200_000
        or last.get("actual_timesteps")!=state["actual_total_timesteps"]
        or last.get("source_fingerprints")!=expected_fingerprints
        or evaluator.get("policy_source")!="ppo_archive_requires_run_provenance_review"
        or evaluator.get("track_ids")!=list(TRAIN_IDS)
        or evaluator.get("heldout_rollouts")!=0
        or evaluator.get("scientific_evidence") is not False
        or evaluator.get("episode_count_per_track")!=1):
        raise ValueError("final checkpoint evaluation contract drifted")
    tracks=evaluator.get("track_summaries")
    if (
        not isinstance(tracks,list) or len(tracks)!=len(TRAIN_IDS)
        or [t.get("track_id") if isinstance(t,dict) else None for t in tracks]!=list(TRAIN_IDS)
        or any(t.get("episode_count")!=1 or t.get("success_count")!=1 or
               t.get("mean_collision_count")!=0 for t in tracks)
    ):
        raise ValueError("final archived 4/4 TRAIN result not independently replicated in summaries")
    model=(run/last["model_path"]).resolve(strict=True)
    if not model.is_relative_to(run) or model.name!=state["final_model"] or _sha(model)!=last["checkpoint_sha256"]:
        raise ValueError("final model hash/identity drifted")
    return {
        "role":role,"run_id":run.name,"ppo_seed":seed,"sampling_seed":seed,
        "requested_steps":state["requested_total_timesteps"],
        "actual_steps":state["actual_total_timesteps"],
        "training_track_ids":list(TRAIN_IDS),
        "deterministic_train_successes":len(TRAIN_IDS),
        "episodes_per_track":1,
        "verified_milestones":len(PILOT_MILESTONES),
        "model_sha256":last["checkpoint_sha256"],
        "state_sha256":_sha(run/"state.json"),
        "manifest_sha256":_sha(run/"evaluations"/"manifest.json"),
        "source_git_revision":last["git_commit"],
        "provenance_at_training": (
            "missing_unrecoverable_from_present_files" if state.get("provenance_file") is None
            else "original_provenance_declared_needs_separate_review"
        ),
    }


def validate_linked_audits(
    *, audit_type:str,document:dict[str,Any],records:list[dict[str,Any]],
)->dict[str,Any]:
    """Require exact two-run identity/hash/track boundary in archived derived audits."""
    expected_schema={
        "trajectory_xz":"wp7-recovery-dual-seed-trajectory-audit-v1",
        "swept_3d_proxy":"wp7-recovery-dual-seed-3d-body-proxy-audit-v1",
    }[audit_type]
    audits=document.get("audits")
    if (document.get("schema_version")!=expected_schema
        or document.get("scientific_evidence") is not False
        or document.get("heldout_rollouts")!=0
        or not isinstance(audits,list) or len(audits)!=2):
        raise ValueError("independent geometry audit outer schema/rollout boundary invalid")
    summaries=[]
    for item,record in zip(audits,records,strict=True):
        if (not isinstance(item,dict)
            or item.get("run_id")!=record["run_id"]
            or item.get("source_git_revision")!=record["source_git_revision"]
            or item.get("final_model_sha256")!=record["model_sha256"]
            or item.get("scientific_evidence") is not False
            or item.get("heldout_rollouts")!=0
            or (item.get("evidence_manifest_sha256") if audit_type=="trajectory_xz"
                else item.get("manifest_sha256"))!=record["manifest_sha256"]):
            raise ValueError("geometry audit does not match registered model/run/manifest")
        rows=item.get("results") if audit_type=="trajectory_xz" else item.get("track_results")
        if (not isinstance(rows,list) or len(rows)!=len(TRAIN_IDS)
            or [x.get("track_id") if isinstance(x,dict) else None for x in rows]!=list(TRAIN_IDS)
            or any(x.get("success") is not True or x.get("simulator_obstacle_hit_count")!=0 for x in rows)):
            raise ValueError("geometry audit TRAIN outcome/track identity mismatch")
        slalom=rows[1]
        if audit_type=="trajectory_xz":
            gap=slalom.get("min_continuous_xz_point_to_obstacle_m")
            corner=rows[2].get("max_horizontal_route_deviation_m")
            if not isinstance(gap,(float,int)) or gap<=0 or not isinstance(corner,(float,int)) or corner<=0:
                raise ValueError("missing or nonpositive XZ geometry observations")
            summaries.append({"run_id":record["run_id"],
                             "slalom_centerline_xz_gap_m":gap,
                             "right_turn_max_deviation_m":corner})
        else:
            proxies=slalom.get("profiles")
            if not isinstance(proxies,list) or len(proxies)!=3:
                raise ValueError("expected three predefined 3D proxy sensitivities")
            names=(
                "sensitivity_small_3p2x1p4x1p0",
                "sensitivity_medium_4p0x1p8x1p2",
                "sensitivity_large_5p0x2p2x1p4",
            )
            if [p.get("name") for p in proxies]!=list(names):
                raise ValueError("unexpected 3D body sensitivity profiles")
            if any(p.get("conservative_clearance_pass") is not True
                   for p in proxies[:2]):
                raise ValueError("small and medium 3D proxy clearance audit regressed")
            # Do not require the large proxy to pass: calibration is inconclusive.
            if proxies[2].get("conservative_clearance_pass") is not (record["ppo_seed"]==1):
                raise ValueError("recorded large proxy status changed from original evidence")
            summaries.append({
                "run_id":record["run_id"],
                "sensitivity_margins_m": {
                    p["name"]:p["min_conservative_gap_m"] for p in proxies
                },
                "large_proxy_result":(
                    "clearance_pass_for_assumed_geometry" if proxies[2]["conservative_clearance_pass"]
                    else "inconclusive_not_collision_proof"
                ),
            })
    return {"audit_type":audit_type,"runs":summaries}



def validate_right_turn_road_union(
    *,document:dict[str,Any],records:list[dict[str,Any]],
    xz_summaries:list[dict[str,Any]],
)->dict[str,Any]:
    """Link corridor diagnostics to exactly the original TRAIN models.

    A verified centre-on-road finding never clears body/route policy blockers.
    """
    entries=document.get("audits")
    if (document.get("schema_version")!="wp7-recovery-dual-seed-right-turn-road-union-audit-v1"
        or document.get("scientific_evidence") is not False
        or document.get("heldout_rollouts")!=0
        or not isinstance(entries,list) or len(entries)!=2
        or len(records)!=2 or len(xz_summaries)!=2):
        raise ValueError("right-turn corridor report boundary/schema invalid")
    summaries=[]
    for item,record,xz in zip(entries,records,xz_summaries,strict=True):
        if (not isinstance(item,dict)
            or item.get("run_id")!=record["run_id"]
            or item.get("source_git_revision")!=record["source_git_revision"]
            or item.get("source_manifest_sha256")!=record["manifest_sha256"]
            or item.get("final_model_sha256")!=record["model_sha256"]
            or item.get("track_id")!="right-turn-v1"
            or item.get("scientific_evidence") is not False
            or item.get("heldout_rollouts")!=0
            or item.get("success") is not True
            or item.get("road_patch_count")!=2
            or item.get("road_widths_m")!=[12.0,12.0]
            or item.get("is_physical_body_clearance") is not False
            or item.get("is_approved_corner_cutting_threshold") is not False):
            raise ValueError("right-turn road union model/protocol identity mismatch")
        gap=item.get("centerline_uncovered_road_path_xz_m")
        deviation=item.get("max_sampled_route_deviation_m")
        upper=item.get("upper_bound_route_deviation_between_samples_m")
        if (isinstance(gap,bool) or not isinstance(gap,(int,float)) or gap!=0.
            or item.get("centerline_outside_road_sample_count")!=0
            or isinstance(deviation,bool) or not isinstance(deviation,(int,float))
            or abs(deviation-xz["right_turn_max_deviation_m"])>2e-6
            or isinstance(upper,bool) or not isinstance(upper,(int,float))
            or upper<deviation or upper>deviation+2.0):
            raise ValueError("right-turn road-union centreline observations drifted")
        summaries.append({
            "run_id":record["run_id"],
            "continuous_centerline_outside_road_m":gap,
            "max_sampled_route_deviation_m":deviation,
            "between_sample_upper_bound_m":upper,
            "route_corner_cutting_policy":"not_approved",
            "body_footprint_policy":"not_registered",
        })
    return {"audit_type":"right_turn_road_union","runs":summaries}



def validate_stochastic_train_screen(
    *, document:dict[str,Any],records:list[dict[str,Any]],geometry_sha256:str,
)->dict[str,Any]:
    """Attest action-sensitivity evidence without confusing it with HELD-OUT."""
    from .robustness_screen import (
        EPISODES_PER_TRACK,PROTOCOL_ID,SCREEN_LOWER_BOUND,
        one_sided_clopper_pearson_lower,seed_for,
    )
    screens=document.get("screens")
    if (document.get("schema_version")!="wp7-recovery-dual-seed-action-sensitivity-v1"
        or document.get("status")!="exploratory_not_scientific"
        or document.get("scientific_evidence") is not False
        or document.get("heldout_rollouts")!=0
        or document.get("scientific_freeze_permitted") is not False
        or document.get("heldout_evaluation_permitted") is not False
        or document.get("geometry_review_status")!="unapproved"
        or document.get("geometry_review_sha256")!=geometry_sha256
        or not isinstance(screens,list) or len(screens)!=2):
        raise ValueError("stochastic screen metadata/geometry boundary invalid")
    summaries=[]
    for s,r in zip(screens,records,strict=True):
        if (not isinstance(s,dict)
            or s.get("schema_version")!="wp7-recovery-train-stochastic-screen-v1"
            or s.get("run_id")!=r["run_id"]
            or s.get("model_sha256")!=r["model_sha256"]
            or s.get("original_manifest_sha256")!=r["manifest_sha256"]
            or s.get("original_training_git_commit")!=r["source_git_revision"]
            or s.get("protocol_id")!=PROTOCOL_ID
            or s.get("geometry_review_sha256")!=geometry_sha256
            or s.get("geometry_status")!="unapproved"
            or s.get("environment_variation")!="none_fixed_TRAIN_road_initial_state"
            or s.get("sampling_mode")!="stochastic_ppo_policy_actions_torch_seeded_independently_per_episode"
            or s.get("episodes_per_train_track")!=EPISODES_PER_TRACK
            or s.get("confidence_level")!=0.95
            or s.get("predeclared_screen_lower_bound")!=SCREEN_LOWER_BOUND
            or s.get("scientific_evidence") is not False
            or s.get("heldout_rollouts")!=0
            or s.get("scientific_freeze_permitted") is not False
            or s.get("heldout_evaluation_permitted") is not False):
            raise ValueError("stochastic screen run identity or protocol drifted")
        tracks=s.get("train_tracks")
        if (not isinstance(tracks,list) or len(tracks)!=len(TRAIN_IDS)
            or [x.get("track_id") if isinstance(x,dict) else None for x in tracks]!=list(TRAIN_IDS)):
            raise ValueError("stochastic screen uses non-TRAIN or mismatched track identities")
        result=[]
        for track in tracks:
            episodes=track.get("episodes")
            if not isinstance(episodes,list) or len(episodes)!=EPISODES_PER_TRACK:
                raise ValueError("stochastic screen episode count drifted")
            for i,e in enumerate(episodes):
                if (not isinstance(e,dict) or e.get("index")!=i
                    or e.get("torch_action_seed")!=seed_for(run_id=r["run_id"],track_id=track["track_id"],index=i)
                    or type(e.get("success")) is not bool):
                    raise ValueError("stochastic screen episode identity/seed drifted")
            success=sum(int(e["success"]) for e in episodes)
            lower=one_sided_clopper_pearson_lower(success,EPISODES_PER_TRACK)
            expected_pass=lower>=SCREEN_LOWER_BOUND
            if (track.get("episode_count")!=EPISODES_PER_TRACK
                or track.get("success_count")!=success
                or abs(track.get("success_rate",-99)-success/EPISODES_PER_TRACK)>1e-8
                or abs(track.get("one_sided_95pct_cp_lower",-99)-round(lower,8))>1e-8
                or track.get("screen_lower_bound_ge_0p80") is not expected_pass):
                raise ValueError("stochastic screen success CI or predeclared rule drifted")
            result.append({
                "track_id":track["track_id"],
                "success_count":success,"trials":EPISODES_PER_TRACK,
                "one_sided_95pct_lower":round(lower,8),
                "screen_pass":expected_pass,
            })
        all_pass=all(x["screen_pass"] for x in result)
        if s.get("all_train_track_screens_pass") is not all_pass:
            raise ValueError("stochastic screen run-level outcome drifted")
        summaries.append({"run_id":r["run_id"],"tracks":result,"all_screen_pass":all_pass})
    if all(x["all_screen_pass"] for x in summaries):
        raise ValueError("registered screen expected to flag unresolved TRAIN sensitivity")
    return {
        "audit_type":"stochastic_train_screen",
        "runs":summaries,
        "environment_generalization_established":False,
    }


def build_freeze_readiness_report(
    *,repository_root:str|Path,calibration_run:str|Path,confirmation_run:str|Path,
    xz_audit_path:str|Path,swept_3d_audit_path:str|Path,
    road_union_audit_path:str|Path, stochastic_screen_path:str|Path,
)->dict[str,Any]:
    root=Path(repository_root).resolve(strict=True)
    _require_clean_git(root)
    preflight=build_pilot_preflight(
        repository_root=root,
        candidate_path=root/"configs/recovery/wp7-c2-candidate.yml",
        runtime_config_path=root/"configs/recovery/wp7-pilot-200k.yml",
    )
    profile=TrainingRunConfig.from_yaml(root/"configs/recovery/wp7-pilot-200k.yml")
    candidates=(Path(calibration_run).resolve(strict=True),Path(confirmation_run).resolve(strict=True))
    if candidates[0]==candidates[1]:
        raise ValueError("calibration and confirmation must be independent run directories")
    records=[
        _verify_single(root=root,run=run,seed=seed,role=role,
                      profile=profile,expected_fingerprints=preflight["fingerprints"])
        for run,(seed,other,role) in zip(candidates,SEED_PAIRS,strict=True)
        if seed==other
    ]
    if len(records)!=2 or records[0]["source_git_revision"]!=records[1]["source_git_revision"]:
        raise ValueError("registered independent seeds do not share original source revision")
    changed_since_training=_diff_since_training(root,records[0]["source_git_revision"])
    links=[]
    for kind,path,known in (
        ("trajectory_xz",xz_audit_path,KNOWN_AUDIT_SHA256["trajectory_xz"]),
        ("swept_3d_proxy",swept_3d_audit_path,KNOWN_AUDIT_SHA256["swept_3d_proxy"]),
    ):
        file=Path(path).resolve(strict=True)
        if not file.is_file() or _sha(file)!=known:
            raise ValueError(f"{kind} independent audit SHA256 differs from previously published evidence")
        item=validate_linked_audits(audit_type=kind,document=_json(file),records=records)
        links.append({"kind":kind,"sha256":known,**item})
    road_file=Path(road_union_audit_path).resolve(strict=True)
    if not road_file.is_file() or _sha(road_file)!=KNOWN_AUDIT_SHA256["right_turn_road_union"]:
        raise ValueError("right turn road-union audit SHA256 differs from published evidence")
    road_check=validate_right_turn_road_union(
        document=_json(road_file),records=records,
        xz_summaries=links[0]["runs"],
    )
    links.append({
        "kind":"right_turn_road_union",
        "sha256":KNOWN_AUDIT_SHA256["right_turn_road_union"],
        **road_check,
    })
    screen_file=Path(stochastic_screen_path).resolve(strict=True)
    if not screen_file.is_file() or _sha(screen_file)!=KNOWN_AUDIT_SHA256["stochastic_train_screen"]:
        raise ValueError("original stochastic TRAIN screening report SHA256 mismatch")
    screen_check=validate_stochastic_train_screen(
        document=_json(screen_file),records=records,
        geometry_sha256=_sha(root/"configs/recovery/wp7-vehicle-geometry-review.yml"),
    )
    links.append({
        "kind":"stochastic_train_screen",
        "sha256":KNOWN_AUDIT_SHA256["stochastic_train_screen"],
        **screen_check,
    })
    return {
        "schema_version":"wp7-recovery-freeze-readiness-v3",
        "status":"review_required",
        "scientific_freeze_permitted":False,
        "heldout_evaluation_permitted":False,
        "heldout_rollouts":0,
        "scientific_evidence":False,
        "evidence_integrity":"verified_at_inspection_time_not_independent_external_attestation",
        "original_training_source_git_revision":records[0]["source_git_revision"],
        "committed_audit_only_changes_since_training":list(changed_since_training),
        "source_diff_ruling":"no_committed_physics_reconstruction_or_training_module_diff_since_pilot",
        "retrospective_not_training_time":True,
        "registered_runs":records,
        "linked_audits":links,
        "blocking_decisions":list(BLOCKERS),
        "next_decision":"specify_real_vehicle_geometry_and_corner_cutting_corridor; approve_future_prospective_provenance_and_robustness_protocol",
    }
