"""Independent TRAIN-only continuous road-centerline / route-corridor audit.

Right Turn surfaces are two horizontal oriented rectangles. Clip each finite
vehicle-center segment against their UNION, not only simulator step endpoints.
No body footprint/width or approved route-corner-cutting threshold is assumed.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import hypot, isfinite
from pathlib import Path

from ..sandbox.geometry.vector import Vec3
from ..sandbox.track.patch import PlanarSurfacePatch
from .algorithms.ppo import PPOModel
from .pilot_milestones import verify_pilot_evidence_manifest
from .pilot_preflight import TRAIN_IDS
from .reconstruction import RecoveryCandidate, build_training_scenarios
from .route_navigation import AuthoredRoute


def _rectangle_frame(patch: PlanarSurfacePatch) -> tuple[Vec3, Vec3]:
    """Only horizontal patch footprints: do not silently flatten a ramp."""
    if not isinstance(patch, PlanarSurfacePatch):
        raise ValueError("expected PlanarSurfacePatch")
    if abs(patch.normal.x)>1e-9 or abs(patch.normal.y-1.)>1e-9 or abs(patch.normal.z)>1e-9:
        raise ValueError("road corridor audit requires horizontal road patches")
    if abs(patch.forward.y)>1e-9:
        raise ValueError("road corridor audit requires horizontal forward axis")
    return patch.right,patch.forward


def clip_xz_segment_to_patch(
    first:Vec3,second:Vec3,patch:PlanarSurfacePatch,
) -> tuple[float,float] | None:
    """Closed t interval [0,1] inside the patch's finite XZ rectangle."""
    right,forward=_rectangle_frame(patch)
    offset_a=first-patch.origin
    delta=second-first
    low,high=0.,1.
    for axis,half in ((right,patch.width/2.),(forward,patch.length/2.)):
        start=offset_a.dot(axis)
        direction=delta.dot(axis)
        if abs(direction)<1e-14:
            if abs(start)>half+1e-10:
                return None
            continue
        enter=(-half-start)/direction
        leave=(half-start)/direction
        low=max(low,min(enter,leave))
        high=min(high,max(enter,leave))
        if low>high+1e-12:
            return None
    if low>high:
        return None
    return max(0.,low),min(1.,high)


def continuous_union_road_coverage(
    first:Vec3,second:Vec3,patches:tuple[PlanarSurfacePatch,...],
)->tuple[float,float]:
    """Return covered XZ length and uncovered XZ length (metres).

    Empty patches are invalid. Point samples can use a zero-length segment.
    Boundary tangencies have zero covered length and are not false road passes.
    """
    if not isinstance(first,Vec3) or not isinstance(second,Vec3):
        raise ValueError("segment endpoints must be Vec3")
    if not isinstance(patches,tuple) or not patches or not all(isinstance(x,PlanarSurfacePatch) for x in patches):
        raise ValueError("road patches must be nonempty tuple of PlanarSurfacePatch")
    length=hypot(second.x-first.x,second.z-first.z)
    intervals=sorted(
        interval for patch in patches
        if (interval:=clip_xz_segment_to_patch(first,second,patch)) is not None
    )
    if length<1e-12:
        return 0.,0. if intervals else 0.  # stationary vehicle has zero travelled distance
    coverage=0.
    merged=[]
    for lo,hi in intervals:
        if merged and lo<=merged[-1][1]+1e-12:
            merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
        else:
            merged.append((lo,hi))
    coverage=sum(max(0.,hi-lo) for lo,hi in merged)
    coverage=max(0.,min(1.,coverage))
    return length*coverage,length*(1.-coverage)


def _sha(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def audit_right_turn_corridor(
    *, repository_root:str|Path, run_directory:str|Path,
)->dict[str,object]:
    """Replay archived final PPO on Right Turn TRAIN only; no held-out resolution."""
    root=Path(repository_root).resolve(strict=True)
    run=Path(run_directory).resolve(strict=True)
    check=verify_pilot_evidence_manifest(run)
    if check["status"]!="verified_exploratory_evidence_not_scientific":
        raise ValueError("requires archived real exploratory PPO evidence")
    report=json.loads((run/"evaluations"/"milestone-200000.json").read_text("utf-8"))
    manifest=json.loads((run/"evaluations"/"manifest.json").read_text("utf-8"))
    eval_=report["evaluation"]
    if eval_["track_ids"]!=list(TRAIN_IDS) or eval_["policy_source"]!="ppo_archive_requires_run_provenance_review":
        raise ValueError("requires original four TRAIN track source")
    for entry in manifest["source_fingerprints"]:
        source=(root/entry["path"]).resolve(strict=True)
        if not source.is_relative_to(root) or _sha(source)!=entry["sha256"]:
            raise ValueError("training source fingerprint differs")
    model=(run/report["model_path"]).resolve(strict=True)
    if not model.is_relative_to(run) or _sha(model)!=report["checkpoint_sha256"]:
        raise ValueError("archived model hash mismatch")
    scenarios=build_training_scenarios(RecoveryCandidate.from_yaml(root/"configs/recovery/wp7-c2-candidate.yml"))
    if tuple(s.track_id for s in scenarios.scenarios)!=TRAIN_IDS:
        raise ValueError("TRAIN source boundary invalid")
    right_turn=scenarios.scenarios[2]
    if right_turn.track_id!="right-turn-v1":
        raise ValueError("right-turn TRAIN ordering changed")
    factory=right_turn.factory
    patches=factory.track.surfaces
    if len(patches)!=2:
        raise ValueError("right-turn patch count changed")
    for patch in patches:
        _rectangle_frame(patch)
    route=AuthoredRoute(factory.navigation)
    expected=eval_["track_summaries"][2]
    if expected["track_id"]!="right-turn-v1" or expected["episode_count"]!=1:
        raise ValueError("expected single archived right turn TRAIN episode")

    policy=PPOModel.load(model,device="cpu")
    env=factory.create()
    try:
        obs,info=env.reset()
        point=Vec3(*info["telemetry"]["vehicle_position"])
        max_sampled=route.horizontal_distance(point)
        upper=max_sampled
        outside=0.
        traversed=0.
        samples_outside=0
        steps=0
        terminated=truncated=False
        while not (terminated or truncated):
            action=policy.predict(obs,deterministic=True)
            obs,_,terminated,truncated,info=env.step(action)
            next_point=Vec3(*info["telemetry"]["vehicle_position"])
            covered,uncovered=continuous_union_road_coverage(point,next_point,patches)
            outside+=uncovered
            traversed+=covered+uncovered
            a=route.horizontal_distance(point)
            b=route.horizontal_distance(next_point)
            max_sampled=max(max_sampled,a,b)
            # horizontal_distance to a fixed polyline is 1-Lipschitz,
            # and every point on a segment is within half a segment
            # length of one endpoint. This is an upper bound, not exact maximum.
            upper=max(upper,max(a,b)+hypot(next_point.x-point.x,next_point.z-point.z)/2.)
            if not any(clip_xz_segment_to_patch(next_point,next_point,p) is not None for p in patches):
                samples_outside+=1
            steps+=1
            point=next_point
        passed=bool(info["finished"] and terminated and not truncated)
        if passed!=(expected["success_count"]==1):
            raise ValueError("Right Turn outcome differs from original final checkpoint")
        if passed and abs(steps*factory.simulation_config.fixed_dt-expected["mean_completion_time_for_successes"])>1e-6:
            raise ValueError("Right Turn replay time mismatch")
    finally:
        env.close()

    return {
        "schema_version":"wp7-recovery-right-turn-road-union-audit-v1",
        "status":"diagnostic_requires_route_corridor_policy",
        "scientific_evidence":False,
        "heldout_rollouts":0,
        "run_id":run.name,
        "source_git_revision":report["git_commit"],
        "source_manifest_sha256":_sha(run/"evaluations"/"manifest.json"),
        "final_model_sha256":report["checkpoint_sha256"],
        "track_id":"right-turn-v1",
        "success":passed,
        "simulator_steps":steps,
        "road_patch_count":len(patches),
        "road_widths_m":[patch.width for patch in patches],
        "centerline_total_path_xz_m":round(traversed,6),
        "centerline_uncovered_road_path_xz_m":round(outside,6),
        "centerline_outside_road_sample_count":samples_outside,
        "max_sampled_route_deviation_m":round(max_sampled,6),
        "upper_bound_route_deviation_between_samples_m":round(upper,6),
        "engine_offtrack_limit_m":6.0,
        "is_physical_body_clearance":False,
        "is_approved_corner_cutting_threshold":False,
        "limitations":[
            "Road union coverage applies only to vehicle CENTRE on horizontal patches, not body footprint.",
            "Route deviation is measured against authored path, not permitted corner-cutting policy.",
            "Upper bound uses 1-Lipschitz property; may conservatively overestimate true continuous deviation.",
            "Road exit beyond the union does not automatically mean an obstacle impact.",
            "Replay is deterministic, same TRAIN asset as original experiment; not independent random robustness.",
        ],
    }
