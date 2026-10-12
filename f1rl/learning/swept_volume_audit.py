"""Conservative 3D swept body-envelope audit for pre-existing WP7 TRAIN models.

No simulator, reward, model, source track, or HELD-OUT state is modified.
The body profiles are sensitivity assumptions, not measured vehicle geometry.
The envelope is conservative for linear position and shortest-arc quaternion
interpolation between discrete simulator states, not a continuous physics proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import acos, cos, hypot, isfinite, sqrt
from pathlib import Path

from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.geometry.vector import Vec3
from ..sandbox.track.volume import OrientedBoxVolume
from .algorithms.ppo import PPOModel
from .pilot_milestones import verify_pilot_evidence_manifest
from .pilot_preflight import TRAIN_IDS
from .reconstruction import RecoveryCandidate, build_training_scenarios


_AXES = (Vec3(1.,0.,0.), Vec3(0.,1.,0.), Vec3(0.,0.,1.))


def segment_to_oriented_box_distance_3d(
    first: Vec3, second: Vec3, box: OrientedBoxVolume
) -> float:
    """Exact Euclidean distance from segment to any rotated rectangular prism.

    The squared point-to-box distance along the segment is piecewise quadratic.
    Split at all +/-half-extent slab crossings, then minimize each quadratic.
    """
    if not isinstance(first, Vec3) or not isinstance(second, Vec3):
        raise ValueError("segment endpoints must be Vec3")
    if not isinstance(box, OrientedBoxVolume):
        raise ValueError("obstacle volume must be OrientedBoxVolume")
    axes = tuple(box.orientation.rotate(unit) for unit in _AXES)
    a = tuple((first - box.center).dot(axis) for axis in axes)
    b = tuple((second - box.center).dot(axis) for axis in axes)
    half = (box.half_extents.x,box.half_extents.y,box.half_extents.z)
    breaks = [0.,1.]
    for x,y,h in zip(a,b,half,strict=True):
        delta=y-x
        if abs(delta)>1e-14:
            for boundary in (-h,h):
                t=(boundary-x)/delta
                if 0.<t<1.:
                    breaks.append(t)
    breaks=sorted(set(breaks))

    def dist2(t:float)->float:
        return sum(
            max(abs(x+(y-x)*t)-h,0.)**2
            for x,y,h in zip(a,b,half,strict=True)
        )

    best=min(dist2(t) for t in breaks)
    for lo,hi in zip(breaks,breaks[1:]):
        mid=(lo+hi)/2.
        quadratic=0.
        linear=0.
        for x,y,h in zip(a,b,half,strict=True):
            v=x+(y-x)*mid
            if v < -h:
                intercept=x+h
            elif v > h:
                intercept=x-h
            else:
                continue
            slope=y-x
            quadratic+=slope*slope
            linear+=intercept*slope
        if quadratic>1e-24:
            optimal=max(lo,min(hi,-linear/quadratic))
            best=min(best,dist2(optimal))
    return sqrt(max(best,0.))


@dataclass(frozen=True,slots=True)
class BodyProxy:
    """Assumed rigid box centered offset_y above vehicle position, LOCAL axes."""
    name: str
    length_m: float
    width_m: float
    height_m: float
    center_y_offset_m: float
    longitudinal_stations: int=9

    def __post_init__(self)->None:
        if not isinstance(self.name,str) or not self.name.strip():
            raise ValueError("proxy name must be nonempty")
        if not all(isfinite(x) and x>0. for x in (self.length_m,self.width_m,self.height_m)):
            raise ValueError("proxy dimensions must be finite and positive")
        if not isfinite(self.center_y_offset_m):
            raise ValueError("proxy center_y_offset_m must be finite")
        if isinstance(self.longitudinal_stations,bool) or not isinstance(self.longitudinal_stations,int) or self.longitudinal_stations<2:
            raise ValueError("proxy must use at least two longitudinal stations")

    @property
    def longitudinal_offsets(self)->tuple[float,...]:
        return tuple(
            -self.length_m/2.+self.length_m*i/(self.longitudinal_stations-1)
            for i in range(self.longitudinal_stations)
        )

    @property
    def cross_section_radius_m(self)->float:
        # The nearest station is at most half an interval from a body point.
        interval=self.length_m/(self.longitudinal_stations-1)
        return sqrt((interval/2.)**2+(self.width_m/2.)**2+(self.height_m/2.)**2)


PROXY_PROFILES=(
    BodyProxy("sensitivity_small_3p2x1p4x1p0",3.2,1.4,1.0,0.45),
    BodyProxy("sensitivity_medium_4p0x1p8x1p2",4.0,1.8,1.2,0.55),
    BodyProxy("sensitivity_large_5p0x2p2x1p4",5.0,2.2,1.4,0.65),
)


def _rotation_angle(first:Quaternion,second:Quaternion)->float:
    dot=abs(first.w*second.w+first.x*second.x+first.y*second.y+first.z*second.z)
    return 2.*acos(min(1.,max(0.,dot)))


def swept_body_proxy_clearance(
    first_position:Vec3,first_rotation:Quaternion,
    second_position:Vec3,second_rotation:Quaternion,
    box:OrientedBoxVolume,proxy:BodyProxy,
)->float:
    """Signed conservative clearance lower bound for the *assumed* vehicle box.

    All box points are inside spheres about longitudinal stations, radius
    sqrt(width^2/4+height^2/4+(station spacing)^2/4).
    For linear position / shortest-arc rigid-body rotation interpolation, the
    station's curved motion deviates from its endpoint chord by <=
    |local station vector|*(1-cos(total_angle/2)); subtract that bound.
    Positive result proves separation under these explicitly stated
    assumptions. Zero/negative is *inconclusive*, not proof of collision.
    """
    if not isinstance(proxy,BodyProxy):
        raise ValueError("proxy must be BodyProxy")
    angle=_rotation_angle(first_rotation,second_rotation)
    radius=proxy.cross_section_radius_m
    gaps=[]
    for station in proxy.longitudinal_offsets:
        local=Vec3(0.,proxy.center_y_offset_m,station)
        first=first_position+first_rotation.rotate(local)
        second=second_position+second_rotation.rotate(local)
        rotation_chord_error=local.magnitude*(1.-cos(angle/2.))
        gaps.append(
            segment_to_oriented_box_distance_3d(first,second,box)
            -radius-rotation_chord_error
        )
    return min(gaps)


def _sha(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def audit_3d_swept_run(
    *, repository_root: str | Path, run_directory: str | Path,
    profiles: tuple[BodyProxy,...]=PROXY_PROFILES,
)->dict[str,object]:
    """Replay only previously trained final PPO on registered four TRAIN assets."""
    root=Path(repository_root).resolve(strict=True)
    run=Path(run_directory).resolve(strict=True)
    if not profiles or len(set(p.name for p in profiles))!=len(profiles) or not all(isinstance(p,BodyProxy) for p in profiles):
        raise ValueError("profiles must be nonempty, uniquely named BodyProxy instances")

    evidence=verify_pilot_evidence_manifest(run)
    if evidence["status"]!="verified_exploratory_evidence_not_scientific":
        raise ValueError("only previously verified real exploratory PPO runs are eligible")
    report=json.loads((run/"evaluations"/"milestone-200000.json").read_text("utf-8"))
    manifest=json.loads((run/"evaluations"/"manifest.json").read_text("utf-8"))
    if (report["evaluation"]["track_ids"]!=list(TRAIN_IDS)
        or report["evaluation"]["policy_source"]!="ppo_archive_requires_run_provenance_review"
        or report["evaluation"]["episode_count_per_track"]!=1):
        raise ValueError("archived TRAIN evaluation contract is invalid")
    for source in manifest["source_fingerprints"]:
        path=(root/source["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or _sha(path)!=source["sha256"]:
            raise ValueError("model training source fingerprint drift: "+source["path"])
    model=(run/report["model_path"]).resolve(strict=True)
    if not model.is_relative_to(run) or _sha(model)!=report["checkpoint_sha256"]:
        raise ValueError("model archive SHA mismatch")
    policy=PPOModel.load(model,device="cpu")
    scenarios=build_training_scenarios(
        RecoveryCandidate.from_yaml(root/"configs/recovery/wp7-c2-candidate.yml")
    )
    if tuple(s.track_id for s in scenarios.scenarios)!=TRAIN_IDS:
        raise ValueError("training split boundary changed")

    rows=[]
    for scenario,expected in zip(scenarios.scenarios,report["evaluation"]["track_summaries"],strict=True):
        if expected["track_id"]!=scenario.track_id or expected["episode_count"]!=1:
            raise ValueError("archived track summary mismatch")
        factory=scenario.factory
        env=factory.create()
        try:
            obs,info=env.reset()
            state=env._simulation.snapshot.world.vehicle  # audit read-only of immutable snapshot
            if tuple((state.position.x,state.position.y,state.position.z))!=info["telemetry"]["vehicle_position"]:
                raise ValueError("simulator reset pose disagrees with public telemetry")
            minimum_center={ob.id:float("inf") for ob in factory.track.obstacles}
            minimum_by_proxy={
                profile.name:{ob.id:float("inf") for ob in factory.track.obstacles}
                for profile in profiles
            }
            count=0
            engine_hits=0
            terminated=truncated=False
            while not (terminated or truncated):
                action=policy.predict(obs,deterministic=True)
                obs,_,terminated,truncated,info=env.step(action)
                later=env._simulation.snapshot.world.vehicle
                if tuple((later.position.x,later.position.y,later.position.z))!=info["telemetry"]["vehicle_position"]:
                    raise ValueError("simulator pose disagrees with public telemetry")
                count+=1
                engine_hits+=len(info["telemetry"]["obstacle_hit_ids"])
                for obstacle in factory.track.obstacles:
                    oid=obstacle.id
                    box=obstacle.volume
                    minimum_center[oid]=min(
                        minimum_center[oid],
                        segment_to_oriented_box_distance_3d(state.position,later.position,box)
                    )
                    for proxy in profiles:
                        minimum_by_proxy[proxy.name][oid]=min(
                            minimum_by_proxy[proxy.name][oid],
                            swept_body_proxy_clearance(
                                state.position,state.orientation,
                                later.position,later.orientation,
                                box,proxy,
                            )
                        )
                state=later
            success=bool(terminated and info["finished"] and not truncated)
            if success != (expected["success_count"]==1) or engine_hits!=int(expected["mean_collision_count"]):
                raise ValueError("deterministic 3D replay disagrees with archived success/collision")
            if success and abs(count*factory.simulation_config.fixed_dt-expected["mean_completion_time_for_successes"])>1e-6:
                raise ValueError("deterministic 3D replay disagrees with archived completion time")
            rows.append({
                "track_id":scenario.track_id,
                "success":success,
                "steps":count,
                "simulator_obstacle_hit_count":engine_hits,
                "obstacle_count":len(factory.track.obstacles),
                "min_exact_swept_centerline_to_box_3d_m": (
                    None if not minimum_center else round(min(minimum_center.values()),6)
                ),
                "profiles":[{
                    "name":proxy.name,
                    "assumed_length_m":proxy.length_m,
                    "assumed_width_m":proxy.width_m,
                    "assumed_height_m":proxy.height_m,
                    "assumed_center_y_offset_m":proxy.center_y_offset_m,
                    "station_count":proxy.longitudinal_stations,
                    "min_conservative_gap_m":(
                        None if not minimum_by_proxy[proxy.name]
                        else round(min(minimum_by_proxy[proxy.name].values()),6)
                    ),
                    "conservative_clearance_pass":(
                        None if not minimum_by_proxy[proxy.name]
                        else min(minimum_by_proxy[proxy.name].values())>1e-7
                    ),
                } for proxy in profiles],
            })
        finally:
            env.close()
    return {
        "schema_version":"wp7-recovery-swept-3d-proxy-audit-v1",
        "scientific_evidence":False,
        "status":"proxy_sensitivity_only_requires_real_vehicle_geometry",
        "heldout_rollouts":0,
        "run_id":run.name,
        "source_git_revision":report["git_commit"],
        "final_model_sha256":report["checkpoint_sha256"],
        "manifest_sha256":_sha(run/"evaluations"/"manifest.json"),
        "method":"exact_3d_segment_to_oriented_box; conservative_longitudinal_sphere_cover_plus_quaternion_rotation_sagitta",
        "assumed_interpolation":"linear_centroid_and_shortest_arc_slerp_between_discrete_simulator_poses",
        "limitations":[
            "No physical vehicle length/width/height or local body centre was established in the original simulator.",
            "Proxy-positive means guaranteed separation only for the assumed shape and interpolation; proxy-negative is inconclusive.",
            "Discrete simulation can differ from linear-position/slerp interpolation between timesteps.",
            "No new episode randomness is introduced; deterministic TRAIN replay cannot prove robustness.",
            "No HELD-OUT assets or models were accessed.",
        ],
        "track_results":rows,
    }
