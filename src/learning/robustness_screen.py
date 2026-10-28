"""Registered TRAIN-only stochastic-policy sensitivity screen for saved WP7 PPO.

Not environmental generalization: the four TRAIN environments have fixed
geometry, initial state and reset. No HELD-OUT is instantiated or inspected.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from math import comb, isfinite
from pathlib import Path
from typing import Callable, ContextManager
import json

from .algorithms.ppo import PPOModel
from .evaluation import evaluate_episode
from .geometry_review import load_geometry_review
from .pilot_milestones import verify_pilot_evidence_manifest
from .pilot_preflight import TRAIN_IDS
from .reconstruction import RecoveryCandidate, build_training_scenarios


PROTOCOL_ID="wp7-train-stochastic-action-sensitivity-v1"
EPISODES_PER_TRACK=16
LOWER_CONFIDENCE=0.95
SCREEN_LOWER_BOUND=0.80


def seed_for(*,run_id:str,track_id:str,index:int)->int:
    if not run_id or track_id not in TRAIN_IDS or isinstance(index,bool) or not isinstance(index,int) or index<0:
        raise ValueError("invalid registered TRAIN seed identity")
    raw=f"{PROTOCOL_ID}|{run_id}|{track_id}|{index}".encode("utf-8")
    return int.from_bytes(sha256(raw).digest()[:4],"big")


def _binomial_tail(n:int,k:int,p:float)->float:
    return sum(comb(n,j)*p**j*(1.-p)**(n-j) for j in range(k,n+1))


def one_sided_clopper_pearson_lower(successes:int,trials:int,confidence:float=LOWER_CONFIDENCE)->float:
    if (any(type(x) is not int for x in (successes,trials))
        or not 0<=successes<=trials or trials<=0):
        raise ValueError("binomial observations must be valid integer counts")
    if isinstance(confidence,bool) or not isinstance(confidence,(float,int)) or not isfinite(confidence) or not 0<confidence<1:
        raise ValueError("confidence must be in (0,1)")
    if successes==0:
        return 0.
    alpha=1.-confidence
    lo,hi=0.,1.
    for _ in range(70):
        mid=(lo+hi)/2.
        if _binomial_tail(trials,successes,mid)<alpha:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2.


class _StochasticPolicy:
    def __init__(self,source:PPOModel):
        self.source=source

    def predict(self,observation,*,deterministic:bool=True):
        # Evaluate_episode requests deterministic=True. This named wrapper
        # explicitly overrides it for the separately labelled stochastic screen.
        if deterministic is not True:
            raise ValueError("unexpected evaluator predict contract")
        return self.source.predict(observation,deterministic=False)


def _hash(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def screen_saved_train_policy(*,repository_root:str|Path,run_directory:str|Path, rng_scope:Callable[[int],ContextManager[None]])->dict[str,object]:
    if not callable(rng_scope):
        raise ValueError('rng_scope must be a callable context manager factory')
    root=Path(repository_root).resolve(strict=True)
    run=Path(run_directory).resolve(strict=True)
    geometry=load_geometry_review(root/"configs/recovery/wp7-vehicle-geometry-review.yml")
    assert not geometry.heldout_evaluation_authorized
    verified=verify_pilot_evidence_manifest(run)
    if verified["status"]!="verified_exploratory_evidence_not_scientific" or verified["heldout_rollouts"]!=0:
        raise ValueError("requires verified original exploratory TRAIN model")
    report=json.loads((run/"evaluations/milestone-200000.json").read_text("utf-8"))
    manifest=json.loads((run/"evaluations/manifest.json").read_text("utf-8"))
    if (report["evaluation"]["track_ids"]!=list(TRAIN_IDS)
        or report["evaluation"]["policy_source"]!="ppo_archive_requires_run_provenance_review"
        or report["evaluation"]["episode_count_per_track"]!=1):
        raise ValueError("original four-TRAIN evidence contract drifted")
    for item in manifest["source_fingerprints"]:
        src=(root/item["path"]).resolve(strict=True)
        if not src.is_relative_to(root) or _hash(src)!=item["sha256"]:
            raise ValueError("original training source fingerprint drift")
    archive=(run/report["model_path"]).resolve(strict=True)
    if not archive.is_relative_to(run) or _hash(archive)!=report["checkpoint_sha256"]:
        raise ValueError("original saved PPO archive SHA256 mismatch")
    scenarios=build_training_scenarios(
        RecoveryCandidate.from_yaml(root/"configs/recovery/wp7-c2-candidate.yml")
    )
    if tuple(item.track_id for item in scenarios.scenarios)!=TRAIN_IDS:
        raise ValueError("TRAIN boundary changed")
    policy=_StochasticPolicy(PPOModel.load(archive,device="cpu"))
    tracks=[]
    for scenario in scenarios.scenarios:
        episodes=[]
        for index in range(EPISODES_PER_TRACK):
            seed=seed_for(run_id=run.name,track_id=scenario.track_id,index=index)
            # The experimental CLI owns Torch RNG isolation; production
            # modules contain no direct torch dependency.
            with rng_scope(seed):
                metric=evaluate_episode(
                    track_id=scenario.track_id,
                    factory=scenario.factory,
                    policy=policy,
                )
            episodes.append({
                "index":index,"torch_action_seed":seed,
                "success":metric.success,"truncated":metric.truncated,
                "steps":metric.steps,"collisions":metric.collision_count,
                "final_progress":round(metric.final_normalized_route_progress,8),
                "completion_time":metric.completion_time,
                "reward":round(metric.episode_return,8),
            })
        successes=sum(int(x["success"]) for x in episodes)
        lower=one_sided_clopper_pearson_lower(successes,len(episodes))
        tracks.append({
            "track_id":scenario.track_id,
            "success_count":successes,
            "episode_count":len(episodes),
            "success_rate":successes/len(episodes),
            "one_sided_95pct_cp_lower":round(lower,8),
            "screen_lower_bound_ge_0p80":lower>=SCREEN_LOWER_BOUND,
            "mean_final_progress":round(sum(x["final_progress"] for x in episodes)/len(episodes),8),
            "total_engine_collision_events":sum(x["collisions"] for x in episodes),
            "episodes":episodes,
        })
    return {
        "schema_version":"wp7-recovery-train-stochastic-screen-v1",
        "protocol_id":PROTOCOL_ID,
        "status":"exploratory_action_sampling_not_environment_generalization",
        "scientific_evidence":False,
        "heldout_rollouts":0,
        "run_id":run.name,
        "model_sha256":report["checkpoint_sha256"],
        "original_manifest_sha256":_hash(run/"evaluations/manifest.json"),
        "original_training_git_commit":report["git_commit"],
        "geometry_review_sha256":geometry.source_sha256,
        "geometry_status":geometry.status,
        "episodes_per_train_track":EPISODES_PER_TRACK,
        "confidence_level":LOWER_CONFIDENCE,
        "predeclared_screen_lower_bound":SCREEN_LOWER_BOUND,
        "sampling_mode":"stochastic_ppo_policy_actions_torch_seeded_independently_per_episode",
        "environment_variation":"none_fixed_TRAIN_road_initial_state",
        "scientific_freeze_permitted":False,
        "heldout_evaluation_permitted":False,
        "train_tracks":tracks,
        "all_train_track_screens_pass":all(x["screen_lower_bound_ge_0p80"] for x in tracks),
    }
