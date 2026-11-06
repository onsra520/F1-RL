"""Real independent Physics v4 control-tape execution for TRAIN Right/Left v2.

ENGINEERING ONLY: no PPO, no HELD-OUT geometry, no scientific signoff.
Uses source-registered owner-v3 physics and race setup with exact v4 overlay,
rebinding the independently-authored v2 TRAIN geometry without modifying
any frozen source/config/model. Explicitly records its nonregistered factory
rebinding instead of claiming full TRAIN-5 scientific protocol acceptance.
"""
from __future__ import annotations
import argparse, json, sys, traceback, math
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

sys.path.insert(0,str(ROOT))

from f1rl.research.wp7_protocol import load_registration,prepare_scenarios
from f1rl.research.wp3_eval_only_coordinator import (
    FULL06_RUNTIME_PROFILE, apply_runtime_profile,validate_runtime_profile
)
from f1rl.learning.reconstruction import RecoveryCandidate,build_training_scenarios
from f1rl.learning.episode import EpisodeConfig
from f1rl.research.wp7_arc_trajectory import ArcPath,_pieces
from f1rl.research.wp7_executed_feasibility import _arc_control,ARC_CONTROLLER
from f1rl.sandbox.ballistic_flight import FlightState
from f1rl.sandbox.independent_suspension_dynamics import SuspensionDynamicsState
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.learning.crash_terminal_registered import validate_registered_crash_terminal_info

SOURCES=[
 "assets/tracks/training/right-turn-train-v2.json",
 "assets/tracks/training/left-turn-train-v2.json",
 "assets/tracks/manifest-five-track-paired-v2.json",
 "configs/experiments/ppo-generalization-five-track-paired-v2.yml",
 "configs/recovery/wp7-c2-five-track-paired-v2.yml",
 "f1rl/research/wp7_protocol.py",
 "f1rl/research/wp3_eval_only_coordinator.py",
 "f1rl/research/wp7_executed_feasibility.py",
]
def digest_bytes(data):return sha256(data).hexdigest()
def digest(obj):
    return digest_bytes(json.dumps(obj,sort_keys=True,allow_nan=False,
         separators=(',',':')).encode('utf-8'))
def source_hashes():
    return {n:digest_bytes((ROOT/n).read_bytes()) for n in SOURCES}
def build_factory(name):
    # The separately authored 5-TRAIN engineering experiment is parsed and
    # each world source hash verified before any simulator is created.
    config=RecoveryCandidate.from_yaml(
       ROOT/"configs/recovery/wp7-c2-five-track-paired-v2.yml")
    assert tuple(config.experiment.tracks.training_ids)==(
      "straight-sprint-v1","slalom-v1","right-turn-train-v2",
      "ramp-v1","left-turn-train-v2")
    raw=next(s.factory for s in build_training_scenarios(config).scenarios
             if s.track_id==name)
    # Current registered owner protocol is four-track-only. This demo composes
    # its unchanged registered v3 physics/reward/action settings with the new
    # TRAIN v2 world and spawn, NOT a registered 5-track training factory.
    owner_reg=load_registration(ROOT,profile="spatial_rotating_owner_v3")
    old=next(s.factory for s in prepare_scenarios(
        ROOT,owner_reg,seed_index=0).scenarios
             if s.track_id=="right-turn-v1")
    height=old.vehicle_parameters.ride_height
    pos=raw.initial_state.vehicle.position+Vec3(0.,height,0.)
    initial=replace(raw.initial_state,
        vehicle=replace(raw.initial_state.vehicle,position=pos),
        suspension=SuspensionDynamicsState(pos.y),
        flight=FlightState("grounded"))
    factory=replace(old,track=raw.track,
        navigation=raw.navigation,initial_state=initial,
        reset_config=None,reset_track_id=None,
        episode_config=EpisodeConfig(1100))
    factory,provenance=apply_runtime_profile(factory,FULL06_RUNTIME_PROFILE)
    assert provenance==validate_runtime_profile(factory,FULL06_RUNTIME_PROFILE)
    assert factory.requires_scientific_admission
    return factory,provenance

def control_path(name):
    sign=1 if name=="right-turn-train-v2" else -1
    path=ArcPath((-3.*sign,5.),0.,
       ((60.,0.),(5.*math.pi,sign*.1),(73.,0.)))
    pieces=_pieces(path)
    # Check exact symbolic route endpoints and navigation authoring
    final=pieces[-1]
    from f1rl.research.wp7_arc_trajectory import _advance
    x,z,_=_advance(final[1],final[2],final[3],final[4],final[5])
    assert abs(x-sign*80)<1e-7 and abs(z-75)<1e-7
    return pieces,60+5*math.pi+73

def _stamp(info,action,tick):
    telemetry=info["telemetry"]
    race=info["race"]
    row={
      "tick":tick,
      "action":[float(a) for a in action],
      "position":list(telemetry["vehicle_position"]),
      "velocity":list(telemetry["vehicle_velocity"]),
      "speed":telemetry["vehicle_speed"],
      "grounded":telemetry["grounded"],
      "checkpoint_completed":info.get("completed_checkpoint_id"),
      "next_checkpoint_index":info.get("next_checkpoint_index"),
      "checkpoint_events":list(race.get("completed_checkpoint_ids",())),
      "directed_crossings":list(race.get("directed_crossings",())),
      "crossing_times_s":list(race.get("tick_local_crossing_times_s",())),
      "finished":info["finished"],
      "finished_now":info["finished_now"],
      "terminal_reason":race.get("terminal_reason"),
      "terminal_failure":race.get("terminal_failure"),
      "obstacle_hit_ids":list(telemetry["obstacle_hit_ids"]),
    }
    return row
def run_episode(factory, pieces, finish_s, mode, tape=None,limit=1100):
    env=factory.create()
    rows=[]
    info=None
    ended=False
    exception=None
    controls=[]
    outcome={}
    try:
        observation,info=env.reset(seed=20261106)
        assert tuple(observation.shape)==(16,)
        assert info["physics"]["configuration"]["ground_motion_version"]==(
          "spatial_road_edge_partial_lifecycle_v4")
        assert info["race"]["configuration"]["version"]==(
          "wp7_directed_lifecycle_crash_contract_v4")
        assert info["race"]["physical_crash_contract"]["schema_version"]==(
          "wp3_physical_crash_terminal_v1")
        for tick in range(1,limit+1):
            if tape is None:
                action_model=_arc_control(
                    env._simulation.snapshot,factory,pieces,finish_s)
                action=np.asarray([
                    action_model.steering,action_model.throttle,
                    action_model.brake],dtype=np.float32)
            else:
                if tick>len(tape):break
                action=np.asarray(tape[tick-1],dtype=np.float32)
            controls.append([float(x) for x in action])
            observation,reward,terminated,truncated,info=env.step(action)
            assert env.observation_space.contains(observation)
            rec=_stamp(info,action,tick)
            rec.update(reward=float(reward),terminated=terminated,truncated=truncated)
            rows.append(rec)
            if terminated or truncated:
                ended=True
                if info.get("physical_crash_terminal") is not None:
                    outcome["physical_crash_terminal"]=info["physical_crash_terminal"]
                    outcome["certified_crash_info_valid"]=bool(
                        validate_registered_crash_terminal_info(info))
                if info.get("roll_envelope_terminal") is not None:
                    outcome["roll_envelope_terminal"]=info["roll_envelope_terminal"]
                outcome["research_outcome"]=info.get("research_outcome")
                break
    except Exception as exc:
        exception=type(exc).__name__+": "+str(exc)
        outcome["traceback"]=traceback.format_exc(limit=7)
        outcome["last_research_outcome"]=env.last_research_outcome
    finally:
        env.close()
    outcome.update(
       mode=mode,episode_steps=len(rows),action_count=len(controls),
       ended=ended,
       finished=bool(info.get("finished",False)) if info else False,
       checkpoints_completed=[
           r["checkpoint_completed"] for r in rows
           if r["checkpoint_completed"] is not None],
       finish_tick=next((r["tick"] for r in rows if r["finished_now"]),None),
       termination=(rows[-1]["terminal_reason"] if rows else None),
       exception=exception,
       tick_rows_sha256=digest(rows),
       controller_controls_sha256=digest(controls),
       last_position=rows[-1]["position"] if rows else None,
       last_speed_mps=rows[-1]["speed"] if rows else None,
       rows=rows,
       controls=controls,
    )
    return outcome
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-root",required=True,help="External evidence directory (not inside checkout)")
    parser.add_argument("--track",choices=["right-turn-train-v2","left-turn-train-v2"],required=True)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--limit",type=int,default=1100)
    args=parser.parse_args()
    assert args.run_id.replace("-","").replace("_","").isalnum()
    assert 1<=args.limit<=2500
    target=Path(args.out_root).resolve()/"STEP3-INDEPENDENT-V4-CONTROLS"/args.run_id
    if target.is_relative_to(ROOT):
        raise ValueError("Engineering evidence must be outside the repository")
    target.mkdir(parents=True,exist_ok=False)
    factory,profile=build_factory(args.track)
    pieces,finish_s=control_path(args.track)
    model=run_episode(factory,pieces,finish_s,"controller",limit=args.limit)
    tape=model.pop("controls")
    rows=model.pop("rows")
    (target/"control-tape.json").write_text(json.dumps(tape,separators=(',',':')),encoding="utf-8")
    with (target/"episode.jsonl").open("x",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+"\n")
    replay=run_episode(factory,pieces,finish_s,"replay",tape=tape,limit=args.limit)
    replay.pop("rows")
    replay.pop("controls")
    same=(model["tick_rows_sha256"]==replay["tick_rows_sha256"]
          and model["controller_controls_sha256"]==replay["controller_controls_sha256"]
          and model["finished"]==replay["finished"]
          and model["termination"]==replay["termination"]
          and model["exception"]==replay["exception"])
    log={
      "type":"TRAIN-only engineering independent controller in real Physics-v4 Gym",
      "run_id":args.run_id,
      "track_id":args.track,
      "source_bindings":source_hashes(),
      "git_head":"1de279c37e62a27531d661480d20fa6e095c4e84",
      "runtime_profile":profile,
      "factory_origin":"registered four-track owner-v3 physics/action/race overlay + separately parsed paired TRAIN-5 v2 track, no scientific admission",
      "controller_source":"f1rl.research.wp7_executed_feasibility._arc_control",
      "controller_parameters":ARC_CONTROLLER,
      "path_signed_arc":asdict(ArcPath((-3.*(1 if args.track.startswith("right") else -1),5.),0.,
                             ((60.,0.),(5*math.pi,(1 if args.track.startswith("right") else -1)*.1),(73.,0.)))),
      "seed":20261106,
      "limit":args.limit,
      "ppo_models_loaded":False,
      "heldout_geometry_file_opened":False,
      "training_authorized":False,
      "scientific_evidence":False,
      "controller_episode":model,
      "replay_episode":replay,
      "replay_exact_match":same,
    }
    (target/"summary.json").write_text(json.dumps(log,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
      "run_id":args.run_id,"track":args.track,
      "finished":model["finished"],"steps":model["episode_steps"],
      "checkpoints":model["checkpoints_completed"],
      "terminal":model["termination"],"exception":model["exception"],
      "position":model["last_position"],"speed":model["last_speed_mps"],
      "replay_exact_match":same,
      "output":str(target)
    },allow_nan=False),flush=True)
    if not same:
        raise AssertionError("Deterministic replay diverged")
if __name__=="__main__":main()
