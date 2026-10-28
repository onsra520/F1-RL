"""Run exactly 3 frozen model/seed events in UNREGISTERED development v4.

Never claims certified physics or lets 'no Python error' mint a valid-scored
episode. Does not touch old evaluation, checkpoint, campaign, or heldout.
"""
from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
from hashlib import sha256
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from f1rl.learning.algorithms.ppo import PPOModel
from f1rl.local_demo.evaluation import _evaluate_episode
from f1rl.research.wp7_protocol import build_protocol,prepare_evaluation_factory

import os
OLD=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
OUT=ROOT/"docs"/"research"/"WP3-V4V-THREE-FROZEN-REPLAY-DEV-CRASH-20261024.json"
CASES=(("C-seed0","right-turn-v1","nominal",0),
       ("A-seed1","ramp-v1","randomized_train",17),
       ("C-seed0","slalom-v1","nominal",2))
V4="spatial_road_edge_partial_lifecycle_v4"

def sha(path):
    h=sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):h.update(data)
    return h.hexdigest()

def main():
    if OUT.exists():raise FileExistsError("refuse overwrite existing scratch")
    roster=json.loads((OLD/"evaluation"/"evaluation-roster.json").read_text("utf8"))
    provenance=json.loads((OLD/"WP3-EVAL-SOURCE-PROVENANCE.json").read_text("utf8"))
    assert roster["sha256"]==provenance["evaluation_roster_sha256"]
    assert roster["expected_episodes"]==5760 and roster["expected_cells"]==96
    proto=build_protocol(ROOT,profile="spatial_rotating_owner_v3")
    models={r["model_id"]:r for r in provenance["models"]}
    logs=[]
    for mid,track,stratum,index in CASES:
        seq=next(s for s in roster["seed_rows"] if
             (s["model_id"],s["track_id"],s["stratum"])==(mid,track,stratum))
        e=seq["episodes"][index]
        modelpath=OLD/"runs"/mid/"model-final.zip"
        assert sha(modelpath)==models[mid]["model_final_sha256"]
        model=PPOModel.load(modelpath,device="cpu")
        assert model.num_timesteps==65536
        factory,_=prepare_evaluation_factory(ROOT,proto,
                      f"multitrack-seed{seq['seed_index']}",track,stratum,0)
        if stratum=="randomized_train":
            factory=replace(factory,reset_config=replace(
                factory.reset_config,seed=e["environment_seed"]))
        factory=replace(factory,physics_config=replace(
            factory.physics_config,ground_motion_version=V4))
        factory=replace(factory,race_config=replace(factory.race_config,version='wp7_directed_lifecycle_crash_dev_v3'))
        assert factory.race_config.hard_corridor_m==4.
        record={"model_id":mid,"track":track,"stratum":stratum,
                "episode":index,"env_seed":e["environment_seed"],
                "policy_seed":e["policy_seed"],
                "model_final_sha256":sha(modelpath),
                "scratch_physics_version":V4,
                "scientific_evidence":False}
        start=time.monotonic()
        try:
            result=_evaluate_episode(model,factory,track,e["environment_seed"],
                  e["policy_seed"],factory.simulation_config.fixed_dt)
            record.update({
                "status":"ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED",
                "success_flag_unaccepted":bool(result["success"]),
                "failure_reason":result.get("failure_reason"),
                "duration_s":result.get("episode_time"),
                "progress_not_certified":result.get("final_normalized_route_progress")})
        except Exception as error:
            record.update({"status":"INVALID_UNCERTIFIED_PHYSICS_OR_EVAL",
                           "error_type":type(error).__name__,"error":str(error)})
        record["walltime_s"]=round(time.monotonic()-start,4)
        logs.append(record)
        print("V4_REPLAY",mid,track,stratum,index,
              record["status"],record.get("error_type"),flush=True)
    doc={"schema_version":"wp3_v4_unregistered_three_case_scratch_v1",
         "created_at_utc":datetime.now(timezone.utc).isoformat(),
         "source_base_commit":subprocess.check_output(
            ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
         "frozen_train_sha":"fe83f517479f3b990028d6409304fae83b8199a9",
         "roster_sha256":roster["sha256"],"roster_file_sha256":sha(OLD/"evaluation"/"evaluation-roster.json"),
         "physics_profile":V4,"experiment_status":"SCRATCH_NOT_REGISTERED",
         "all_required_replays_scientifically_certified":False,
         "full_5760_eval_admission":"BLOCKED",
         "registered_evaluation_executed":False,
         "original_evidence_mutated":False,"heldout_access":False,
         "cases":logs}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("x",encoding="utf8",newline="\n") as f:
        json.dump(doc,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    print("V4_CASES_COMPLETED",len(logs),
          "INVALID",sum(r["status"].startswith("INVALID") for r in logs),
          "FULL_EVAL_GATE",doc["full_5760_eval_admission"],flush=True)

if __name__=="__main__":main()
