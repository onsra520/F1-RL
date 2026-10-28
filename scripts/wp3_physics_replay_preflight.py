"""Read-only three-case original WP3 frozen model/seed physics preflight.

Never creates a campaign, registered evaluation, model checkpoint, or new
scientific evidence. Uses the original locked evaluation roster and exact
frozen final checkpoints. Fails the FULL_EVAL_GATE if ANY physics exception.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from hashlib import sha256
import json,sys,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.learning.algorithms.ppo import PPOModel
from src.local_demo.evaluation import _evaluate_episode
from src.research.wp7_protocol import build_protocol,prepare_evaluation_factory

import os
E=Path(os.environ.get("F1RL_WP3_EVIDENCE", r"D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1"))
OUT=ROOT/"docs"/"research"/"WP3-CE0E55D-THREE-REPLAY-PREFLIGHT-20261024.json"
CASES=(
    ("C-seed0","right-turn-v1","nominal",0),
    ("A-seed1","ramp-v1","randomized_train",17),
    ("C-seed0","slalom-v1","nominal",2),
)
def h(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if actual!='ce0e55d7a428826b2d4ed3218641105bd0206f4b':
        raise ValueError('source HEAD is not the exact handed-off physical completion base')
    if OUT.exists():
        raise FileExistsError('scratch replay report already exists; refuse overwrite/re-draw')
    roster=json.loads((E/"evaluation/evaluation-roster.json").read_text("utf8"))
    prov=json.loads((E/"WP3-EVAL-SOURCE-PROVENANCE.json").read_text("utf8"))
    assert roster["expected_cells"]==96 and roster["expected_episodes"]==5760
    assert roster["sha256"]==prov["evaluation_roster_sha256"]
    bymodel={x["model_id"]:x for x in prov["models"]}
    protocol=build_protocol(ROOT,profile="spatial_rotating_owner_v3")
    results=[]
    for model_id,track,stratum,ep in CASES:
        row=next(x for x in roster["seed_rows"] if x["model_id"]==model_id
                 and x["track_id"]==track and x["stratum"]==stratum)
        seeds=row["episodes"][ep]
        final=E/"runs"/model_id/"model-final.zip"
        assert h(final)==bymodel[model_id]["model_final_sha256"]
        model=PPOModel.load(final,device="cpu")
        assert model.num_timesteps==65536
        fac,_=prepare_evaluation_factory(ROOT,protocol,
            f"multitrack-seed{row['seed_index']}",track,stratum,0)
        if stratum=="randomized_train":
            fac=replace(fac,reset_config=replace(fac.reset_config,seed=seeds["environment_seed"]))
        assert fac.race_config.hard_corridor_m==4.
        begin=time.monotonic()
        entry={"model_id":model_id,"track":track,"stratum":stratum,
               "episode":ep,"seed_environment":seeds["environment_seed"],
               "seed_policy":seeds["policy_seed"],
               "model_sha256":h(final),"source_physics_profile":"spatial_rotating_owner_v3"}
        try:
            metrics=_evaluate_episode(model,fac,track,seeds["environment_seed"],
                seeds["policy_seed"],fac.simulation_config.fixed_dt)
            entry.update(status="VALID_EPISODE",success=bool(metrics["success"]),
                         failure_reason=metrics.get("failure_reason"),
                         episode_time=metrics.get("episode_time"),
                         route_progress=metrics.get("final_normalized_route_progress"))
        except Exception as exc:
            entry.update(status="INVALID_UNSCORED_PHYSICS_OR_EVAL",
                         error_type=type(exc).__name__,error=str(exc))
        entry["elapsed_s"]=round(time.monotonic()-begin,5)
        results.append(entry)
        print("REPLAY",model_id,track,stratum,ep,entry["status"],
              entry.get("error_type",entry.get("failure_reason")),flush=True)
    full_gate=all(row["status"]=="VALID_EPISODE" for row in results)
    payload={"schema_version":"wp3_three_frozen_case_scratch_preflight_v1",
             "run_date_utc":datetime.now(timezone.utc).isoformat(),
             "source_commit":"ce0e55d7a428826b2d4ed3218641105bd0206f4b",
             "training_commit":"fe83f517479f3b990028d6409304fae83b8199a9",
             "original_roster_sha256":roster["sha256"],
             "original_roster_file_sha256":h(E/"evaluation/evaluation-roster.json"),
             "mode":"TRAIN_ONLY_SCRATCH_NO_REGISTERED_EVAL",
             "original_evidence_immutable":True,
             "physics_profile":"spatial_rotating_owner_v3",
             "new_physics_profile_registered":False,
             "full_eval_gate": "PASS" if full_gate else "BLOCKED",
             "completed_valid_scratch_episodes":sum(r["status"]=="VALID_EPISODE" for r in results),
             "invalid_scratch_episodes":sum(r["status"]!="VALID_EPISODE" for r in results),
             "results":results,
             "new_registered_eval_runs":0,
             "scientific_evidence":False,
             "heldout_access":False}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("x",encoding="utf8",newline="\n") as output:
        json.dump(payload,output,indent=2,ensure_ascii=False,allow_nan=False)
        output.write("\n")
    print("FROZEN_REPLAY_GATE",payload["full_eval_gate"],
          "INVALID",payload["invalid_scratch_episodes"],flush=True)

if __name__=="__main__":main()
