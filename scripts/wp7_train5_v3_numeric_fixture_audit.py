"""One-shot TRAIN-5 v3 numerical engineering fixture, not physical calibration.
Fixed scripted inputs; no PPO, training, HELD-OUT or reserved seeds.
"""
from __future__ import annotations

import argparse, json, math, sys, traceback
from pathlib import Path
from hashlib import sha256

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
E=Path(r"D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106")
SPEC=ROOT/"configs/research/wp7-train5-v3-fixture-and-development-criteria.json"


class FixtureContractError(ValueError):pass


def digest(p:Path)->str:
    return sha256(p.read_bytes()).hexdigest()


def load_spec()->dict:
    from f1rl.research.wp7_train5_v2_protocol import TRAIN5_IDS,HELDOUT_IDS
    x=json.loads(SPEC.read_text("utf8"))
    if (x.get("schema")!="f1rl_train5_v3_nontraining_fixture_and_proposed_development_criteria_v1"
        or tuple(x.get("track_ids",()))!=TRAIN5_IDS
        or tuple(x.get("heldout_ids_not_read",()))!=HELDOUT_IDS
        or x.get("fixture_reset_seeds")!=[20261107,20261108,20261109]
        or x.get("reset_indices")!=[0,1]
        or x.get("expected_fixture_reset_count")!=30
        or x["permissions"].get("ppo_training") is not False
        or x["permissions"].get("heldout") is not False
        or x["scientific_admission_approved"] is not False):
        raise FixtureContractError("Unregistered fixture contract or permission changed")
    return x


def validate_leaf(leaf:dict, limits:dict)->float:
    values=("energy_residual_j","energy_tolerance_j",
            "dissipation_j","linear_impulse_residual_ns","angular_impulse_residual_nms")
    if any(type(leaf.get(n)) not in (int,float) or not math.isfinite(leaf[n]) for n in values):
        raise FixtureContractError("Non-finite physics budget term")
    if leaf["energy_tolerance_j"]<=0:
        raise FixtureContractError("No positive physics tolerance")
    frac=abs(leaf["energy_residual_j"])/leaf["energy_tolerance_j"]
    if (frac>limits["energy_residual_over_tolerance_max"]
        or leaf["dissipation_j"]<limits["min_dissipation_j"]
        or leaf["linear_impulse_residual_ns"]>limits["linear_impulse_residual_ns_max"]
        or leaf["angular_impulse_residual_nms"]>limits["angular_impulse_residual_nms_max"]):
        raise FixtureContractError("V4 energy/impulse conservation violated")
    return frac


def run_fixture()->dict:
    from f1rl.research.wp7_train5_reset_perturbation_v3 import (
        prepare_train5_perturbed_scenarios,
    )
    from f1rl.research.wp7_train5_v2_protocol import TRAIN5_IDS
    x=load_spec()
    limits=x["predeclared_numerical_acceptance"]
    assert x["scripted_control_profile"]=={
        "steering":0,"throttle":0.5,"brake":0,"steps_per_reset":12}
    rows=[]
    offsets={}
    for seed in x["fixture_reset_seeds"]:
        scenarios=prepare_train5_perturbed_scenarios(
            ROOT,evidence_root=E,perturbation_seed=seed,
        )
        if tuple(y.track_id for y in scenarios.scenarios)!=TRAIN5_IDS:
            raise FixtureContractError("TRAIN5 factory identities drifted")
        for item in scenarios.scenarios:
            for index in x["reset_indices"]:
                env=item.factory.create(reset_rank=x["reset_rank"],reset_sequence=index)
                record={"track_id":item.track_id,"fixture_seed":seed,
                        "reset_index":index,"valid_observations":0,
                        "valid_actions":0,"finite_rewards":0,
                        "physics_leaves":0,"steps":0,"terminal":None,
                        "max_energy_residual_fraction":0.0,"max_speed":0.}
                try:
                    obs,info=env.reset(seed=seed)
                    if not env.observation_space.contains(obs):
                        raise FixtureContractError("Invalid perturbed initial observation")
                    meta=info.get("train_reset")
                    if (not isinstance(meta,dict) or meta.get("seed")!=seed or
                        meta.get("track_id")!=item.track_id or
                        meta.get("reset_index")!=index or
                        meta.get("rank")!=x["reset_rank"]):
                        raise FixtureContractError("Missing independent reset provenance")
                    lateral=meta["lateral_offset_m"]
                    heading=meta["heading_jitter_deg"]
                    if (not x["reset_lateral_range_m"][0]<=lateral<=x["reset_lateral_range_m"][1]
                        or not x["reset_heading_range_deg"][0]<=heading<=x["reset_heading_range_deg"][1]):
                        raise FixtureContractError("Out-of-contract perturbation")
                    record["reset"]={k:meta[k] for k in (
                        "configuration_sha256","world_sha256","draw_seed",
                        "lateral_offset_m","heading_jitter_deg","attempt",
                    )}
                    record["valid_observations"]+=1
                    offsets[(seed,item.track_id,index)]=(lateral,heading)
                    for _ in range(x["scripted_control_profile"]["steps_per_reset"]):
                        action=np.asarray([0.,0.5,0.],dtype=np.float32)
                        if not env.action_space.contains(action):
                            raise FixtureContractError("Invalid control action")
                        record["valid_actions"]+=1
                        obs,reward,t,u,info=env.step(action)
                        if not env.observation_space.contains(obs):
                            raise FixtureContractError("Invalid transition observation")
                        if not math.isfinite(float(reward)):
                            raise FixtureContractError("Invalid reward")
                        outcome=info.get("research_outcome")
                        if not isinstance(outcome,dict) or outcome.get("valid_sample") is not True:
                            raise FixtureContractError("Invalid scientific transition")
                        record["valid_observations"]+=1
                        record["finite_rewards"]+=1
                        record["steps"]+=1
                        phys=info["physics"]
                        if phys.get("sha256")!=item.factory.physics_config.sha256:
                            raise FixtureContractError("Physics config changed")
                        stage=phys.get("spatial_ground",{})
                        if stage.get("version") is None:
                            raise FixtureContractError("No spatial ground version")
                        leaves=stage.get("substeps",[])
                        if not leaves:
                            raise FixtureContractError("No physics contact substeps")
                        for leaf in leaves:
                            frac=validate_leaf(leaf,limits)
                            record["max_energy_residual_fraction"]=max(
                                record["max_energy_residual_fraction"],frac)
                        record["physics_leaves"]+=len(leaves)
                        speed=float(info["telemetry"]["vehicle_speed"])
                        if not math.isfinite(speed) or speed>item.factory.vehicle_parameters.max_ground_speed:
                            raise FixtureContractError("Invalid numerical speed")
                        record["max_speed"]=max(speed,record["max_speed"])
                        if t or u:
                            record["terminal"]=info["research_outcome"]["reason"]
                            break
                    rows.append(record)
                finally:
                    env.close()
    if len(rows)!=x["expected_fixture_reset_count"]:
        raise FixtureContractError("Incomplete registered numerical fixture")
    if len(set(offsets))!=30 or len(set(offsets.values()))<6:
        raise FixtureContractError("Not enough independent draws")
    for name in ("valid_observations","valid_actions","finite_rewards"):
        if any(row[name]<1 for row in rows):
            raise FixtureContractError(name+" incomplete")
    return {
        "schema":"f1rl_train5_v3_numeric_fixture_audit_result_v1",
        "spec_sha256":digest(SPEC),
        "scope":"SIMULATOR_NUMERICAL_ONLY_NOT_EXTERNAL_PHYSICAL_CALIBRATION",
        "registered_reset_fixture_count":len(rows),
        "simulated_total_steps":sum(row["steps"] for row in rows),
        "physics_contact_substeps":sum(row["physics_leaves"] for row in rows),
        "max_energy_fraction":max(row["max_energy_residual_fraction"] for row in rows),
        "invalid_numerical_transitions":0,
        "distinct_reset_draw_identities":len(set(x["reset"]["draw_seed"] for x in rows)),
        "fixture_seed_roster":x["fixture_reset_seeds"],
        "training_reserved_seeds_consumed":0,
        "heldout_access":False,
        "ppo_training":False,
        "independent_empirical_physics_calibration_verified":False,
        "results":rows,
        "verdict":"PASS_NUMERICAL_RESET_FIXTURES_ONLY",
    }


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--execute-once",action="store_true")
    args=parser.parse_args()
    if args.preflight==args.execute_once:
        parser.error("Select exactly one mode")
    s=load_spec()
    if args.preflight:
        print(json.dumps({"result":"PASS_PREPARE_ONLY","fixture_count":s["expected_fixture_reset_count"],
                          "training_seeds_consumed":0,"training_permitted":False}))
        return 0
    import subprocess
    status=subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True).strip()
    if status:
        raise FixtureContractError("Dirty source cannot run registered numerical fixture")
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    output=E/("HANDOFF-TRAIN5-V3-NUMERIC-FIXTURE-"+head[:10].upper()+"-20261107")
    if output.exists():
        raise FixtureContractError("One-shot numeric fixture already attempted")
    output.mkdir()
    (output/"ATTEMPT.json").write_text(json.dumps({
        "head":head,"spec_sha256":digest(SPEC),"attempt_consumed":True,
        "training_permission":False,"reserved_seeds_consumed":0},indent=2)+"\n",encoding="utf8")
    try:
        result=run_fixture()
        result["source_commit"]=head
        (output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
        print(json.dumps({"result":result["verdict"],"count":result["registered_reset_fixture_count"],
                          "steps":result["simulated_total_steps"],"contact_substeps":result["physics_contact_substeps"],
                          "output":str(output)},sort_keys=True),flush=True)
        return 0
    except BaseException:
        (output/"FAIL_CLOSED.json").write_text(json.dumps({
            "result":"FAIL_NO_RERUN","traceback":traceback.format_exc(),
            "training_permission":False},indent=2)+"\n",encoding="utf8")
        raise

if __name__=="__main__":
    raise SystemExit(main())
