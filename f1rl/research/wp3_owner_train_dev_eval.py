"""Owner-authorized TRAIN-development 5,760-slot evaluation coordinator.

Never changes the registered *scientific* evaluation gate, which still requires
independent Physics-v4 certification. This is a distinct owner-approved,
TRAIN-only development protocol with immutable training models and append-only
per-episode evidence. It cannot claim scientific signoff or a ranking until
every canonical cell has 60 valid episodes.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import subprocess
import time
import traceback

from .wp3_eval_only_admission import (
    MODELS, TRACKS, STRATA, _roster_check, digest, hash_file, _json)
from .wp3_eval_only_coordinator import preflight, _source, _production_episode, PHYSICS, RACE, CONTRACT, TRAIN_HEAD

SCHEMA = "wp3_owner_approved_train_development_eval_v1"
APPROVAL = "owner_approved_train_development_only"
MAX_WORKERS = 3  # 16 GB RAM, low free memory on handoff machine
MODEL_ORDER = tuple(MODELS)


def _write_exclusive(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _append(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf8", newline="\n") as f:
        f.write(json.dumps(data, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def plan_roster(roster: dict) -> list[dict]:
    # Existing validator enforces all 480 unique paired seed slots, 96 cells,
    # no collisions, correct task, checkpoints and seed index.
    _roster_check(roster, {"evaluation_roster_sha256": roster["sha256"],
                           "training_source_commit": TRAIN_HEAD})
    # Preserve the EXACT canonical cell enumeration order as frozen in the
    # original roster, not a newly sorted candidate/model order.
    ordered = roster["seed_rows"]
    assert len(ordered) == 96
    cells = []
    for n, r in enumerate(ordered):
        m, t, s = r["model_id"], r["track_id"], r["stratum"]
        cell = {"index": n, "model_id": m, "track": t, "stratum": s,
                "seed_index": r["seed_index"],
                "episodes": r["episodes"], "episodes_sha256": digest(r["episodes"])}
        cells.append(cell)
    assert sum(len(x["episodes"]) for x in cells) == 5760
    return cells


def cell_filename(cell: dict) -> str:
    return f"{cell['index']:03d}-{cell['model_id']}-{cell['track']}-{cell['stratum']}.jsonl"


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Disallow corrupt/incomplete JSON lines: caller must preserve file and
    # report interruption rather than silently truncate or retry.
    with path.open("r", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_cell(events: list[dict], cell: dict, run_id: str) -> tuple[list[dict], dict | None]:
    """Verify immutable plan, no retry, and monotonically ordered attempts.

    Returns completed event records and an unmatched episode_started marker.
    """
    results, pending = [], None
    for event in events:
        if event.get("run_id") != run_id or event.get("cell_index") != cell["index"]:
            raise ValueError("run ID or cell identity mismatch")
        kind = event.get("event")
        idx = event.get("episode")
        if type(idx) is not int or not 0 <= idx < 60:
            raise ValueError("invalid original episode slot")
        if kind == "episode_started":
            if pending is not None or idx != len(results):
                raise ValueError("reordered/duplicate/unfinished episode attempt")
            pair = cell["episodes"][idx]
            if (event.get("environment_seed") != pair["environment_seed"]
                    or event.get("policy_seed") != pair["policy_seed"]):
                raise ValueError("paired seed drift in actual event")
            pending = event
        elif kind in ("episode_completed", "episode_invalid"):
            if pending is None or idx != pending["episode"]:
                raise ValueError("missing start/duplicate evaluation result")
            rec = event.get("result")
            if type(rec) is not dict or rec.get("status") not in (
                    "ENGINEERING_EPISODE_COMPLETED", "ENGINEERING_EPISODE_INVALID"):
                raise ValueError("malformed episode result")
            if (kind == "episode_completed") != (
                    rec["status"] == "ENGINEERING_EPISODE_COMPLETED"):
                raise ValueError("result contract mismatch")
            results.append(rec)
            pending = None
        else:
            raise ValueError("unknown event type")
    return results, pending


def _case(cell: dict, index: int) -> dict:
    s = cell["episodes"][index]
    return {"model_id": cell["model_id"], "track": cell["track"],
            "stratum": cell["stratum"], "seed_index": cell["seed_index"],
            "episode": index, "env_seed": s["environment_seed"],
            "policy_seed": s["policy_seed"]}


def _outcome(rec: dict) -> str:
    if rec["status"] != "ENGINEERING_EPISODE_COMPLETED":
        return "invalid"
    if rec.get("success_unaccepted") is True:
        return "finish"
    return rec.get("failure_reason") or "terminal_other"


def _event(run_id: str, cell: dict, index: int, kind: str, **fields) -> dict:
    return {"run_id": run_id, "cell_index": cell["index"], "model_id": cell["model_id"],
            "track": cell["track"], "stratum": cell["stratum"], "episode": index,
            "event": kind, **fields}


def execute_cell(root: str, history: str, out: str, run_id: str,
                 cell: dict, probe: bool = False) -> dict:
    """Subprocess worker; may run after interruption. No replacement/redraw.

    Model is loaded once per cell and CPU torch threads are restricted.
    A half-written/stale event is never deleted.
    """
    from pathlib import Path
    root, history, out = Path(root), Path(history), Path(out)
    manifest = _json(out / "RUN-MANIFEST.json")
    if manifest["source_head"] != _source(root):
        raise RuntimeError("evaluator execution source has changed")
    if manifest["cells"][cell["index"]] != cell:
        raise RuntimeError("canonical plan cell not source bound")
    if manifest["sha256"] != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise RuntimeError("manifest checksum mismatch")
    if not probe:
        # Framework imports belong to the approved PPO backend, not here.
        os.environ.setdefault('OMP_NUM_THREADS','1')
        os.environ.setdefault('MKL_NUM_THREADS','1')
    path = out / "cells" / cell_filename(cell)
    cache = {}
    prior, pending = parse_cell(read_events(path), cell, run_id)
    if pending is not None:
        # Do not retry a possibly already executed event. Keep invalid as
        # terminal evidence and abort cell.
        i = pending["episode"]
        rec = {"status":"ENGINEERING_EPISODE_INVALID", "error_type":"InterruptedAttempt",
               "error":"Started with no persisted completion; no redraw",
               "success_unaccepted":False, "outcome":"invalid"}
        _append(path, _event(run_id, cell, i, "episode_invalid", result=rec))
        prior.append(rec)
    if any(r["status"] != "ENGINEERING_EPISODE_COMPLETED" for r in prior):
        return {"index":cell["index"],"status":"ABORTED_INVALID","completed":len(prior),
                "valid":sum(r["status"]=="ENGINEERING_EPISODE_COMPLETED" for r in prior)}
    for i in range(len(prior),60):
        case = _case(cell,i)
        _append(path, _event(run_id, cell, i, "episode_started",
                             environment_seed=case["env_seed"],
                             policy_seed=case["policy_seed"]))
        begun = time.monotonic()
        try:
            if probe:
                metric = {"success":True,"failure_reason":None,
                          "episode_time":1.0,"final_normalized_route_progress":1.,
                          "collision_count":0,"completed_checkpoints":1}
            else:
                metric = _production_episode(root,history,case,cache)
            rec = {
                "status":"ENGINEERING_EPISODE_COMPLETED",
                "success_unaccepted":bool(metric["success"]),
                "failure_reason":metric.get("failure_reason"),
                "episode_time_s":metric.get("episode_time"),
                "progress_unaccepted":metric.get("final_normalized_route_progress"),
                "collision_count":metric.get("collision_count"),
                "completed_checkpoints":metric.get("completed_checkpoints"),
                "hard_route_violation":metric.get("hard_route_violation"),
                "timeout":metric.get("timeout"),
                "steering_rate_rms_rad_per_s":metric.get("steering_rate_rms_rad_per_s"),
                "runtime_s":round(time.monotonic()-begun,4),
            }
        except Exception as exc:
            rec={"status":"ENGINEERING_EPISODE_INVALID",
                 "success_unaccepted":False, "error_type":type(exc).__name__,
                 "error":str(exc),"traceback":traceback.format_exc(limit=5),
                 "runtime_s":round(time.monotonic()-begun,4)}
        rec["outcome"]=_outcome(rec)
        kind=("episode_completed" if rec["status"]=="ENGINEERING_EPISODE_COMPLETED"
              else "episode_invalid")
        _append(path,_event(run_id,cell,i,kind,result=rec))
        print(f"TRAIN_DEV_CELL {cell['index']:03d}/096 model={cell['model_id']} "
              f"track={cell['track']} stratum={cell['stratum']} "
              f"episode={i+1:02d}/60 outcome={rec['outcome']}",flush=True)
        if kind=="episode_invalid":
            return {"index":cell["index"],"status":"ABORTED_INVALID",
                    "completed":i+1,"valid":i}
    return {"index":cell["index"],"status":"CELL_COMPLETE","completed":60,"valid":60}


def new_run(root: Path, history: Path, out: Path, run_id: str,
            approval_message: str) -> dict:
    root,history,out=Path(root).resolve(),Path(history).resolve(),Path(out).resolve()
    if (not re.fullmatch(r"WP3-OWNER-TRAINDEV-[a-zA-Z0-9_-]{6,75}",run_id)
            or approval_message != "I approve the internal TRAIN-development evaluation"):
        raise ValueError("explicit owner-authorization and new unique owner run ID required")
    check,roster,prov=preflight(root,history)
    if check["current_evaluation_source_commit"]!=_source(root):
        raise ValueError("source has changed during preflight")
    assert len(prov["models"])==12 and len(roster["seed_rows"])==96
    assets={}
    for m in prov["models"]:
        p=history/"runs"/m["model_id"]/"model-final.zip"
        actual=hash_file(p)
        if actual!=m["model_final_sha256"]:
            raise ValueError("frozen final PPO model mutated")
        assets[m["model_id"]]=actual
    cells=plan_roster(roster)
    if out.exists():
        raise FileExistsError(f"new evaluation output must not already exist: {out}")
    manifest={
        "schema_version":SCHEMA,"run_id":run_id,
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_head":check["current_evaluation_source_commit"],
        "training_source":TRAIN_HEAD,
        "roster_sha256":roster["sha256"],
        "roster_file_sha256":hash_file(history/"evaluation/evaluation-roster.json"),
        "training_provenance_sha256":hash_file(history/"WP3-EVAL-SOURCE-PROVENANCE.json"),
        "original_final_model_sha256":assets,
        "physics_version":PHYSICS,"race_version":RACE,
        "crash_terminal_contract":CONTRACT,
        "protocol":"owner_approved_internal_train_dev_v1",
        "approval_basis":APPROVAL,
        "owner_authorization":"Owner explicitly approved proceeding with TRAIN development Evaluation in chat on 2026-10-24",
        "independent_physics_review":None,
        "scientific_signoff":False,"scientific_evidence":False,
        "scientific_registered_evaluation":False,
        "research_validation_status":"REQUIRES_INDEPENDENT_SCIENTIFIC_REVIEW",
        "heldout_access":False,"retraining":False,
        "policy_inference":"CPU stochastic deterministic_false",
        "hard_corridor_m":4.0,
        "expected_cells":96,"expected_episode_slots":5760,
        "on_invalid":"abort affected cell, keep invalid, no redraw",
        "cells":cells,
    }
    manifest["sha256"]=digest(manifest)
    out.mkdir(parents=True)
    (out/"cells").mkdir()
    _write_exclusive(out/"RUN-MANIFEST.json",manifest)
    _write_exclusive(out/"RUN-STATE-INITIAL.json",{
        "run_id":run_id,"status":"OWNER_AUTHORIZED_TRAIN_DEVELOPMENT",
        "scientific_registered":False,"scientific_gate_still_blocked":True,
        "episodes_started":0,"source_head":manifest["source_head"],
        "source_manifest_sha256":manifest["sha256"]})
    return manifest


def load_run(root: Path, history: Path, out: Path) -> dict:
    manifest=_json(out/"RUN-MANIFEST.json")
    if manifest["sha256"]!=digest({k:v for k,v in manifest.items() if k!="sha256"}):
        raise ValueError("tampered run manifest checksum")
    if (manifest["schema_version"]!=SCHEMA or manifest["source_head"]!=_source(root)
            or manifest["training_source"]!=TRAIN_HEAD
            or manifest["approval_basis"]!=APPROVAL
            or manifest["scientific_evidence"] is not False
            or manifest["heldout_access"] is not False
            or manifest["expected_cells"]!=96
            or manifest["expected_episode_slots"]!=5760):
        raise ValueError("owner Eval-only integrity or scientific boundary changed")
    r=_json(history/"evaluation/evaluation-roster.json")
    p=_json(history/"WP3-EVAL-SOURCE-PROVENANCE.json")
    _roster_check(r,p)
    if (hash_file(history/"evaluation/evaluation-roster.json")!=manifest["roster_file_sha256"]
            or hash_file(history/"WP3-EVAL-SOURCE-PROVENANCE.json")!=manifest["training_provenance_sha256"]
            or plan_roster(r)!=manifest["cells"]):
        raise ValueError("original frozen roster or original provenance drift")
    for mid, expected in manifest["original_final_model_sha256"].items():
        if hash_file(history/"runs"/mid/"model-final.zip")!=expected:
            raise ValueError("frozen model ZIP changed")
    return manifest


def snapshot(root:Path,history:Path,out:Path) -> dict:
    manifest=load_run(root,history,out)
    counts=Counter()
    rows=[]
    cell_status=[]
    for cell in manifest["cells"]:
        name=cell_filename(cell)
        ev=read_events(out/"cells"/name)
        results,pending=parse_cell(ev,cell,manifest["run_id"])
        if pending is not None:
            counts["inflight"]+=1
        nvalid=sum(x["status"]=="ENGINEERING_EPISODE_COMPLETED" for x in results)
        ninvalid=len(results)-nvalid
        counts["valid"]+=nvalid
        counts["invalid"]+=ninvalid
        counts["finished_driving"]+=sum(x.get("outcome")=="finish" for x in results)
        counts["crash"]+=sum(x.get("outcome")=="physical_crash" for x in results)
        counts["hard_route"]+=sum(x.get("outcome")=="hard_route_violation" for x in results)
        counts["step_limit"]+=sum(x.get("outcome")=="step_limit" for x in results)
        complete=len(results)==60 and ninvalid==0 and pending is None
        cell_status.append({"index":cell["index"],"model_id":cell["model_id"],
                            "track":cell["track"],"stratum":cell["stratum"],
                            "valid":nvalid,"invalid":ninvalid,
                            "started_unfinished":pending is not None,
                            "complete":complete,"event_log_sha256":hash_file(out/"cells"/name)
                            if (out/"cells"/name).exists() else None})
        for i,x in enumerate(results):
            rows.append({"model_id":cell["model_id"],"track":cell["track"],
                         "stratum":cell["stratum"],"episode":i,"outcome":x["outcome"],
                         "status":x["status"],"success":x.get("success_unaccepted"),
                         "progress":x.get("progress_unaccepted"),
                         "collision_count":x.get("collision_count")})
    complete=sum(x["complete"] for x in cell_status)
    summary={
        "schema_version":SCHEMA,"run_id":manifest["run_id"],
        "source_head":manifest["source_head"],"manifest_sha256":manifest["sha256"],
        "valid_episodes":counts["valid"],"invalid_episodes":counts["invalid"],
        "inflight_episodes":counts["inflight"],
        "started_episodes":counts["valid"]+counts["invalid"]+counts["inflight"],
        "unattempted_slots":5760-(counts["valid"]+counts["invalid"]+counts["inflight"]),
        "complete_cells":complete,"total_cells":96,
        "driving_finishes":counts["finished_driving"],"physical_crashes":counts["crash"],
        "hard_route":counts["hard_route"],"step_limit":counts["step_limit"],
        "all_cells_complete":complete==96,
        "scientific_evidence":False,"scientific_registered_evaluation":False,
        "scientific_gate":"BLOCKED_PENDING_INDEPENDENT_REVIEW",
        "cell_status":cell_status}
    return summary


def run(root:Path, history:Path, out:Path, workers:int=1) -> dict:
    if type(workers) is not int or not 1<=workers<=MAX_WORKERS:
        raise ValueError("workers must be 1..3 on 16GB host")
    manifest=load_run(root,history,out)
    # Single controlling process: exclusive lock. Does not discard orphan attempt
    # evidence; resume explicitly calls worker recovery.
    lock=out/"OWNER-RUNNING.lock"
    try:
        with lock.open("x",encoding="utf8") as f:
            f.write(f"{os.getpid()}\n")
    except FileExistsError:
        raise RuntimeError("Run already has a controller; no concurrent owners")
    try:
        # Canonical 12 x 4 x 2 cell order; concurrent completion may differ, but
        # cell identity and per-cell episode sequence are immutable.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(execute_cell,str(root),str(history),str(out),
                                 manifest["run_id"],cell):cell["index"]
                     for cell in manifest["cells"]}
            for fut in as_completed(futures):
                ix=futures[fut]
                try:
                    result=fut.result()
                    print(f"CELL_RESULT {ix+1:03d}/096 {result}",flush=True)
                except Exception as exc:
                    # Keep files, leave cell incomplete, and surface the error.
                    print(f"CELL_WORKER_CRASH {ix:03d} {type(exc).__name__} {exc}",
                          flush=True)
        result=snapshot(root,history,out)
        _write_exclusive(out/"OWNER-TRAINDEV-FINAL-SUMMARY.json",result)
        # Generate all requested reports/three SVGs once and ONLY once after
        # the complete 96-cell attempt. Incomplete cells suppress ranking.
        from .wp3_owner_train_dev_reports import write_reports
        provenance=_json(history/"WP3-EVAL-SOURCE-PROVENANCE.json")
        write_reports(out,history,manifest,result,provenance)
        return result
    finally:
        lock.unlink(missing_ok=True)
