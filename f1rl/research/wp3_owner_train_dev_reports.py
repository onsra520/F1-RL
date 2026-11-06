"""Produce audit-bound WP3 TRAIN-development reports without false scientific ranks."""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import statistics
from xml.sax.saxutils import escape

from .wp3_eval_only_admission import hash_file,digest
from .wp3_owner_train_dev_eval import cell_filename,parse_cell,read_events

def _exclusive(path:Path,text:str):
    with path.open("x",encoding="utf8",newline="\n") as f:f.write(text)


def _json_exclusive(path:Path,obj):
    with path.open("x",encoding="utf8",newline="\n") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")


def _model_stats(rows):
    valid=[r for r in rows if r["status"]=="ENGINEERING_EPISODE_COMPLETED"]
    n=len(valid)
    finishes=sum(r["outcome"]=="finish" for r in valid)
    mean=lambda field: (sum(float(r[field]) for r in valid if r.get(field) is not None)/
                         sum(r.get(field) is not None for r in valid)
                         if any(r.get(field) is not None for r in valid) else None)
    return {"attempted":len(rows),"valid":n,"invalid":len(rows)-n,
            "finishes":finishes,
            "finish_rate_unaccepted":(finishes/n if n else None),
            "mean_route_progress":mean("progress_unaccepted"),
            "mean_completed_checkpoints":mean("completed_checkpoints"),
            "mean_episode_time_s":mean("episode_time_s"),
            "mean_steering_rate_rms":mean("steering_rate_rms_rad_per_s"),
            "collision_rate":(sum((r.get("collision_count") or 0)>0 for r in valid)/n if n else None),
            "hard_route_rate":(sum(r["outcome"]=="hard_route_violation" for r in valid)/n if n else None),
            "timeout_rate":(sum(r["outcome"]=="step_limit" for r in valid)/n if n else None),
            "outcomes":dict(Counter(r["outcome"] for r in valid))}


def _svg_base(body,w=900,h=490):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<style>text{font:15px Arial,sans-serif;fill:#17243b}'
            '.title{font:bold 23px Arial,sans-serif}.small{font:12px Arial,sans-serif}</style>'
            +body+"</svg>\n")


def _bars(title,values,note):
    colors=["#4361ee","#00a6a6","#f4a261","#b36bd4"]
    body=f'<text x="42" y="45" class="title">{escape(title)}</text>'
    body+=f'<text x="42" y="75" class="small">{escape(note)}</text>'
    if not values:
        body+='<text x="42" y="160">Evaluation incomplete; rankings withheld.</text>'
        return _svg_base(body)
    for i,(label,val) in enumerate(values):
        y=122+i*81
        body+=f'<text x="42" y="{y+18}">{escape(label)}</text>'
        body+=f'<rect x="180" y="{y}" width="{float(val)*600:.1f}" height="28" fill="{colors[i%4]}"/>'
        body+=f'<text x="{185+float(val)*600:.1f}" y="{y+21}" class="small">{float(val):.1%}</text>'
    return _svg_base(body)


def _failures_svg(outcomes, complete):
    body='<text x="42" y="45" class="title">Episode termination outcomes</text>'
    body+=('<text x="42" y="75" class="small">Owner-approved TRAIN development. '
           'Not independently certified.</text>')
    total=sum(outcomes.values())
    if not total:
        body+='<text x="42" y="150">No evaluated episodes.</text>'
        return _svg_base(body)
    colors=["#4169e1","#e46f63","#d89b35","#64748b","#6b7280"]
    for i,(name,count) in enumerate(sorted(outcomes.items(),
                                          key=lambda kv:-kv[1])[:5]):
        y=120+i*68
        bar=550*count/total
        body+=f'<text x="42" y="{y+18}">{escape(name)}</text>'
        body+=f'<rect x="245" y="{y}" width="{bar:.1f}" height="25" fill="{colors[i]}"/>'
        body+=f'<text x="{255+bar:.1f}" y="{y+19}">{count}</text>'
    if not complete:
        body+='<text x="42" y="470" class="small">INCOMPLETE: displayed counts only, no ranking.</text>'
    return _svg_base(body)


def _learning_progress_svg(history,model_ids,expected_hashes):
    # Aggregate TRAIN-only step reward in 4,096-timestep buckets for each of
    # the original frozen 12 run logs; digest-verify before reading any data.
    curves={}
    for mid in model_ids:
        p=history/"runs"/mid/"training-telemetry.jsonl"
        if hash_file(p)!=expected_hashes[mid]:
            raise ValueError("frozen training telemetry SHA mismatch")
        bins=[[] for _ in range(16)]
        with p.open("r",encoding="utf8") as stream:
            for line in stream:
                obj=json.loads(line)
                if obj.get("event_type")!="step":
                    continue
                step=obj.get("global_timestep")
                reward=obj.get("registered_reward")
                if type(step) is not int or not 1<=step<=65536 or not isinstance(reward,(int,float)):
                    continue
                bins[(step-1)//4096].append(float(reward))
        if any(not b for b in bins):
            raise ValueError("no real TRAIN-step reward in one telemetry bin")
        curves[mid]=[sum(b)/len(b) for b in bins]
    series={k:[statistics.mean(curves[f"{k}-seed{i}"][t] for i in range(3))
               for t in range(16)] for k in "ABCD"}
    vals=[v for a in series.values() for v in a]
    lo=min(vals)
    hi=max(vals)
    if hi-lo <1e-9:hi=lo+1.
    body='<text x="42" y="45" class="title">Frozen TRAIN learning progress</text>'
    body+=('<text x="42" y="75" class="small">Mean registered reward per 4096 training steps,'
           ' averaged over 3 original PPO seeds. TRAIN telemetry only.</text>')
    colors={"A":"#4361ee","B":"#00a6a6","C":"#f4a261","D":"#b36bd4"}
    for i,(k,arr) in enumerate(series.items()):
        pts=[]
        for j,v in enumerate(arr):
            x=100+j*(680/15)
            y=410-(v-lo)/(hi-lo)*250
            pts.append(f"{x:.1f},{y:.1f}")
        body+=f'<polyline points="{" ".join(pts)}" stroke="{colors[k]}" fill="none" stroke-width="2.5"/>'
        body+=f'<text x="{130+i*155}" y="453">{k} candidate</text>'
        body+=f'<line x1="{105+i*155}" y1="448" x2="{124+i*155}" y2="448" stroke="{colors[k]}" stroke-width="3"/>'
    body+=f'<text x="100" y="430" class="small">4k timesteps</text>'
    body+=f'<text x="675" y="430" class="small">65,536</text>'
    body+=f'<text x="40" y="168" class="small">Max {hi:.3f}</text>'
    body+=f'<text x="40" y="418" class="small">Min {lo:.3f}</text>'
    return _svg_base(body)


def write_reports(out:Path, history:Path, manifest:dict, snapshot:dict,
                  training_provenance:dict) -> dict:
    """Exclusive outputs; complete or explicitly incomplete, never fictitious."""
    if snapshot["run_id"]!=manifest["run_id"] or snapshot["manifest_sha256"]!=manifest["sha256"]:
        raise ValueError("run summary not bound to manifest")
    cell_rows={}
    per_cell=[]
    all_rows=[]
    for cell in manifest["cells"]:
        path=out/"cells"/cell_filename(cell)
        result,pending=parse_cell(read_events(path),cell,manifest["run_id"])
        enriched=[]
        for i,r in enumerate(result):
            x=dict(r,model_id=cell["model_id"],track=cell["track"],
                   stratum=cell["stratum"],episode=i)
            enriched.append(x);all_rows.append(x)
        st=_model_stats(enriched)
        st.update(cell_index=cell["index"],model_id=cell["model_id"],
                  track=cell["track"],stratum=cell["stratum"],
                  unattempted=60-len(result)-(pending is not None),
                  inflight=pending is not None,
                  event_log_sha256=hash_file(path) if path.exists() else None)
        per_cell.append(st)
        cell_rows[(cell["model_id"],cell["track"],cell["stratum"])]=enriched
    complete=(len(per_cell)==96 and all(x["valid"]==60 and x["invalid"]==0
                                        and not x["inflight"] for x in per_cell))
    if complete != snapshot["all_cells_complete"]:
        raise ValueError("completeness drift in evaluator snapshots")
    by_model={}
    by_candidate={}
    for model in sorted(manifest["original_final_model_sha256"]):
        rows=[r for r in all_rows if r["model_id"]==model]
        by_model[model]=_model_stats(rows)
    for candidate in "ABCD":
        blocks=[by_model[f"{candidate}-seed{i}"] for i in range(3)]
        if complete:
            means=[x["finish_rate_unaccepted"] for x in blocks]
            by_candidate[candidate]={
                "per_training_seed_finish_rates":means,
                "mean_seed_finish_rate":statistics.mean(means),
                "sample_seed_sd":statistics.stdev(means),
                "min_seed_rate":min(means),"max_seed_rate":max(means)}
        else:
            by_candidate[candidate]={"status":"WITHHELD_INCOMPLETE_96_CELL_EVAL",
                "observed_only_sample_size":sum(x["valid"] for x in blocks)}
    all_outcomes=dict(Counter(x["outcome"] for x in all_rows))
    result={
        "schema_version":"wp3_owner_traindev_evaluation_report_v1",
        "run_id":manifest["run_id"],"execution_source":manifest["source_head"],
        "training_source":manifest["training_source"],
        "protocol":manifest["protocol"],"physics":manifest["physics_version"],
        "manifest_sha256":manifest["sha256"],
        "status":"COMPLETE_96_CELLS_TRAIN_DEVELOPMENT_NOT_SCIENTIFICALLY_SIGNED" if complete
                 else "INCOMPLETE_RANKING_WITHHELD",
        "canonical_expected_cells":96,"canonical_expected_episodes":5760,
        "complete_cells":snapshot["complete_cells"],
        "valid_episodes":snapshot["valid_episodes"],
        "invalid_episodes":snapshot["invalid_episodes"],
        "unattempted_slots":snapshot["unattempted_slots"],
        "per_cell":per_cell,"by_model":by_model,
        "candidate_descriptive_only":by_candidate,"outcomes":all_outcomes,
        "scientific_evidence":False,"independent_physics_signoff":None,
        "heldout_access":False,
        "n_independent_training_seeds_per_candidate":3,
        "causal_claims":"Not established; curriculum corridor and reward treatment vary by candidate, confounded without additional experimental control."}
    result["sha256"]=digest(result)
    _json_exclusive(out/"WP3-TRAIN-EVALUATION.json",result)
    comparison=["# WP3 Candidate comparison — TRAIN-development","",
       f"Run: {manifest['run_id']} | Source: {manifest['source_head']}",
       "NO independently signed scientific Physics v4 certification.",
       f"Status: **{result['status']}**.","",
       "| Model | Valid/480 | Finishes | Observed finish rate |",
       "|---|---:|---:|---:|"]
    for mid,x in by_model.items():
        rate=f"{100*x['finish_rate_unaccepted']:.2f}%" if x["finish_rate_unaccepted"] is not None else "—"
        comparison.append(f"| {mid} | {x['valid']}/480 | {x['finishes']} | {rate} |")
    comparison+=["","## Four candidate summaries",""]
    if complete:
        comparison+=["| Candidate | 3 seed finish rates | Mean | SD (n=3) | Range |",
                     "|---|---|---:|---:|---|"]
        for k,b in by_candidate.items():
            rates=", ".join(f"{x:.2%}" for x in b["per_training_seed_finish_rates"])
            comparison.append(f"| {k} | {rates} | {b['mean_seed_finish_rate']:.2%} | {b['sample_seed_sd']:.2%} | {b['min_seed_rate']:.2%}–{b['max_seed_rate']:.2%} |")
    else:
        comparison.append("WITHHELD: all 96 cells must each contain 60 valid episodes first.")
    comparison.append("\nNo confirmatory superiority claim with 3 independent PPO training seeds.")
    _exclusive(out/"WP3-CANDIDATE-COMPARISON.md","\n".join(comparison)+"\n")
    _exclusive(out/"WP3-STATISTICAL-READINESS.md",
       "# WP3 Statistical Readiness\n\n"
       f"Status: **{result['status']}**. Valid: {snapshot['valid_episodes']}/5,760; "
       f"invalid {snapshot['invalid_episodes']}; "
       f"unattempted {snapshot['unattempted_slots']}.\n\n"
       "There are only 3 independently trained PPO seeds per candidate; 60 evaluation episodes per cell "
       "are conditional repeats, not 60 independent training replicates. No confirmatory statistical "
       "significance, causal reward-penalty attribution, or scientific robustness claims are licensed. "
       "HELD-OUT is sealed and no new training was performed.\n")
    _exclusive(out/"WP3-FINAL-STATUS.md",
       "# WP3 TRAIN Development Eval Status\n\n"
       f"Run: {manifest['run_id']}; execution source: {manifest['source_head']}.\n\n"
       f"**{result['status']}**. {snapshot['complete_cells']}/96 cells complete; "
       f"{snapshot['valid_episodes']} valid, {snapshot['invalid_episodes']} invalid, "
       f"{snapshot['unattempted_slots']} unattempted; "
       f"{snapshot['inflight_episodes']} started but unclosed.\n\n"
       "Owner-authorization is recorded; this is distinct from scientific reviewer "
       "approval. Physics domain independent signoff: PENDING. Scientific registered "
       "Evaluation remains BLOCKED. Historical evidence, TRAIN source and HELD-OUT remain unmodified.\n")
    candidate_bars=([(k,v["mean_seed_finish_rate"]) for k,v in by_candidate.items()]
                    if complete else [])
    _exclusive(out/"WP3-MODEL-COMPARISON.svg",
       _bars("Candidate finish rates (mean across 3 PPO seeds)",candidate_bars,
             "TRAIN development only; no independently certified scientific model ranking."))
    _exclusive(out/"WP3-FAILURE-DISTRIBUTION.svg",
               _failures_svg(all_outcomes,complete))
    telemetry={m["model_id"]:m["training_telemetry_sha256"] for m in training_provenance["models"]}
    _exclusive(out/"WP3-LEARNING-PROGRESS.svg",
        _learning_progress_svg(history,list(manifest["original_final_model_sha256"]),telemetry))
    produced=sorted(p for p in out.iterdir() if p.name.startswith("WP3-") and p.is_file())
    sha_manifest={"run_id":manifest["run_id"],"source_sha":manifest["source_head"],
                  "files":[{"name":p.name,"sha256":hash_file(p)} for p in produced],
                  "cell_logs":[{"name":x["event_log_sha256"],"cell":x["cell_index"]}
                               for x in per_cell]}
    sha_manifest["sha256"]=digest(sha_manifest)
    _json_exclusive(out/"WP3-OUTPUT-SHA256-MANIFEST.json",sha_manifest)
    return result
