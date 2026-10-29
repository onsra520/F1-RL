"""Deterministic WP3 evaluation reports and charts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ._artifacts import _write_new_or_match
from ._evaluation_events import _read_events


def _svg_header(title: str, width: int = 1000, height: int = 560) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" font-weight="bold">{title}</text>']


def _svg_text(lines: list[str], x: float, y: float, value: str, *, size: int = 13,
              anchor: str = "middle", color: str = "#222") -> None:
    value = (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    lines.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{value}</text>')


def _write_learning_chart(campaign_root: Path, schedule: list[dict], evaluation_result: dict) -> None:
    grouped: dict[str, list[list[float]]] = {c: [] for c in ("A", "B", "C", "D")}
    run_root = campaign_root / "runs"
    for model in schedule:
        path = run_root / model["model_id"] / "training-telemetry.jsonl"
        losses = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event_type") == "ppo_update":
                    value = event.get("metrics", {}).get("train/loss")
                    losses.append(float(value) if value is not None else math.nan)
        grouped[model["candidate_id"]].append(losses)
    width, height = 1000, 560
    left, right, top, bottom = 80, 940, 70, 470
    finite_values = [value for rows in grouped.values() for row in rows for value in row if math.isfinite(value)]
    low = min(finite_values, default=0.0)
    high = max(finite_values, default=1.0)
    if high <= low:
        high = low + 1.0
    palette = {"A": "#2455a4", "B": "#cf601b", "C": "#18845d", "D": "#a53363"}
    lines = _svg_header("WP3 PPO Training Loss by Candidate (mean across 3 seeds)", width, height)
    for tick in range(6):
        y = bottom - (bottom-top)*tick/5
        value = low + (high-low)*tick/5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>')
        _svg_text(lines, left-8, y+4, f"{value:.3g}", anchor="end", size=11)
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>')
    _svg_text(lines, 28, (top+bottom)/2, "Training loss", size=12)
    _svg_text(lines, (left+right)/2, 525, "PPO rollout update", size=12)
    for candidate, rows in grouped.items():
        length = min((len(row) for row in rows), default=0)
        values = [sum(row[i] for row in rows if math.isfinite(row[i]))/
                  sum(math.isfinite(row[i]) for row in rows) for i in range(length)]
        points = []
        for i, value in enumerate(values):
            x = left + (right-left)*i/max(1, length-1)
            y = bottom - (bottom-top)*(value-low)/(high-low)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{palette[candidate]}" stroke-width="2" points="{" ".join(points)}"/>')
        _svg_text(lines, right-8, top+18+20*("ABCD".index(candidate)), candidate,
                  anchor="end", size=13, color=palette[candidate])
    lines.append("</svg>")
    _write_new_or_match(campaign_root / "WP3-LEARNING-PROGRESS.svg", "\n".join(lines)+"\n")


def _write_comparison_chart(campaign_root: Path, comparison: dict) -> None:
    summary = comparison["candidate_finish_rate_summary"]
    candidates = ["A", "B", "C", "D"]
    width, height = 900, 520
    left, right, top, bottom = 90, 840, 70, 430
    lines = _svg_header("TRAIN Valid Autonomous Finish Rate by Candidate", width, height)
    for tick in range(6):
        rate = tick/5
        y = bottom-(bottom-top)*rate
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>')
        _svg_text(lines, left-8, y+4, f"{rate:.0%}", anchor="end", size=11)
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>')
    palette = {"A": "#2455a4", "B": "#cf601b", "C": "#18845d", "D": "#a53363"}
    for index, candidate in enumerate(candidates):
        x = left+90+index*170
        value = summary[candidate]["finish_rate_mean_across_seed_blocks"]
        if value is not None:
            y = bottom-(bottom-top)*value
            lines.append(f'<rect x="{x-24}" y="{y:.1f}" width="48" height="{bottom-y:.1f}" fill="{palette[candidate]}" opacity="0.8"/>')
            sd = summary[candidate]["finish_rate_seed_block_sd"] or 0.0
            y1=bottom-(bottom-top)*min(1.0,value+sd); y2=bottom-(bottom-top)*max(0.0,value-sd)
            lines.append(f'<line x1="{x}" y1="{y1:.1f}" x2="{x}" y2="{y2:.1f}" stroke="#111" stroke-width="2"/>')
            for seed_index, seed_rate in summary[candidate]["seed_block_finish_rates"].items():
                if seed_rate is not None:
                    sy=bottom-(bottom-top)*seed_rate
                    lines.append(f'<circle cx="{x+(int(seed_index)-1)*10}" cy="{sy:.1f}" r="4" fill="#111"/>')
        _svg_text(lines,x,bottom+24,candidate,size=14)
    _svg_text(lines,(left+right)/2,490,"Mean across 3 PPO seed blocks; vertical whiskers show seed-block SD; dots show blocks",size=12)
    lines.append("</svg>")
    _write_new_or_match(campaign_root / "WP3-MODEL-COMPARISON.svg", "\n".join(lines)+"\n")


def _write_failure_chart(campaign_root: Path, cells: list[dict]) -> None:
    categories = ["valid_finish", "hard_route_violation", "timeout", "collision", "other"]
    candidates = ["A", "B", "C", "D"]
    counts = {candidate: {category: 0 for category in categories} for candidate in candidates}
    for cell in cells:
        event_path = (campaign_root / "evaluation" / "cells" / cell["model_id"] /
                      f"{cell['track_id']}-{cell['stratum']}.jsonl")
        for event in _read_events(event_path):
            if event["event"] != "episode_completed" or not event["record"].get("valid"):
                continue
            item = event["record"]["metrics"]
            if item["success"]:
                category="valid_finish"
            elif item["hard_route_violation"]:
                category="hard_route_violation"
            elif item["timeout"]:
                category="timeout"
            elif item["collision_count"] > 0:
                category="collision"
            else:
                category="other"
            counts[cell["candidate_id"]][category]+=1
    colors={"valid_finish":"#18845d","hard_route_violation":"#b53030","timeout":"#cf601b","collision":"#6e3ca0","other":"#8b969e"}
    width,height=1000,560; top,bottom=70,430
    lines=_svg_header("TRAIN Evaluation Outcomes by Candidate",width,height)
    maximum=max((sum(counts[c].values()) for c in candidates),default=1) or 1
    for i,candidate in enumerate(candidates):
        x=190+i*190; baseline=bottom
        for category in categories:
            value=counts[candidate][category]
            h=(bottom-top)*value/maximum
            baseline-=h
            lines.append(f'<rect x="{x-45}" y="{baseline:.1f}" width="90" height="{h:.1f}" fill="{colors[category]}"/>')
        _svg_text(lines,x,bottom+22,candidate,size=14)
    for i,category in enumerate(categories):
        x=130+(i%3)*270; y=485+(i//3)*22
        lines.append(f'<rect x="{x}" y="{y-12}" width="14" height="14" fill="{colors[category]}"/>')
        _svg_text(lines,x+22,y,category,anchor="start",size=12)
    lines.append("</svg>")
    _write_new_or_match(campaign_root / "WP3-FAILURE-DISTRIBUTION.svg", "\n".join(lines)+"\n")


def _write_evaluation_reports(campaign_root: Path, result: dict,
                              models: list[dict], cells: list[dict], evaluation: dict) -> None:
    comparison = result["comparison"]
    lines=[
        "# WP3 A/B/C/D TRAIN Candidate Comparison", "",
        f"Evaluation status: **{result['status']}**. Scope is simulation-only development; no scientific acceptance claim.",
        "", "## Candidate finish rates", "",
        "| Candidate | Seed 0 | Seed 1 | Seed 2 | Mean across seed blocks | Seed-block SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summaries=comparison["candidate_finish_rate_summary"]
    for candidate in ("A","B","C","D"):
        row=summaries[candidate]
        values=[row["seed_block_finish_rates"][i] for i in range(3)]
        fmt=lambda value: "incomplete" if value is None else f"{value:.3f}"
        lines.append(f"| {candidate} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {fmt(row['finish_rate_mean_across_seed_blocks'])} | {fmt(row['finish_rate_seed_block_sd'])} |")
    lines += ["", "## Per-track finish-rate matrix", "", "| Candidate | " + " | ".join(evaluation["tracks"]) + " |", "|---|"+"|".join(["---:"]*4)+"|"]
    for candidate in ("A","B","C","D"):
        values=[comparison["per_track_finish_rate"][candidate][track]["finish_rate"] for track in evaluation["tracks"]]
        lines.append("| "+candidate+" | "+" | ".join("incomplete" if v is None else f"{v:.3f}" for v in values)+" |")
    lines += ["", "## Stratum breakdown", "", "| Candidate | Nominal | Randomized TRAIN |", "|---|---:|---:|"]
    for candidate in ("A","B","C","D"):
        values=[comparison["per_stratum_finish_rate"][candidate][stratum]["finish_rate"] for stratum in evaluation["strata"]]
        lines.append(f"| {candidate} | {'incomplete' if values[0] is None else f'{values[0]:.3f}'} | {'incomplete' if values[1] is None else f'{values[1]:.3f}'} |")
    lines += ["", "## Secondary development outcomes", "", "| Candidate | Mean route progress | Mean checkpoints | Collision rate | Hard-route violation rate | Timeout rate | Steering-rate RMS | Mean duration (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for candidate in ("A","B","C","D"):
        row=summaries[candidate]
        keys=("mean_route_progress","mean_completed_checkpoints","collision_rate","hard_route_violation_rate","timeout_rate","mean_steering_rate_rms_rad_per_s","mean_episode_duration_seconds")
        values=[row.get(key) for key in keys]
        lines.append("| "+candidate+" | "+" | ".join("n/a" if x is None else f"{x:.3f}" for x in values)+" |")
    effects = comparison["factor_effects"]
    lines += ["", "## Factor contrasts", "",
              "| Contrast | Mean across PPO seeds | Seed-block SD | Per-seed differences |",
              "|---|---:|---:|---|"]
    for name in ("higher_lateral_penalty_minus_lower", "3m_corridor_minus_2m_corridor", "factorial_interaction"):
        summary = effects.get("mean_and_seed_sd", {}).get(name)
        values = effects.get("per_seed_block", {}).get(name)
        if summary is None or values is None:
            lines.append(f"| {name} | incomplete | incomplete | incomplete |")
        else:
            lines.append(f"| {name} | {summary['mean']:.4f} | {summary['seed_block_sd']:.4f} | "
                         + ", ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "## Failure-reason counts", ""]
    for candidate in ("A", "B", "C", "D"):
        reasons = summaries[candidate].get("failure_reason_counts", {})
        lines.append(f"- **{candidate}:** " + (", ".join(f"{reason}={count}" for reason, count in reasons.items()) or "incomplete cells"))
    lines += ["", "Factor contrasts are descriptive paired differences across the three seed blocks; they are not hypothesis tests.",
              f"Best descriptive candidate: **{comparison['best_development_candidate'] or 'none (incomplete cells)'}**. Apparent differences are uncertain with three tuning seed blocks.",
              "", "Charts: `WP3-MODEL-COMPARISON.svg`, `WP3-FAILURE-DISTRIBUTION.svg`, and `WP3-LEARNING-PROGRESS.svg`.",
              "", "WP7 and Gate 2 remain **BLOCKED**.",""]
    _write_new_or_match(campaign_root/"WP3-CANDIDATE-COMPARISON.md","\n".join(lines))
    stats=("# WP3 Statistical Readiness\n\n"
           "This is a tuning/development experiment with three independent PPO seed blocks per candidate. Candidate and factor differences are reported descriptively; no confirmatory p-values, significance claims, or WP7 robustness claims are made. The campaign has 60 episodes in each of 8 TRAIN track/stratum cells per model, but these repeated episodes do not replace independent training seed blocks.\n\n"
           "The proposal's 24 confirmation seed blocks, MDE 0.15, paired SD 0.20, target power 0.80, and multiplicity plan remain unapproved planning assumptions. These results may inform a later power-planning proposal, but do not authorize confirmation training. Report cross-seed means, sample SDs, and ranges; preserve per-seed and per-track/stratum cells. If any cell is invalid, do not redraw it and do not rank incomplete candidates.\n\n"
           f"Current evaluation status: **{result['status']}**. Invalid cells: {len(result['invalid_cells'])}.\n\n"
           "No inference here grants scientific acceptance or changes WP7/Gate 2, both of which remain **BLOCKED**.\n")
    _write_new_or_match(campaign_root/"WP3-STATISTICAL-READINESS.md",stats)
    _write_learning_chart(campaign_root,[{"model_id":r["model_id"],"candidate_id":r["candidate_id"],"run_relative_path":f"runs/{r['model_id']}"} for r in models],result)
    _write_comparison_chart(campaign_root,comparison)
    _write_failure_chart(campaign_root,cells)
    final=("# WP3 Final Status\n\n"
           f"**Training: {len(models)}/12 engineering-validated models. Evaluation: {result['status']}.**\n\n"
           f"Actual PPO training timesteps: {len(models)*65536:,} of 786,432. Evaluation attempted {result['total_valid_episode_rows']:,} valid episodes across {result['completed_cells']}/96 complete cells and {len(result['invalid_cells'])} invalid cells.\n\n"
           f"Best descriptive development candidate: {comparison['best_development_candidate'] or 'not ranked because cells are incomplete'}. With three tuning seed blocks, this is uncertain and not definitive. The registered 4 m hard-route boundary and final-checkpoint-only evaluation were preserved.\n\n"
           "No HELD-OUT access, scientific confirmation, or scientific acceptance evaluation occurred. WP7 and Gate 2 remain **BLOCKED**. See the candidate comparison and statistical-readiness report for per-seed outcomes and limits.\n")
    _write_new_or_match(campaign_root/"WP3-FINAL-STATUS.md",final)
