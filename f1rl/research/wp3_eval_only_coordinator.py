"""Standalone WP3 Eval-only coordinator. New engineering evidence; scientific full Eval fail-closed.

The frozen 12 final PPO ZIPs, paired original roster and historical evaluation
are read-only. Never resumes the original campaign; never accesses HELD-OUT.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable
import csv
import json
import os
import subprocess
import time

from .wp3_eval_only_admission import (
    REPLAY_CASES, TRACKS, STRATA, MODELS, _json, _roster_check,
    hash_file, digest, build_eval_only_preflight, refuse_full_eval,
)
from ..learning.crash_terminal_registered import CRASH_V1, PHYSICS_V4, RACE_V4

FULL06_RUNTIME_PROFILE = "wp3_full06_physics_v4"
FULL06_RUNTIME_PROFILE_SPEC = {
    "physics_version": PHYSICS_V4,
    "race_version": RACE_V4,
    "terminal_contract": CRASH_V1,
    "max_abs_tilt_rad": .7,
}
RUNTIME_PROFILE_SPECS = {FULL06_RUNTIME_PROFILE: FULL06_RUNTIME_PROFILE_SPEC}
PHYSICS = PHYSICS_V4
RACE = RACE_V4
CONTRACT = CRASH_V1
TRAIN_HEAD = "fe83f517479f3b990028d6409304fae83b8199a9"
DEFAULT_MODELS = ("A-seed1", "C-seed0")
SCHEMA = "wp3_eval_only_engineering_coordinator_v1"


class RuntimeProfileMismatch(ValueError):
    pass


def runtime_profile_spec(requested_profile: str) -> dict:
    if not isinstance(requested_profile, str) or requested_profile not in RUNTIME_PROFILE_SPECS:
        raise ValueError(f"unsupported smoke runtime profile: {requested_profile!r}")
    return {"profile": requested_profile, **RUNTIME_PROFILE_SPECS[requested_profile]}


def effective_runtime_profile(factory) -> dict:
    physics = factory.physics_config
    race = factory.race_config
    suspension = None if physics is None else physics.independent_suspension_policy
    return {
        "physics_version": None if physics is None else physics.ground_motion_version,
        "race_version": None if race is None else race.version,
        "terminal_contract": CONTRACT if race is not None and race.version == RACE else None,
        "max_abs_tilt_rad": None if suspension is None else suspension.max_abs_tilt_rad,
    }


def validate_runtime_profile(factory, requested_profile: str) -> dict:
    expected = runtime_profile_spec(requested_profile)
    effective = effective_runtime_profile(factory)
    expected_effective = {key: value for key, value in expected.items() if key != "profile"}
    if effective != expected_effective:
        raise RuntimeProfileMismatch(
            f"runtime profile mismatch: requested={requested_profile!r} "
            f"expected={expected_effective!r} effective={effective!r}")
    return {"requested_profile": requested_profile, "effective": effective}


def apply_runtime_profile(factory, requested_profile: str):
    expected = runtime_profile_spec(requested_profile)
    suspension = replace(factory.physics_config.independent_suspension_policy,
                         max_abs_tilt_rad=expected["max_abs_tilt_rad"])
    factory = replace(factory,
        physics_config=replace(factory.physics_config,
            ground_motion_version=expected["physics_version"],
            independent_suspension_policy=suspension),
        race_config=replace(factory.race_config, version=expected["race_version"]))
    provenance = validate_runtime_profile(factory, requested_profile)
    return factory, provenance


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   text=True, stderr=subprocess.PIPE).strip()


def _source(root: Path) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Eval-only requires a clean, committed source tree")
    if subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                       "ce0e55d7a428826b2d4ed3218641105bd0206f4b", head],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode:
        raise ValueError("Eval-only source is not a descendant of Issue #51 handoff")
    return head


def _write_new(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf8", newline="\n") as dest:
        json.dump(obj, dest, ensure_ascii=False, indent=2, allow_nan=False)
        dest.write("\n")
        dest.flush()
        os.fsync(dest.fileno())


def _event(file, entry: dict) -> None:
    file.write(json.dumps(entry, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def roster_cases(roster: dict, model_ids: tuple[str, ...] = DEFAULT_MODELS,
                 episode_index: int = 4) -> list[dict]:
    """Deterministic small diagnostic: all 4 TRAIN tracks x 2 strata x N models.

    Not a shortcut for the full 5,760-slot registered evaluation.
    """
    if (not model_ids or len(model_ids) != len(set(model_ids))
            or any(m not in MODELS for m in model_ids)
            or type(episode_index) is not int or not 0 <= episode_index < 60):
        raise ValueError("model IDs or original roster episode index invalid")
    cells = {(r["model_id"], r["track_id"], r["stratum"]): r
             for r in roster["seed_rows"]}
    if len(cells) != 96:
        raise ValueError("original frozen 96-cell roster missing/duplicate")
    cases = []
    for track in TRACKS:
        for stratum in STRATA:
            for mid in model_ids:
                cell = cells[(mid, track, stratum)]
                seeds = cell["episodes"][episode_index]
                cases.append({"model_id": mid, "track": track, "stratum": stratum,
                              "episode": episode_index,
                              "seed_index": cell["seed_index"],
                              "env_seed": seeds["environment_seed"],
                              "policy_seed": seeds["policy_seed"]})
    for track in TRACKS:
        for stratum in STRATA:
            matched = [c for c in cases if (c["track"], c["stratum"]) == (track, stratum)]
            if len({(c["env_seed"], c["policy_seed"]) for c in matched}) != 1:
                raise ValueError("candidate-independent paired seed mismatch")
    return cases


def preflight(root: Path, history: Path) -> tuple[dict, dict, dict]:
    root = Path(root).resolve(strict=True)
    history = Path(history).resolve(strict=True)
    source = _source(root)
    old_replay = root / "docs/research/WP3-CE0E55D-THREE-REPLAY-PREFLIGHT-20261024.json"
    check = build_eval_only_preflight(root, history, old_replay)
    if check["source_dirty"] or check["frozen_final_checkpoint_count"] != 12:
        raise ValueError("immutable TRAIN model preflight not clean")
    roster = _json(history / "evaluation/evaluation-roster.json")
    provenance = _json(history / "WP3-EVAL-SOURCE-PROVENANCE.json")
    _roster_check(roster, provenance)
    if provenance["training_source_commit"] != TRAIN_HEAD:
        raise ValueError("frozen training source mismatch")
    # This is a NEW derived preflight, not the historic receipt. Bind HEAD and
    # recalculate its canonical digest before invoking the fail-closed gate.
    check["current_evaluation_source_commit"] = source
    check["sha256"] = digest({k: v for k, v in check.items() if k != "sha256"})
    return check, roster, provenance


def forbid_registered_full_eval(root: Path, history: Path) -> None:
    """No simulated reviewer signature, permissive override or implicit admission."""
    check, _, _ = preflight(root, history)
    refuse_full_eval(check)
    raise AssertionError("registered 5,760 Evaluation was not blocked")


def _production_episode(root: Path, history: Path, case: dict,
                        cache: dict, runtime_profile: str = FULL06_RUNTIME_PROFILE) -> dict:
    from ..learning.algorithms.ppo import PPOModel
    from ..local_demo.evaluation import _evaluate_episode
    from .wp7_protocol import build_protocol, prepare_evaluation_factory

    model_id = case["model_id"]
    if model_id not in cache:
        model = PPOModel.load(history / "runs" / model_id / "model-final.zip", device="cpu")
        if model.num_timesteps != 65536:
            raise ValueError("wrong final PPO timestep count")
        cache[model_id] = model
    if "protocol" not in cache:
        cache["protocol"] = build_protocol(root, profile="spatial_rotating_owner_v3")
    factory, _ = prepare_evaluation_factory(
        root, cache["protocol"], f"multitrack-seed{case['seed_index']}",
        case["track"], case["stratum"], 0)
    if case["stratum"] == "randomized_train":
        if factory.reset_config is None:
            raise ValueError("randomized TRAIN reset config missing")
        factory = replace(factory, reset_config=replace(
            factory.reset_config, seed=case["env_seed"]))
    factory, runtime_provenance = apply_runtime_profile(factory, runtime_profile)
    if factory.race_config.hard_corridor_m != 4:
        raise ValueError("frozen original 4m boundary changed")
    metrics = _evaluate_episode(cache[model_id], factory, case["track"],
                                case["env_seed"], case["policy_seed"],
                                factory.simulation_config.fixed_dt)
    if (metrics.get("deterministic_prediction") is not False
            or metrics.get("environment_seed") != case["env_seed"]
            or metrics.get("policy_seed") != case["policy_seed"]):
        raise ValueError("stochastic paired-seed evaluation contract drifted")
    metrics["runtime_profile"] = runtime_provenance
    return metrics


def _describe(row: dict) -> str:
    if row["status"] != "ENGINEERING_EPISODE_COMPLETED":
        return "INVALID_RUNTIME"
    if row["success_unaccepted"] is True:
        return "finish"
    return row.get("failure_reason") or "terminal_other"


def _write_report(out: Path, report: dict) -> None:
    lines = [
        "# WP3 Issue #51 | New engineering Eval-only evidence",
        "",
        f"Run ID: `{report['run_id']}`",
        f"Source HEAD: `{report['evaluation_source_commit']}`",
        f"TRAIN SHA: `{TRAIN_HEAD}`",
        f"Roster SHA-256: `{report['roster_sha256']}`",
        f"Requested runtime profile: `{report['requested_runtime_profile']}`",
        f"Expected effective runtime: `{json.dumps(report['expected_effective_runtime'], sort_keys=True)}`",
        "",
        "**UNREGISTERED ENGINEERING ONLY — full 5,760-slot scientific Eval BLOCKED.**",
        "No training, no HELD-OUT, no modification of historical Eval.",
        "",
        f"Completed {report['completed']}/{report['planned']}; invalid {report['invalid']}.",
        "",
        "| Model | Completed | Finish | Physical crash | Step limit | Invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, x in report["by_model"].items():
        lines.append(f"| {model} | {x['completed']} | {x['finish']} | "
                     f"{x['physical_crash']} | {x['step_limit']} | {x['invalid']} |")
    lines += ["", "## Per-episode results", "",
              "| Model | Track | Stratum | Episode | Outcome | Progress* |",
              "|---|---|---|---:|---|---:|"]
    for x in report["episodes"]:
        lines.append(f"| {x['model_id']} | {x['track']} | {x['stratum']} | "
                     f"{x['episode']} | {x['outcome']} | {x.get('progress_not_certified', '—')} |")
    lines += ["", "*Uncertified engineering telemetry, not scientific performance.",
              "Only a small number of seeds were sampled; no candidate rankings allowed.",
              "New independently reviewed physics certification and a separately registered"
              " source/protocol are mandatory before full Eval."]
    with (out / "REPORT.md").open("x", encoding="utf8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")


def run_engineering(root: Path, history: Path, out: Path, run_id: str,
                    *, episode_index: int = 4,
                    model_ids: tuple[str, ...] = DEFAULT_MODELS,
                    runtime_profile: str = FULL06_RUNTIME_PROFILE,
                    evaluate: Callable | None = None) -> dict:
    """Run an immutable, new and bounded engineering diagnostic (not full Eval).

    All events are append-only. Interrupted runs are left interrupted; reusing
    a run ID/output is rejected instead of silently replaying failed episodes.
    """
    root, history, out = Path(root).resolve(), Path(history).resolve(), Path(out).resolve()
    requested_runtime_profile = runtime_profile_spec(runtime_profile)
    if (not run_id.startswith("WP3-ENG-") or len(run_id) > 100
            or not all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in run_id)):
        raise ValueError("explicit distinct engineering-only run ID required")
    check, roster, prov = preflight(root, history)
    try:
        refuse_full_eval(check)
    except RuntimeError as exc:
        if not str(exc).startswith("FULL_EVAL_BLOCKED"):
            raise
    else:
        raise AssertionError("unauthorized full evaluation gate opened")
    cases = roster_cases(roster, model_ids, episode_index)
    # No 5,760 episode override. This path is expressly limited to small diagnostics.
    if len(cases) > 32:
        raise ValueError("engineering-only plan too large; do not bypass registration gate")
    provenance_by_model = {m["model_id"]: m for m in prov["models"]}
    hashes = {
        str(history / "evaluation/evaluation-roster.json"):
            hash_file(history / "evaluation/evaluation-roster.json"),
        str(history / "WP3-EVAL-SOURCE-PROVENANCE.json"):
            hash_file(history / "WP3-EVAL-SOURCE-PROVENANCE.json"),
    }
    for m in prov["models"]:
        path = history / "runs" / m["model_id"] / "model-final.zip"
        sha = hash_file(path)
        if sha != m["model_final_sha256"]:
            raise ValueError("one frozen PPO checkpoint changed")
        hashes[str(path)] = sha
    if out.exists():
        raise FileExistsError(f"never overwrite or resume prior evidence: {out}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "cells").mkdir()
    manifest = {
        "schema_version": SCHEMA, "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_source_commit": check["current_evaluation_source_commit"],
        "original_training_source_commit": TRAIN_HEAD,
        "physics_profile": PHYSICS, "crash_terminal_contract": CONTRACT,
        "race_version": RACE,
        "requested_runtime_profile": runtime_profile,
        "expected_effective_runtime": {
            key: value for key, value in requested_runtime_profile.items() if key != "profile"},
        "roster_sha256": roster["sha256"], "preflight_sha256": check["sha256"],
        "original_expected_cells": 96, "original_expected_episodes": 5760,
        "diagnostic_planned_cases": len(cases),
        "episode_index": episode_index,
        "original_immutable_file_sha256": hashes,
        "inference": "CPU/stochastic/deterministic=False",
        "scientific_evidence": False, "registered_evaluation": False,
        "full_evaluation_admitted": False, "heldout_access": False,
        "no_retraining": True, "cases": cases,
    }
    manifest["sha256"] = digest(manifest)
    _write_new(out / "RUN-MANIFEST.json", manifest)
    rows = []
    cache = {}
    for case in cases:
        index = len(rows)
        tag = f"{index:02d}-{case['model_id']}-{case['track']}-{case['stratum']}"
        path = out / "cells" / (tag + ".jsonl")
        row = dict(case)
        row.update({"status": "ENGINEERING_EPISODE_INVALID",
                    "scientific_evidence": False,
                    "registered_evaluation": False})
        with path.open("x", encoding="utf8", newline="\n") as stream:
            _event(stream, {"event": "episode_started", "case": case})
            began = time.monotonic()
            try:
                data = (evaluate(root, history, case, cache) if evaluate is not None
                        else _production_episode(root, history, case, cache, runtime_profile))
                runtime_provenance = data.get("runtime_profile")
                if runtime_provenance is None and evaluate is None:
                    raise RuntimeProfileMismatch(
                        "production episode omitted validated runtime profile provenance")
                if runtime_provenance is None:
                    runtime_provenance = {
                        "requested_profile": runtime_profile,
                        "effective": None,
                        "validation": "not_available_from_injected_evaluator",
                    }
                row.update({
                    "status": "ENGINEERING_EPISODE_COMPLETED",
                    "success_unaccepted": bool(data["success"]),
                    "failure_reason": data.get("failure_reason"),
                    "progress_not_certified": data.get("final_normalized_route_progress"),
                    "episode_time_s": data.get("episode_time"),
                    "completed_checkpoints": data.get("completed_checkpoints"),
                    "collision_count": data.get("collision_count"),
                    "hard_route_violation": data.get("hard_route_violation"),
                    "timeout": data.get("timeout"),
                    "runtime_profile": runtime_provenance,
                })
            except RuntimeProfileMismatch:
                raise
            except Exception as error:
                row.update({"error_type": type(error).__name__, "error": str(error)})
            row["runtime_s"] = round(time.monotonic() - began, 4)
            row["outcome"] = _describe(row)
            _event(stream, {"event": ("episode_completed" if row["status"] ==
                                     "ENGINEERING_EPISODE_COMPLETED" else "episode_invalid"),
                            "record": row})
        rows.append(row)
        print(f"EVAL-ONLY {len(rows)}/{len(cases)} {row['model_id']} "
              f"{row['track']} {row['stratum']} {row['outcome']}", flush=True)
    for name, sha in hashes.items():
        if hash_file(Path(name)) != sha:
            raise RuntimeError(f"immutable original input changed: {name}")
    if _source(root) != manifest["evaluation_source_commit"]:
        raise RuntimeError("source modified during engineering run")
    by_model = {}
    for mid in model_ids:
        group = [r for r in rows if r["model_id"] == mid]
        counts = Counter(r["outcome"] for r in group)
        by_model[mid] = {"completed": sum(r["status"] == "ENGINEERING_EPISODE_COMPLETED"
                                          for r in group),
                         "invalid": counts["INVALID_RUNTIME"],
                         "finish": counts["finish"],
                         "physical_crash": counts["physical_crash"],
                         "step_limit": counts["step_limit"]}
    report = {
        "schema_version": SCHEMA, "run_id": run_id,
        "evaluation_source_commit": manifest["evaluation_source_commit"],
        "original_training_source_commit": TRAIN_HEAD,
        "requested_runtime_profile": runtime_profile,
        "expected_effective_runtime": {
            key: value for key, value in requested_runtime_profile.items() if key != "profile"},
        "roster_sha256": roster["sha256"],
        "plan_sha256": manifest["sha256"],
        "completed": sum(r["status"] == "ENGINEERING_EPISODE_COMPLETED" for r in rows),
        "planned": len(rows),
        "invalid": sum(r["status"] != "ENGINEERING_EPISODE_COMPLETED" for r in rows),
        "by_model": by_model,
        "episodes": rows,
        "full_5760_eval_gate": "BLOCKED",
        "scientific_evidence": False, "official_eval_started": False,
        "heldout_access": False,
        "per_cell_sha256": {p.name: hash_file(p) for p in sorted((out/"cells").glob("*.jsonl"))},
    }
    report["sha256"] = digest(report)
    _write_new(out / "WP3-ENGINEERING-EVALUATION.json", report)
    with (out / "episodes.csv").open("x", encoding="utf8", newline="") as f:
        fields = ["model_id", "track", "stratum", "episode", "env_seed",
                  "policy_seed", "outcome", "status", "success_unaccepted",
                  "failure_reason", "progress_not_certified", "episode_time_s",
                  "collision_count", "completed_checkpoints", "error_type"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_report(out, report)
    print(f"ENGINEERING_EVAL_DONE completed={report['completed']} "
          f"invalid={report['invalid']} full_eval=BLOCKED", flush=True)
    return report
