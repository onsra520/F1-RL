"""Separate simulation-only, TRAIN-only evaluation of WP3 final checkpoints."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

from src.evaluation.evaluator import evaluate_episode

from ..rl.algorithms.ppo import PPOModel
from ..validation.wp7_protocol import build_protocol, prepare_evaluation_factory
from ._artifacts import (
    _sha256,
    _strict_json,
    _write_new_json,
    _write_new_or_match,
)
from ._evaluation_events import _append_event, _read_events
from ._evaluation_reports import _write_evaluation_reports
from ._evaluation_roster import _digest, build_evaluation_roster

__all__ = ("build_evaluation_roster", "run_evaluation", "_digest",
           "_write_new_or_match", "_write_evaluation_reports", "_summarize")

def _cell_summary(model: dict, track_id: str, stratum: str, rows: list[dict], status: str) -> dict:
    valid = [row for row in rows if row.get("valid") is True]
    invalid = [row for row in rows if row.get("valid") is False]
    if status == "completed" and len(valid) != 60:
        raise ValueError("complete evaluation cell must contain exactly 60 valid episode records")
    metrics = [row["metrics"] for row in valid]
    def mean(key):
        return None if not metrics else sum(float(item[key]) for item in metrics) / len(metrics)
    successes = sum(bool(item["success"]) for item in metrics)
    return {
        "schema_version": "f1rl_wp3_train_eval_cell_v1", "model_id": model["model_id"],
        "candidate_id": model["candidate_id"], "seed_index": model["seed_index"],
        "track_id": track_id, "stratum": stratum, "status": status,
        "expected_episodes": 60, "valid_episodes": len(valid), "invalid_episodes": len(invalid),
        "success_count": successes,
        "valid_autonomous_finish_rate": successes / len(valid) if len(valid) == 60 else None,
        "mean_route_progress": mean("final_normalized_route_progress"),
        "mean_completed_checkpoints": mean("completed_checkpoints"),
        "collision_rate": (sum(item["collision_count"] > 0 for item in metrics) / len(metrics)
                           if len(metrics) == 60 else None),
        "mean_collision_count": mean("collision_count"),
        "hard_route_violation_rate": (sum(item["hard_route_violation"] for item in metrics) / len(metrics)
                                       if len(metrics) == 60 else None),
        "timeout_rate": (sum(item["timeout"] for item in metrics) / len(metrics)
                         if len(metrics) == 60 else None),
        "mean_steering_rate_rms_rad_per_s": mean("steering_rate_rms_rad_per_s"),
        "mean_episode_duration_seconds": mean("episode_time"),
        "failure_reasons": _failure_counts(metrics),
        "invalid_episode": invalid[0] if invalid else None,
        "scientific_evidence": False, "heldout_access": False,
    }


def _failure_counts(metrics: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in metrics:
        reason = "valid_finish" if item["success"] else item.get("failure_reason") or "unspecified_failure"
        result[reason] = result.get(reason, 0) + 1
    return dict(sorted(result.items()))


def _validate_prior_cell(path: Path, model: dict, track_id: str, stratum: str) -> dict | None:
    summary_path = path.with_suffix(".json")
    if not path.exists():
        if summary_path.exists():
            raise ValueError("evaluation summary exists without its append-only episode event log")
        return None
    rows = _read_events(path)
    starts = [row for row in rows if row["event"] == "cell_started"]
    if len(starts) != 1 or starts[0].get("model_id") != model["model_id"]:
        raise ValueError("existing evaluation cell log has an invalid identity")
    episodes = []
    active = None
    for row in rows:
        if row["event"] == "episode_started":
            if active is not None:
                raise ValueError("prior evaluation cell has an unmatched episode start")
            active = row["episode"]
        elif row["event"] in ("episode_completed", "episode_invalid"):
            if active != row.get("episode"):
                raise ValueError("prior evaluation cell episode order is corrupt")
            episodes.append(row["record"])
            active = None
    if active is not None:
        # Interrupted evaluation is an invalid cell. Never redraw or repeat it.
        invalid = {"episode": active, "valid": False, "error_type": "InterruptedEvaluation",
                   "error": "episode began but no complete result was durably recorded; no redraw"}
        episodes.append(invalid)
        _append_event(path, {"event": "episode_invalid", "episode": active, "record": invalid})
        active = None
    finished = [row for row in rows if row["event"] == "cell_finished"]
    if finished:
        summary = finished[-1]["summary"]
        if summary.get("sha256") != _digest({k: v for k, v in summary.items() if k != "sha256"}):
            raise ValueError("prior evaluation cell summary checksum mismatch")
        prefix = b"".join(path.read_bytes().splitlines(keepends=True)[:-1])
        if summary.get("event_log_sha256") and sha256(prefix).hexdigest() != summary["event_log_sha256"]:
            raise ValueError("prior evaluation cell event-log prefix checksum mismatch")
        if summary_path.exists():
            prior, _ = _strict_json(summary_path)
            if prior != summary:
                raise ValueError("prior evaluation cell summary differs from append-only event log")
        else:
            _write_new_json(summary_path, summary)
        return summary
    status = "completed" if len(episodes) == 60 and all(r.get("valid") for r in episodes) else "invalid_aborted"
    if status == "completed" or any(not r.get("valid") for r in episodes):
        summary = _cell_summary(model, track_id, stratum, episodes, status)
        summary["sha256"] = _digest(summary)
        _append_event(path, {"event": "cell_finished", "summary": summary})
        if not summary_path.exists():
            _write_new_json(summary_path, summary)
        return summary
    return None


def _evaluate_episode(model: PPOModel, factory, track_id: str, env_seed: int, policy_seed: int,
                      fixed_dt: float) -> dict:
    action_history: list[float] = []
    checkpoint_count = 0
    hard_route_violation = False
    final_reason = None
    demo_training_seen = False
    def observe(event: dict[str, object]) -> None:
        nonlocal checkpoint_count, hard_route_violation, final_reason, demo_training_seen
        action = np.asarray(event["action"], dtype=np.float64).reshape(-1)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("evaluation action violates the registered 3-component action contract")
        action_history.append(float(action[0]))
        info = event["info"]
        if not isinstance(info, dict):
            raise ValueError("evaluation step info must be a mapping")
        if "demo_training" in info:
            demo_training_seen = True
        checkpoint_count = max(checkpoint_count, int(event["next_checkpoint_index"]))
        outcome = info.get("research_outcome") or {}
        if isinstance(outcome, dict) and outcome.get("reason") == "hard_route_violation":
            hard_route_violation = True
        if event["terminated"] or event["truncated"]:
            if event["finished"]:
                final_reason = "valid_finish"
            elif info.get("truncation_reason"):
                final_reason = info["truncation_reason"]
            elif isinstance(outcome, dict) and outcome.get("reason"):
                final_reason = outcome["reason"]
            elif event["truncated"]:
                final_reason = "timeout"
            elif hard_route_violation:
                final_reason = "hard_route_violation"
            else:
                final_reason = "terminated"
    PPOModel.seed_policy_rng(policy_seed)
    episode = evaluate_episode(track_id=track_id, factory=factory, policy=model,
                               reset_seed=env_seed, deterministic=False,
                               step_observer=observe)
    if demo_training_seen:
        raise ValueError("TRAIN evaluation unexpectedly applied a training-time curriculum")
    if not episode.success and final_reason == "valid_finish":
        raise ValueError("evaluation final reason conflicts with finish result")
    fixed_dt = float(fixed_dt)
    rates = [(right - left) / fixed_dt for left, right in zip(action_history, action_history[1:])]
    steering_rate_rms = math.sqrt(sum(rate * rate for rate in rates) / len(rates)) if rates else 0.0
    return {
        **asdict(episode),
        "completed_checkpoints": checkpoint_count,
        "hard_route_violation": hard_route_violation,
        "timeout": bool(episode.truncated and not episode.success),
        "failure_reason": None if episode.success else final_reason,
        "steering_rate_rms_rad_per_s": steering_rate_rms,
        "environment_seed": env_seed, "policy_seed": policy_seed,
        "deterministic_prediction": False,
    }


def _evaluation_factories(root: Path, protocol: dict, seed_index: int,
                          tracks: list[str], strata: list[str]) -> dict:
    cache = {}
    policy_id = f"multitrack-seed{seed_index}"
    for track in tracks:
        for stratum in strata:
            factory, _ = prepare_evaluation_factory(root, protocol, policy_id, track, stratum, 0)
            if factory.race_config is None or factory.race_config.hard_corridor_m != 4.0:
                raise ValueError("TRAIN evaluator requires the unchanged registered 4 m race contract")
            if factory.reset_config is None and stratum != "nominal":
                raise ValueError("randomized TRAIN evaluation lacks a reset contract")
            if factory.reset_config is not None and factory.reset_config.seed is None:
                raise ValueError("randomized TRAIN evaluation reset seed must be explicit")
            cache[(track, stratum)] = factory
    return cache


def run_evaluation(root: Path, campaign_root: Path, campaign_manifest: dict,
                   config: dict, schedule: list[dict]) -> dict:
    evaluation = config["evaluation"]
    roster_path = campaign_root / "evaluation" / "evaluation-roster.json"
    roster, _ = _strict_json(roster_path)
    if roster.get("sha256") != _digest({k: v for k, v in roster.items() if k != "sha256"}):
        raise ValueError("frozen evaluation roster checksum mismatch")
    expected_roster = build_evaluation_roster(root, campaign_manifest, config)
    if roster.get("sha256") != expected_roster.get("sha256"):
        raise ValueError("evaluation roster differs from the pre-training frozen seed plan")
    cell_root = campaign_root / "evaluation" / "cells"
    cell_root.mkdir(exist_ok=True)
    protocol = build_protocol(root, profile="spatial_rotating_owner_v3")
    all_cells = []
    model_records = []
    evaluation_started = time.monotonic()
    events_path = campaign_root / "evaluation" / "evaluation-events.jsonl"
    for model in schedule:
        model_dir = campaign_root / model["run_relative_path"]
        training_validation, _ = _strict_json(model_dir / "engineering-validation.json")
        if training_validation.get("status") != "ENGINEERING_PASS":
            raise ValueError(f"evaluation refused unvalidated final model {model['model_id']}")
        final_path = model_dir / "model-final.zip"
        model_sha256 = _sha256(final_path)
        state, _ = _strict_json(model_dir / "state.json")
        if state.get("final_model", {}).get("sha256") != model_sha256:
            raise ValueError("evaluation final checkpoint hash differs from the training state")
        policy = PPOModel.load(final_path, device="cpu")
        if policy.num_timesteps != config["training"]["timesteps_per_model"]:
            raise ValueError("evaluation model is not the exact final 65,536-step checkpoint")
        factories = _evaluation_factories(root, protocol, model["seed_index"],
                                          evaluation["tracks"], evaluation["strata"])
        model_cells = []
        for track_id in evaluation["tracks"]:
            for stratum in evaluation["strata"]:
                model_cell_dir = cell_root / model["model_id"]
                model_cell_dir.mkdir(exist_ok=True)
                slug = f"{track_id}-{stratum}"
                event_path = model_cell_dir / f"{slug}.jsonl"
                summary_path = model_cell_dir / f"{slug}.json"
                expected = next(row for row in roster["seed_rows"]
                                if row["model_id"] == model["model_id"]
                                and row["track_id"] == track_id and row["stratum"] == stratum)
                prior = _validate_prior_cell(event_path, model, track_id, stratum)
                if prior is not None:
                    model_cells.append(prior)
                    continue
                if event_path.exists():
                    events = _read_events(event_path)
                    started_episodes = [row for row in events if row["event"] == "episode_started"]
                    finished_episodes = [row for row in events if row["event"] in ("episode_completed", "episode_invalid")]
                    if len(started_episodes) != len(finished_episodes):
                        raise ValueError("evaluation cell contains an interrupted episode; preserving without redraw")
                    completed_rows = [row["record"] for row in finished_episodes]
                    if any(not row.get("valid") for row in completed_rows):
                        summary = _cell_summary(model, track_id, stratum, completed_rows, "invalid_aborted")
                        summary["sha256"] = _digest(summary)
                        _append_event(event_path, {"event": "cell_finished", "summary": summary})
                        if not summary_path.exists():
                            _write_new_json(summary_path, summary)
                        model_cells.append(summary)
                        continue
                    if len(completed_rows) >= 60:
                        if len(completed_rows) != 60:
                            raise ValueError("evaluation cell contains excess episodes")
                        summary = _cell_summary(model, track_id, stratum, completed_rows, "completed")
                        summary["sha256"] = _digest(summary)
                        _append_event(event_path, {"event": "cell_finished", "summary": summary})
                        if not summary_path.exists():
                            _write_new_json(summary_path, summary)
                        model_cells.append(summary)
                        continue
                    # Fully completed prefix is resumed at its next predeclared episode.
                    completed_count = len(completed_rows)
                else:
                    _append_event(event_path, {
                        "event": "cell_started", "model_id": model["model_id"],
                        "candidate_id": model["candidate_id"], "seed_index": model["seed_index"],
                        "track_id": track_id, "stratum": stratum,
                        "evaluation_config_sha256": roster["evaluation_config_sha256"],
                        "model_sha256": model_sha256,
                        "task_boundary_m": 4.0, "curriculum_applied": False,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    })
                    completed_rows = []
                    completed_count = 0
                for episode_seed_row in expected["episodes"][completed_count:]:
                    episode_index = episode_seed_row["episode"]
                    _append_event(event_path, {"event": "episode_started", "episode": episode_index,
                                               "environment_seed": episode_seed_row["environment_seed"],
                                               "policy_seed": episode_seed_row["policy_seed"]})
                    try:
                        factory = factories[(track_id, stratum)]
                        if stratum == "randomized_train":
                            factory = replace(factory, reset_config=replace(
                                factory.reset_config, seed=episode_seed_row["environment_seed"]))
                        if factory.race_config is None or factory.race_config.hard_corridor_m != 4.0:
                            raise ValueError("evaluation changed the registered 4 m boundary")
                        metrics = _evaluate_episode(
                            policy, factory, track_id, episode_seed_row["environment_seed"],
                            episode_seed_row["policy_seed"], factory.simulation_config.fixed_dt,
                        )
                        if stratum == "randomized_train":
                            if metrics["environment_seed"] != factory.reset_config.seed:
                                raise ValueError("randomized TRAIN reset seed does not match evaluation roster")
                        record = {
                            "valid": True, "episode": episode_index,
                            "model_id": model["model_id"], "candidate_id": model["candidate_id"],
                            "seed_index": model["seed_index"], "track_id": track_id,
                            "stratum": stratum, "model_sha256": model_sha256,
                            "metrics": metrics,
                            "evaluation_seconds": None,
                        }
                    except Exception as error:
                        record = {"valid": False, "episode": episode_index,
                                  "model_id": model["model_id"], "candidate_id": model["candidate_id"],
                                  "seed_index": model["seed_index"], "track_id": track_id,
                                  "stratum": stratum, "model_sha256": model_sha256,
                                  "error_type": type(error).__name__, "error": str(error)}
                        completed_rows.append(record)
                        _append_event(event_path, {"event": "episode_invalid", "episode": episode_index,
                                                   "record": record})
                        break
                    completed_rows.append(record)
                    _append_event(event_path, {"event": "episode_completed", "episode": episode_index,
                                               "record": record})
                cell_status = "completed" if len(completed_rows) == 60 and all(r.get("valid") for r in completed_rows) else "invalid_aborted"
                summary = _cell_summary(model, track_id, stratum, completed_rows, cell_status)
                summary.update(model_sha256=model_sha256,
                               event_log_sha256=_sha256(event_path),
                               completed_at=datetime.now(timezone.utc).isoformat())
                summary["sha256"] = _digest(summary)
                _append_event(event_path, {"event": "cell_finished", "summary": summary})
                _write_new_json(summary_path, summary)
                _append_event(events_path, {"event": "cell_finished", "model_id": model["model_id"],
                                            "track_id": track_id, "stratum": stratum,
                                            "status": cell_status, "valid_episodes": summary["valid_episodes"],
                                            "timestamp": datetime.now(timezone.utc).isoformat()})
                model_cells.append(summary)
        model_records.append({"model_id": model["model_id"], "model_sha256": model_sha256,
                              "candidate_id": model["candidate_id"], "seed_index": model["seed_index"],
                              "run_relative_path": model["run_relative_path"], "cells": model_cells})
        all_cells.extend(model_cells)
    invalid_cells = [row for row in all_cells if row["status"] != "completed"]
    comparison = _summarize(model_records, all_cells, evaluation)
    result = {
        "schema_version": "f1rl_wp3_train_evaluation_result_v1",
        "status": "PASS" if not invalid_cells else "PASS_WITH_INVALID_CELLS",
        "scope": "TRAIN_simulation_only_development_not_scientific_acceptance",
        "campaign_manifest_sha256": campaign_manifest["sha256"],
        "evaluation_roster_sha256": roster["sha256"],
        "source_commit": campaign_manifest["source_commit"],
        "expected_models": 12, "expected_cells": 96,
        "expected_episodes": 5760,
        "completed_cells": sum(row["status"] == "completed" for row in all_cells),
        "invalid_cells": invalid_cells,
        "total_valid_episode_rows": sum(row["valid_episodes"] for row in all_cells),
        "execution_seconds": time.monotonic() - evaluation_started,
        "models": model_records, "comparison": comparison,
        "scientific_evidence": False, "heldout_access": False,
        "registered_hard_route_boundary_m": 4.0,
        "training_curriculum_applied": False,
    }
    result["sha256"] = _digest(result)
    result_path = campaign_root / "WP3-TRAIN-EVALUATION.json"
    if result_path.exists():
        existing, _ = _strict_json(result_path)
        if existing.get("sha256") != _digest({k: v for k, v in existing.items() if k != "sha256"}):
            raise ValueError("existing TRAIN evaluation result checksum mismatch")
        comparable_existing = {k: v for k, v in existing.items()
                              if k not in ("execution_seconds", "sha256")}
        comparable_new = {k: v for k, v in result.items()
                          if k not in ("execution_seconds", "sha256")}
        if comparable_existing != comparable_new:
            raise ValueError("existing TRAIN evaluation result would be overwritten with different evidence")
        result = existing
    else:
        _write_new_json(result_path, result)
    _write_evaluation_reports(campaign_root, result, model_records, all_cells, evaluation)
    return result


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _summarize(models: list[dict], cells: list[dict], evaluation: dict) -> dict:
    candidates = ["A", "B", "C", "D"]
    model_rates: dict[str, dict[int, float | None]] = {candidate: {} for candidate in candidates}
    per_track = {}
    per_stratum = {}
    secondary = {}
    for candidate in candidates:
        candidate_cells = [row for row in cells if row["candidate_id"] == candidate]
        per_track[candidate] = {}
        per_stratum[candidate] = {}
        rates = []
        for seed_index in range(3):
            seed_cells = [row for row in candidate_cells if row["seed_index"] == seed_index]
            complete = len(seed_cells) == 8 and all(row["status"] == "completed" for row in seed_cells)
            model_rates[candidate][seed_index] = (
                sum(row["success_count"] for row in seed_cells) / (8 * 60) if complete else None
            )
            if complete:
                rates.append(model_rates[candidate][seed_index])
        for track in evaluation["tracks"]:
            track_cells = [row for row in candidate_cells if row["track_id"] == track]
            complete = len(track_cells) == 6 and all(row["status"] == "completed" for row in track_cells)
            successes = sum(row["success_count"] for row in track_cells)
            per_track[candidate][track] = {
                "successes": successes if complete else None,
                "episodes": 360 if complete else None,
                "finish_rate": successes / 360 if complete else None,
            }
        for stratum in evaluation["strata"]:
            stratum_cells = [row for row in candidate_cells if row["stratum"] == stratum]
            complete = len(stratum_cells) == 12 and all(row["status"] == "completed" for row in stratum_cells)
            successes = sum(row["success_count"] for row in stratum_cells)
            per_stratum[candidate][stratum] = {
                "successes": successes if complete else None,
                "episodes": 720 if complete else None,
                "finish_rate": successes / 720 if complete else None,
            }
        complete_cells = [row for row in candidate_cells if row["status"] == "completed"]
        secondary[candidate] = {
            key: _mean(complete_cells, key)
            for key in ("mean_route_progress", "mean_completed_checkpoints", "collision_rate",
                        "mean_collision_count", "hard_route_violation_rate", "timeout_rate",
                        "mean_steering_rate_rms_rad_per_s", "mean_episode_duration_seconds")
        }
        reason_counts: dict[str, int] = {}
        for cell in complete_cells:
            for reason, count in cell["failure_reasons"].items():
                reason_counts[reason] = reason_counts.get(reason, 0) + count
        secondary[candidate]["failure_reason_counts"] = dict(sorted(reason_counts.items()))
        secondary[candidate]["seed_block_finish_rates"] = model_rates[candidate]
        secondary[candidate]["finish_rate_mean_across_seed_blocks"] = (
            sum(rates) / len(rates) if len(rates) == 3 else None)
        secondary[candidate]["finish_rate_seed_block_sd"] = _sample_sd(rates) if len(rates) == 3 else None
    complete_all = all(model_rates[c][s] is not None for c in candidates for s in range(3))
    effects = {"status": "descriptive_only_three_seed_blocks_no_confirmatory_inference"}
    if complete_all:
        reward_effects = [((model_rates["B"][s] + model_rates["D"][s]) -
                           (model_rates["A"][s] + model_rates["C"][s])) / 2 for s in range(3)]
        corridor_effects = [((model_rates["C"][s] + model_rates["D"][s]) -
                             (model_rates["A"][s] + model_rates["B"][s])) / 2 for s in range(3)]
        interaction_effects = [(model_rates["D"][s] - model_rates["C"][s]) -
                               (model_rates["B"][s] - model_rates["A"][s]) for s in range(3)]
        effects["per_seed_block"] = {
            "higher_lateral_penalty_minus_lower": reward_effects,
            "3m_corridor_minus_2m_corridor": corridor_effects,
            "factorial_interaction": interaction_effects,
        }
        effects["mean_and_seed_sd"] = {
            name: {"mean": sum(values) / 3, "seed_block_sd": _sample_sd(values),
                   "range": [min(values), max(values)]}
            for name, values in effects["per_seed_block"].items()
        }
    best = None
    ranking_eligible = [candidate for candidate in candidates
                        if secondary[candidate]["finish_rate_mean_across_seed_blocks"] is not None]
    if ranking_eligible:
        best = max(ranking_eligible, key=lambda c: secondary[c]["finish_rate_mean_across_seed_blocks"])
    return {
        "candidate_finish_rate_summary": secondary,
        "per_track_finish_rate": per_track,
        "per_stratum_finish_rate": per_stratum,
        "factor_effects": effects,
        "best_development_candidate": best,
        "best_candidate_confidence": "descriptive_only_not_definitive" if best else None,
        "cross_seed_variability": {c: secondary[c]["finish_rate_seed_block_sd"] for c in candidates},
    }
