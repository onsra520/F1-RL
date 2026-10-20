"""Separate simulation-only, TRAIN-only evaluation of WP3 final checkpoints."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from ..learning.algorithms.ppo import PPOModel
from ..learning.evaluation.evaluator import evaluate_episode
from ..learning.pilot_preflight import TRAIN_IDS
from ..research.wp7_protocol import build_protocol, prepare_evaluation_factory
from .campaign import _canonical, _sha256, _strict_json, _write_new_json, _atomic_json


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _seed(domain: str, *parts: object) -> int:
    raw = "/".join([domain, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(raw).digest()[:4], "big")


def build_evaluation_roster(root: Path, campaign_manifest: dict, config: dict) -> dict:
    evaluation = config["evaluation"]
    binding = {
        "schema_version": evaluation["schema_version"],
        "campaign_manifest_sha256": campaign_manifest["sha256"],
        "source_commit": campaign_manifest["source_commit"],
        "simulation_only_config_sha256": campaign_manifest["simulation_only_config_sha256"],
        "registration_sha256": campaign_manifest["registration_sha256"],
        "track_manifest_sha256": campaign_manifest["track_manifest_sha256"],
        "evaluation": evaluation,
    }
    evaluation_config_sha256 = _digest(binding)
    seed_rows = []
    environment_seeds = set()
    policy_seeds = set()
    per_cell = evaluation["episodes_per_track_stratum_model"]
    schedule = campaign_manifest["execution_order"]
    for model in schedule:
        for track_id in evaluation["tracks"]:
            for stratum in evaluation["strata"]:
                episodes = []
                for episode in range(per_cell):
                    environment_seed = _seed(
                        "f1rl-wp3-eval-environment-seed-v1", evaluation_config_sha256,
                        track_id, stratum, episode,
                    )
                    policy_seed = _seed(
                        "f1rl-wp3-eval-policy-seed-v1", evaluation_config_sha256,
                        track_id, stratum, episode,
                    )
                    environment_seeds.add(environment_seed)
                    policy_seeds.add(policy_seed)
                    episodes.append({"episode": episode, "environment_seed": environment_seed,
                                     "policy_seed": policy_seed})
                seed_rows.append({"model_id": model["model_id"], "candidate_id": model["candidate_id"],
                                  "seed_index": model["seed_index"], "track_id": track_id,
                                  "stratum": stratum, "episodes": episodes})
    # Same cell/episode seeds are intentionally paired across all 12 models.
    for track_id in evaluation["tracks"]:
        for stratum in evaluation["strata"]:
            for episode in range(per_cell):
                env_values = {row["episodes"][episode]["environment_seed"] for row in seed_rows
                              if row["track_id"] == track_id and row["stratum"] == stratum}
                policy_values = {row["episodes"][episode]["policy_seed"] for row in seed_rows
                                 if row["track_id"] == track_id and row["stratum"] == stratum}
                if len(env_values) != 1 or len(policy_values) != 1:
                    raise ValueError("evaluation seeds must be identical across candidates and PPO seeds")
    expected_unique_seed_count = len(evaluation["tracks"]) * len(evaluation["strata"]) * per_cell
    if (len(environment_seeds) != expected_unique_seed_count
            or len(policy_seeds) != expected_unique_seed_count):
        raise ValueError("evaluation environment seed derivation collision")
    roster = {
        **binding, "evaluation_config_sha256": evaluation_config_sha256,
        "seed_rows": seed_rows,
        "expected_cells": len(seed_rows),
        "expected_episodes": len(seed_rows) * per_cell,
        "candidate_independent_paired_seeds": True,
        "scientific_evidence": False, "heldout_access": False,
    }
    roster["sha256"] = _digest(roster)
    return roster


def _append_event(path: Path, event: dict) -> None:
    record = {"schema_version": "f1rl_wp3_train_eval_event_v1", **event}
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def _read_events(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("evaluation lifecycle JSONL has an incomplete record")
            row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            if row.get("schema_version") != "f1rl_wp3_train_eval_event_v1":
                raise ValueError("evaluation event schema mismatch")
            records.append(row)
    return records


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
    completed_summaries = {}
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
    _write_evaluation_reports(campaign_root, result, models, cells, evaluation)
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


def _write_new_or_match(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite existing WP3 report: {path.name}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        import os
        os.fsync(stream.fileno())


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
    series = {}
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
    width,height=1000,560; left,right,top,bottom=100,900,70,430
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
