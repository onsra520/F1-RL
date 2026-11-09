"""TRAIN-only development evaluation helpers; never grant scientific admission."""
from math import sqrt
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
from dataclasses import asdict
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import fmean
import json
import os

import yaml

from f1rl.learning.evaluation.evaluator import evaluate_episode


def wilson_interval(successes, total):
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total or total <= 0:
        raise ValueError("valid success and episode counts required")
    z = 1.959963984540054
    proportion = successes / total
    divisor = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / divisor
    margin = z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / divisor
    return max(0.0, centre - margin), min(1.0, centre + margin)


def episode_outcome(metrics, final_info):
    if not isinstance(final_info, dict):
        raise ValueError("final episode info required")
    outcome = final_info.get("research_outcome") or {}
    reason = outcome.get("reason") if isinstance(outcome, dict) else None
    if metrics.success and not metrics.truncated:
        return "finished"
    if metrics.truncated and reason in (None, "step_limit"):
        return "valid_step_limit"
    if not metrics.truncated and reason == "hard_route_violation" and outcome.get("valid_sample") is True:
        return "hard_route_violation"
    if not metrics.truncated and reason == "tilt_envelope_exit" and outcome.get("valid_sample") is True:
        from f1rl.learning.roll_envelope_registered import validate_registered_roll_envelope_info
        if validate_registered_roll_envelope_info(final_info):
            return "tilt_envelope_exit"
    if not metrics.truncated and reason in ("physical_crash", "physical_crash_development"):
        from f1rl.learning.crash_terminal_registered import validate_registered_crash_terminal_info
        from f1rl.learning.evaluation.crash_contract import validate_crash_terminal_info
        if validate_registered_crash_terminal_info(final_info) or validate_crash_terminal_info(final_info):
            return "certified_physical_crash"
    raise ValueError("unclassified or uncertified terminal outcome")


def evaluate_one(*, factory, policy, track_id, condition, environment_seed, policy_seed):
    """Run one CPU stochastic TRAIN episode with isolated policy RNG."""
    import torch

    if condition not in ("nominal", "perturbation"):
        raise ValueError("unregistered development condition")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in (environment_seed, policy_seed)):
        raise ValueError("evaluation seeds must be uint32")
    if environment_seed == policy_seed:
        raise ValueError("environment and policy seeds must differ")
    if condition == "nominal":
        factory = replace(factory, reset_config=None, reset_track_id=None)
    else:
        if factory.reset_config is None or factory.reset_track_id != track_id:
            raise ValueError("registered perturbation reset required")
        factory = replace(factory, reset_config=replace(factory.reset_config, seed=environment_seed))
    terminal = []

    def observe(event):
        if event["terminated"] or event["truncated"]:
            terminal.append(deepcopy(event["info"]))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(policy_seed)
        metrics = evaluate_episode(track_id=track_id, factory=factory, policy=policy,
                                   reset_seed=environment_seed, deterministic=False, step_observer=observe)
    if len(terminal) != 1:
        raise ValueError("exactly one terminal event required")
    research = terminal[0].get("research_outcome")
    if (not isinstance(research, dict) or research.get("version") != "wp7_episode_outcome_v1"
            or research.get("valid_sample") is not True):
        raise ValueError("terminal physics is invalid or uncertified")
    return {"outcome": episode_outcome(metrics, terminal[0]), "valid_physics": True,
            "episode_return": metrics.episode_return}


def validate_loaded_policy(policy, scenarios):
    if (policy.num_timesteps != 200704 or policy._model.observation_space != scenarios.observation_space
            or policy._model.action_space != scenarios.action_space or not policy.policy_weights_finite()):
        raise ValueError("trained policy steps, spaces or weights do not match TRAIN factories")


def evaluate_batch(root, output, evidence_root):
    """Evaluate the six completed outputs once on the registered TRAIN roster."""
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.training import TrainingRunResult
    from f1rl.research.wp7_protocol import _unique
    from f1rl.research.wp7_trusted_reviews import _utc
    from f1rl.research.wp7_train5_internal_execution import OWNER_KEY, _owner_public_key
    from f1rl.research.wp7_train5_v4_admission import development_policy_seeds, development_seeds, load_v4_protocol
    from f1rl.research.wp7_train5_v4_budget import load_v4_budget
    from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS, evaluate_v4_runs, prepare_v4_registered_runner
    from f1rl.research.wp7_train5_v4_dev_metrics import load_registered_development_contract
    from scripts.wp7_train5_batch import _require_clean_source
    from scripts.wp7_train5_execution_authorization import verify_authorization
    from scripts.wp7_train5_v4_factory_preview import preview_registered_factories
    from scripts.wp7_train5_v4_review_manifest import load_review_manifest

    root, output = Path(root).resolve(strict=True), Path(output).resolve(strict=True)
    if output == root or root in output.parents:
        raise ValueError("evaluation output must be outside source checkout")
    result_path = output / "batch-result.json"
    if result_path.is_symlink():
        raise ValueError("batch result cannot be a symlink")
    batch = json.loads(result_path.read_text(encoding="utf8"), object_pairs_hook=_unique)
    if (not isinstance(batch, dict) or batch.get("schema") != "train5_internal_batch_result_v1"
            or not isinstance(batch.get("runs"), list) or len(batch["runs"]) != 6
            or batch.get("scientific_admission") is not False or batch.get("heldout_evaluated") is not False):
        raise ValueError("complete six-run internal batch required")
    _require_clean_source(root, batch.get("source_commit"))
    protocol, protocol_sha = load_v4_protocol(root)
    manifest, manifest_sha = load_review_manifest(root)
    if batch.get("protocol_sha256") != protocol_sha or batch.get("manifest_sha256") != manifest_sha:
        raise ValueError("batch source or manifest differs")
    owner_key = _owner_public_key(root, manifest["review_source_bindings"][OWNER_KEY])
    original = {name: sha256((root / "models" / name / "model-final.zip").read_bytes()).hexdigest()
                for name in FROZEN_CHECKPOINTS}
    if original != FROZEN_CHECKPOINTS or batch.get("frozen_checkpoint_sha256") != original:
        raise ValueError("original frozen checkpoint identity differs")
    if (output / "evaluation-journal.jsonl").exists() or (output / "development-result.json").exists():
        raise ValueError("evaluation already started; preserve existing evidence")
    env_seeds = development_seeds(root, protocol, protocol_sha)
    policy_seeds = development_policy_seeds(root, protocol, protocol_sha)
    expected = {(r["cohort"], r["replicate"]): r for r in protocol["runs"]}
    runs, entries = {}, {}
    for entry in batch["runs"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid run entry")
        key = (entry.get("cohort"), entry.get("replicate"))
        if key not in expected or key in entries or entry.get("actual_steps") != 200704:
            raise ValueError("invalid or duplicated run identity")
        directory = Path(entry["run_directory"]).resolve(strict=True)
        model = Path(entry["final_model"]).resolve(strict=True)
        if directory.parent != output or not model.is_relative_to(directory) or not model.is_file():
            raise ValueError("run or model path outside batch output")
        if sha256(model.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ValueError("trained checkpoint hash changed")
        state = json.loads((directory / "state.json").read_text(encoding="utf8"), object_pairs_hook=_unique)
        reference = prepare_v4_registered_runner(root, output, cohort=key[0], replicate=key[1])
        if (state.get("status") != "completed" or state.get("run_id") != directory.name
                or state.get("execution_scope") != "INTERNAL_SIMULATION_ONLY"
                or state.get("ppo") != asdict(reference._config.ppo)
                or state.get("vector_env") != asdict(reference._config.vector_env)
                or state.get("sampling_seed") != expected[key]["seeds"]["sampling"]
                or state.get("training_reset_seed") != expected[key]["seeds"]["perturbation"]
                or state.get("training_tracks") != protocol["track_ids"]
                or state.get("initial_checkpoint") != reference._initial_checkpoint
                or state.get("mode") != "multitrack" or state.get("requested_total_timesteps") != 200000
                or state.get("checkpoint_every_steps") is not None or state.get("latest_checkpoint") is not None
                or state.get("actual_total_timesteps") != 200704
                or (directory / state.get("final_model", "")).resolve() != model):
            raise ValueError("completed run state differs from registered configuration")
        authorization = state.get("internal_authorization")
        try:
            issued = _utc(authorization["payload"]["issued_at_utc"])
        except (TypeError, KeyError) as exc:
            raise ValueError("saved run authorization is missing") from exc
        verify_authorization(authorization, owner_key, protocol_sha256=protocol_sha, manifest_sha256=manifest_sha,
                             source_commit=batch["source_commit"], cohort=key[0], replicate=key[1],
                             output_root=str(output), now=issued)
        metric_cohort = "scratch" if key[0] == "train5_from_scratch" else "frozen_initialization"
        runs[(metric_cohort, key[1])] = TrainingRunResult(directory.name, directory, model, (), 200000, 200704)
        entries[key] = entry
    if set(entries) != set(expected):
        raise ValueError("missing registered run")
    scenarios = {key: preview_registered_factories(root, Path(evidence_root), cohort=key[0], replicate=key[1])
                 for key in expected}
    policies = {key: PPOModel.load(runs[("scratch" if key[0] == "train5_from_scratch" else "frozen_initialization", key[1])].final_model_path,
                                   device="cpu") for key in expected}
    for key, policy in policies.items():
        validate_loaded_policy(policy, scenarios[key])
    journal = output / "evaluation-journal.jsonl"
    with journal.open("x", encoding="utf8"):
        pass

    def append(event):
        with journal.open("a", encoding="utf8", newline="\n") as handle:
            handle.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **event},
                                    sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    records = []
    for key in expected:
        cohort, replicate = key
        metric_cohort = "scratch" if cohort == "train5_from_scratch" else "frozen_initialization"
        by_track = {row.track_id: row.factory for row in scenarios[key].scenarios}
        for track in protocol["track_ids"]:
            for condition in protocol["development"]["conditions"]:
                for episode in range(60):
                    seed_key = (cohort, replicate, track, condition, episode)
                    identity = {"cohort": metric_cohort, "replicate": replicate, "track_id": track,
                                "condition": condition, "episode_index": episode,
                                "environment_seed": env_seeds[seed_key], "policy_seed": policy_seeds[seed_key]}
                    append({"event": "started", **identity})
                    try:
                        metrics = evaluate_one(factory=by_track[track], policy=policies[key], track_id=track,
                                               condition=condition, environment_seed=env_seeds[seed_key],
                                               policy_seed=policy_seeds[seed_key])
                        row = {"cohort": metric_cohort, "replicate": replicate, "track_id": track,
                               "condition": condition, "episode_index": episode, "checkpoint_step": 200704,
                               "checkpoint_sha256": entries[key]["sha256"], "evaluation_seed": env_seeds[seed_key], **metrics}
                        append({"event": "completed", **identity, "record": row})
                        records.append(row)
                    except BaseException as exc:
                        append({"event": "failed", **identity, "error_type": type(exc).__name__, "error": str(exc)})
                        raise
                print(f"Evaluated {cohort} replicate {replicate}: {track} / {condition} ({len(records)}/3600)", flush=True)
    pilot = yaml.safe_load((root / "configs/recovery/wp7-pilot-200k.yml").read_text(encoding="utf8"))
    summary = evaluate_v4_runs(records, runs, load_registered_development_contract(root), load_v4_budget(root), pilot)
    cells = defaultdict(Counter)
    for row in records:
        cell = (row["cohort"], row["replicate"], row["track_id"], row["condition"])
        cells[cell][row["outcome"]] += 1
    detail = []
    for (cohort, replicate, track, condition), counts in cells.items():
        n = sum(counts.values())
        detail.append({"cohort": cohort, "replicate": replicate, "track_id": track, "condition": condition,
                       "counts": dict(counts), "episodes": n,
                       "finish_rate": counts["finished"] / n,
                       "finish_wilson_95": wilson_interval(counts["finished"], n),
                       "zero_crash_upper_95": 1 - 0.05 ** (1 / n) if counts["certified_physical_crash"] == 0 else None,
                       "zero_hard_route_upper_95": 1 - 0.05 ** (1 / n) if counts["hard_route_violation"] == 0 else None})
    policy_counts = defaultdict(Counter)
    for row in records:
        policy_counts[(row["cohort"], row["replicate"])]["episodes"] += 1
        policy_counts[(row["cohort"], row["replicate"])]["finished"] += int(row["outcome"] == "finished")
    cohort_summaries = {}
    for cohort in ("scratch", "frozen_initialization"):
        rates = [policy_counts[(cohort, i)]["finished"] / policy_counts[(cohort, i)]["episodes"] for i in range(3)]
        cohort_summaries[cohort] = {"policy_finish_rates": rates, "mean": fmean(rates),
                                    "range": [min(rates), max(rates)]}
    if {name: sha256((root / "models" / name / "model-final.zip").read_bytes()).hexdigest()
            for name in FROZEN_CHECKPOINTS} != original:
        raise ValueError("original frozen checkpoint changed during evaluation")
    result = {"schema": "train5_internal_development_v1", "protocol_sha256": protocol_sha,
              "manifest_sha256": manifest_sha, "summary": summary, "cells": detail,
              "cohort_policy_summaries": cohort_summaries,
              "interval_note": "Conditional descriptive intervals per fixed policy/cell; not independent policies.",
              "detection_probability_60": {str(p): 1 - (1 - p) ** 60 for p in (0.01, 0.05, 0.10)},
              "scientific_admission": False, "heldout_evaluated": False}
    with (output / "development-result.json").open("x", encoding="utf8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result
