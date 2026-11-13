"""Run source-bound, non-training PPO diagnostics on M5 development tracks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M4_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m4-m5-development-v2")
DIAG_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-diagnosis-20261111")
SOURCE_RUNNERS = ("scripts/m5_c0_diagnostic.py", "scripts/wp7_dual_turn_v4_feasibility.py")
TRACKS = {"right-turn-train-v2": "right-turn-v1", "ramp-v1": "ramp-v1"}
MODES = ("deterministic", "stochastic")
STEP_LIMIT = 500
WALL_TIMEOUT_SECONDS = 120
CHECKPOINT = M4_ROOT / "runs/C-seed0/dev-0-14f3f2a9cc2e/model-final.zip"
M5_EPISODES = M4_ROOT / "evaluation/episodes.jsonl"
M5_REPORT = M4_ROOT / "evaluation/REPORT.json"
OLD_V2_ROOT = DIAG_ROOT / "right-turn-v2-comparison"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def serialize_episode_metrics(metrics: object) -> dict:
    if not is_dataclass(metrics) or isinstance(metrics, type):
        raise TypeError("episode metrics must be a dataclass instance")
    return asdict(metrics)


def predict_policy_action(model: object, observation, *, deterministic: bool):
    return model.predict(observation, deterministic=deterministic)


def terminal_summary(metrics: object, last_step: dict) -> dict:
    outcome = last_step.get("research_outcome") or {}
    crash = last_step.get("physical_crash_terminal")
    if metrics.success:
        reason = "finish"
    elif isinstance(outcome, dict) and outcome.get("reason"):
        reason = str(outcome["reason"])
    elif crash:
        reason = "physical_crash"
    elif metrics.truncated:
        reason = "step_limit"
    else:
        reason = "terminal_failure"
    return {
        "finished": bool(metrics.success),
        "simulation_timeout": bool(metrics.truncated),
        "wall_timeout": False,
        "reason": reason,
        "next_checkpoint_index": int(last_step["next_checkpoint_index"]),
        "completed_checkpoint_ids": list(last_step.get("completed_checkpoint_ids", ())),
        "collision_obstacle_ids": list(last_step.get("collision_obstacle_ids", ())),
        "physical_crash_terminal": crash,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_self_hash(doc: dict) -> None:
    if doc.get("sha256") != canonical_sha256(
        {key: value for key, value in doc.items() if key != "sha256"}
    ):
        raise ValueError(f"canonical self-hash mismatch: {doc}")


def _source_bindings() -> dict[str, str]:
    listing = subprocess.run(["git", "ls-files", "-z", "--", "f1rl", "configs"], cwd=ROOT,
                             check=True, capture_output=True).stdout.decode("utf-8")
    names = set(filter(None, listing.split("\x00"))) | set(SOURCE_RUNNERS) | {
        "configs/research/wp7-train5-paired-v2-portable-source-revision-v1.json",
        "configs/research/wp7-train5-source-compatibility-v1.json",
        "configs/research/wp7-train5-paired-v2-registration.json",
    }
    return {name: file_sha256(ROOT / name) for name in sorted(names)}


def _make_plan(artifact_root: Path) -> dict:
    episodes = [json.loads(line) for line in M5_EPISODES.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if not episodes or any(row.get("heldout_access") is not False
                           or row.get("scientific_evidence") is not False for row in episodes):
        raise ValueError("M5 seed source is missing or outside TRAIN-only scope")
    roster = []
    for track_id, source_track_id in TRACKS.items():
        rows = sorted((row for row in episodes if row.get("model_id") == "C-seed0"
                       and row.get("track_id") == source_track_id), key=lambda row: row["episode"])
        if len(rows) != 10 or [row["episode"] for row in rows] != list(range(10)):
            raise ValueError(f"expected the predeclared ten M5 seeds for {source_track_id}")
        pairs = [(row["environment_seed"], row["policy_seed"]) for row in rows]
        if len(set(pairs)) != 10 or len({x[0] for x in pairs}) != 10 or len({x[1] for x in pairs}) != 10:
            raise ValueError(f"M5 seed roster is not genuinely varied for {source_track_id}")
        for row in rows:
            roster.append({
                "track_id": track_id,
                "seed_source_track_id": source_track_id,
                "episode": row["episode"],
                "environment_seed": row["environment_seed"],
                "policy_seed": row["policy_seed"],
            })

    old_plan = _load_json(OLD_V2_ROOT / "DIAGNOSTIC-PLAN.json")
    old_report = _load_json(OLD_V2_ROOT / "RECONSTRUCTED-REPORT.json")
    _verify_self_hash(old_plan)
    _verify_self_hash(old_report)
    old_pairs = {(row["environment_seed"], row["policy_seed"]): row
                 for row in old_plan["seed_roster"]}
    reused = []
    for seed in roster:
        if seed["track_id"] != "right-turn-train-v2":
            continue
        old_seed = old_pairs.get((seed["environment_seed"], seed["policy_seed"]))
        if old_seed is None:
            continue
        for mode in MODES:
            case = next((row for row in old_report["episodes"]
                         if row["episode"] == seed["episode"] and row["mode"] == mode), None)
            if case is None or case["step_count"] > STEP_LIMIT or not case["success"]:
                raise ValueError("prior V2 trace cannot satisfy the M5 500-step evaluation limit")
            trace = Path(case["step_artifact"])
            summary = OLD_V2_ROOT / "reconstructed-episodes" / f"{case['case']}.json"
            reused.append({
                "track_id": seed["track_id"], "episode": seed["episode"], "mode": mode,
                "summary_path": str(summary), "summary_sha256": case["summary_file_sha256"],
                "trace_path": str(trace), "trace_sha256": case["step_artifact_sha256"],
                "source_commit": old_report["source_commit"],
                "source_tree_sha256": old_report["source_tree_sha256"],
            })

    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   check=True, capture_output=True, text=True).stdout.strip()
    plan = {
        "schema_version": "f1rl_m5_c0_pure_ppo_diagnostic_plan_v1",
        "status": "TRAIN_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "source_commit": source_commit,
        "source_file_sha256": _source_bindings(),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "checkpoint_timesteps": 65536,
        "track_file_sha256": {
            track_id: file_sha256(ROOT / "assets/tracks/training" / f"{track_id}.json")
            for track_id in TRACKS
        },
        "physics_profile_sha256": "c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37",
        "action_contract": "steering-throttle-brake-v2",
        "mode": "Pure PPO; no Auto Drive or Hybrid overrides",
        "tracks": list(TRACKS),
        "m5_seed_source": {
            "episodes_jsonl": str(M5_EPISODES),
            "episodes_jsonl_sha256": file_sha256(M5_EPISODES),
            "report_json": str(M5_REPORT),
            "report_json_sha256": file_sha256(M5_REPORT),
            "evaluation_plan_json": str(M4_ROOT / "M5-EVALUATION-PLAN.json"),
            "evaluation_plan_sha256": file_sha256(M4_ROOT / "M5-EVALUATION-PLAN.json"),
        },
        "prior_v2_trace_source": {
            "plan_sha256": file_sha256(OLD_V2_ROOT / "DIAGNOSTIC-PLAN.json"),
            "reconstructed_report_sha256": file_sha256(OLD_V2_ROOT / "RECONSTRUCTED-REPORT.json"),
            "reused_case_count": len(reused),
            "reused_cases": reused,
        },
        "seed_roster": roster,
        "inference_modes": list(MODES),
        "episodes_per_track_per_mode": 10,
        "simulation_step_limit": STEP_LIMIT,
        "worker_wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "artifact_root": str(artifact_root.resolve()),
        "heldout_access": False,
        "scientific_evidence": False,
        "training_started": False,
    }
    plan["sha256"] = canonical_sha256(plan)
    return plan


def _verify_plan(plan: dict) -> None:
    _verify_self_hash(plan)
    if plan.get("schema_version") != "f1rl_m5_c0_pure_ppo_diagnostic_plan_v1":
        raise ValueError("unexpected diagnostic plan schema")
    if plan["source_file_sha256"] != _source_bindings():
        raise ValueError("diagnostic source files changed after plan freeze")
    if file_sha256(Path(plan["checkpoint_path"])) != plan["checkpoint_sha256"]:
        raise ValueError("C-seed0 checkpoint hash mismatch")
    if file_sha256(M5_EPISODES) != plan["m5_seed_source"]["episodes_jsonl_sha256"]:
        raise ValueError("historical M5 seed source changed")
    if file_sha256(M5_REPORT) != plan["m5_seed_source"]["report_json_sha256"]:
        raise ValueError("historical M5 result report changed")
    if file_sha256(M4_ROOT / "M5-EVALUATION-PLAN.json") != plan["m5_seed_source"]["evaluation_plan_sha256"]:
        raise ValueError("historical M5 seed plan changed")
    if file_sha256(OLD_V2_ROOT / "DIAGNOSTIC-PLAN.json") != plan["prior_v2_trace_source"]["plan_sha256"]:
        raise ValueError("prior Right Turn V2 plan changed")
    if file_sha256(OLD_V2_ROOT / "RECONSTRUCTED-REPORT.json") != plan["prior_v2_trace_source"]["reconstructed_report_sha256"]:
        raise ValueError("prior Right Turn V2 reconstructed report changed")
    for track_id in TRACKS:
        path = ROOT / "assets/tracks/training" / f"{track_id}.json"
        if file_sha256(path) != plan["track_file_sha256"][track_id]:
            raise ValueError(f"TRAIN track asset changed: {track_id}")


def _case_id(track_id: str, episode: int, mode: str) -> str:
    return f"C-seed0__{track_id}__ep{episode:02d}__{mode}"


def _load_trace(path: Path, expected_sha: str) -> list[dict]:
    if file_sha256(path) != expected_sha:
        raise ValueError(f"trace hash mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or [row["step"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"trace step sequence is invalid: {path}")
    if any(row["terminated"] or row["truncated"] for row in rows[:-1]):
        raise ValueError(f"trace contains an early terminal: {path}")
    return rows


def _reused_summary(plan: dict, case_binding: dict) -> dict:
    old = _load_json(Path(case_binding["summary_path"]))
    _verify_self_hash(old)
    if file_sha256(Path(case_binding["summary_path"])) != case_binding["summary_sha256"]:
        raise ValueError("prior reconstructed summary SHA-256 mismatch")
    rows = _load_trace(Path(case_binding["trace_path"]), case_binding["trace_sha256"])
    last = rows[-1]
    if (len(rows) != old["step_count"] or not last["finished"] or not last["terminated"]
            or last["truncated"] or last["physical_crash_terminal"] is not None
            or last["collision_obstacle_ids"]):
        raise ValueError("prior V2 trace does not end in a clean finish")
    if old["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("prior V2 trace used a different checkpoint")
    result = {
        "track_id": old["track_id"], "success": old["success"], "truncated": old["truncated"],
        "steps": old["step_count"], "episode_time": old["episode_time_s"],
        "completion_time": old["episode_time_s"], "episode_return": old["episode_return"],
        "reward_component_totals": old["reward_component_totals"],
        "collision_count": old["collision_step_count"], "path_length": old["path_length_m"],
        "final_normalized_route_progress": old["final_normalized_route_progress"],
        "mean_speed": old["mean_speed_mps"], "max_speed": old["max_speed_mps"],
    }
    summary = {
        "schema_version": "f1rl_m5_c0_diagnostic_episode_v1",
        "case_id": _case_id(old["track_id"], old["episode"], old["mode"]),
        "model_id": "C-seed0", "track_id": old["track_id"], "episode": old["episode"],
        "mode": old["mode"], "environment_seed": old["environment_seed"],
        "policy_seed": old["policy_seed"], "checkpoint_sha256": old["checkpoint_sha256"],
        "track_file_sha256": old["track_asset_sha256"], "status": "complete_reused_trace",
        "metrics_source": "hash_verified_prior_step_trace_reconstruction",
        "episode_metrics": result,
        "terminal": {
            "finished": True, "simulation_timeout": False, "wall_timeout": False,
            "reason": "finish", "next_checkpoint_index": int(last["next_checkpoint_index"]),
            "completed_checkpoint_ids": None,
            "checkpoint_index_step_counts": old["checkpoint_index_step_counts"],
            "collision_obstacle_ids": [],
            "physical_crash_terminal": None,
        },
        "action_summary": _action_summary(rows),
        "step_count": old["step_count"], "trace_path": str(Path(case_binding["trace_path"]).resolve()),
        "trace_sha256": case_binding["trace_sha256"],
        "reused_source_commit": case_binding["source_commit"],
        "heldout_access": False, "scientific_evidence": False,
    }
    summary["sha256"] = canonical_sha256(summary)
    return summary


def _action_summary(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "mean_steering": sum(row["applied_vehicle_control"]["steering"] for row in rows) / count,
        "mean_throttle": sum(row["applied_vehicle_control"]["throttle"] for row in rows) / count,
        "mean_brake": sum(row["applied_vehicle_control"]["brake"] for row in rows) / count,
        "max_speed_mps": max(float(row.get("vehicle_speed") or 0.0) for row in rows),
        "checkpoint_indices": sorted({int(row["next_checkpoint_index"]) for row in rows}),
        "collision_step_count": sum(bool(row["collision_obstacle_ids"]) for row in rows),
    }


def _run_episode(plan: dict, seed: dict, mode: str) -> dict:
    import numpy as np

    from f1rl.learning.action import BrakingPolicyAction, map_policy_action
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.episode import EpisodeConfig
    from f1rl.learning.evaluation.evaluator import evaluate_episode
    from f1rl.learning.training.env_factory import F1RLEnvFactory
    from scripts.wp7_dual_turn_v4_feasibility import build_factory as build_v2_factory

    track_id = seed["track_id"]
    if track_id == "right-turn-train-v2":
        factory, _ = build_v2_factory(track_id)
    else:
        from f1rl.research.wp3_eval_only_coordinator import (
            FULL06_RUNTIME_PROFILE, apply_runtime_profile,
        )
        from f1rl.research.wp7_protocol import build_protocol, prepare_evaluation_factory

        protocol = build_protocol(ROOT, profile="spatial_rotating_owner_v3")
        factory, _ = prepare_evaluation_factory(
            ROOT, protocol, "multitrack-seed0", track_id, "nominal", seed["episode"]
        )
        factory, _ = apply_runtime_profile(factory, FULL06_RUNTIME_PROFILE)
    factory = replace(factory, episode_config=EpisodeConfig(plan["simulation_step_limit"]))
    if (factory.physics_config.sha256 != plan["physics_profile_sha256"]
            or factory.action_contract.version != plan["action_contract"]):
        raise ValueError("C0 evaluation factory Physics/action contract mismatch")

    model = PPOModel.load(plan["checkpoint_path"], device="cpu")
    if model.num_timesteps != plan["checkpoint_timesteps"]:
        raise ValueError("C0 checkpoint is not the expected 65,536-step model")
    PPOModel.seed_policy_rng(seed["policy_seed"])

    features = list(factory.observation_encoder.feature_names)
    policy_observation = None

    class RecordingPolicy:
        def predict(self, observation, *, deterministic=True):
            nonlocal policy_observation
            policy_observation = observation.copy()
            return predict_policy_action(model, observation, deterministic=deterministic)

    policy = RecordingPolicy()
    output = Path(plan["artifact_root"])
    case_id = _case_id(track_id, seed["episode"], mode)
    trace_path = output / "steps" / f"{case_id}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    old_create = F1RLEnvFactory.create

    def create_with_telemetry(self, *, reset_rank=0, reset_sequence=0):
        env = old_create(self, reset_rank=reset_rank, reset_sequence=reset_sequence)
        env.set_training_telemetry_enabled(True)
        return env

    F1RLEnvFactory.create = create_with_telemetry
    try:
        with trace_path.open("x", encoding="utf-8", newline="\n", buffering=1) as stream:
            def observe(event):
                raw = [float(value) for value in event["action"]]
                applied = map_policy_action(BrakingPolicyAction(*raw))
                info = event["info"]
                telemetry = info.get("telemetry", {})
                race = info.get("race", {})
                physics = info.get("physics", {})
                row = {
                    "schema_version": "f1rl_m5_c0_diagnostic_step_v1",
                    "case_id": case_id, "track_id": track_id, "episode": seed["episode"],
                    "environment_seed": seed["environment_seed"], "policy_seed": seed["policy_seed"],
                    "mode": mode, "step": event["step"], "sim_time_s": event["elapsed_time"],
                    "observation": [float(value) for value in policy_observation],
                    "observation_features": features, "raw_policy_action": raw,
                    "applied_vehicle_control": {
                        "steering": float(applied.steering), "throttle": float(applied.throttle),
                        "brake": float(applied.brake),
                    },
                    "reward": float(event["reward"]),
                    "reward_components": info.get("reward_components", {}),
                    "next_checkpoint_index": int(event["next_checkpoint_index"]),
                    "completed_checkpoint_ids": list(race.get("completed_checkpoint_ids", ())),
                    "collision_obstacle_ids": list(telemetry.get("obstacle_hit_ids", ())),
                    "physical_crash_terminal": info.get("physical_crash_terminal"),
                    "research_outcome": info.get("research_outcome"),
                    "vehicle_position": telemetry.get("vehicle_position"),
                    "vehicle_velocity": telemetry.get("vehicle_velocity"),
                    "vehicle_speed": telemetry.get("vehicle_speed"),
                    "finished": bool(event["finished"]),
                    "terminated": bool(event["terminated"]), "truncated": bool(event["truncated"]),
                    "physics_sha256": physics.get("sha256"),
                }
                line = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.write(line + "\n")
                trace_rows.append(row)
            metrics = evaluate_episode(
                track_id=track_id, factory=factory, policy=policy,
                reset_seed=seed["environment_seed"], deterministic=(mode == "deterministic"),
                step_observer=observe,
            )
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        F1RLEnvFactory.create = old_create

    serialized_metrics = serialize_episode_metrics(metrics)
    terminal = terminal_summary(metrics, trace_rows[-1])
    summary = {
        "schema_version": "f1rl_m5_c0_diagnostic_episode_v1",
        "case_id": case_id, "model_id": "C-seed0", "track_id": track_id,
        "episode": seed["episode"], "mode": mode,
        "environment_seed": seed["environment_seed"], "policy_seed": seed["policy_seed"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "track_file_sha256": plan["track_file_sha256"][track_id],
        "physics_profile_sha256": plan["physics_profile_sha256"],
        "status": "complete", "metrics_source": "EpisodeMetrics_dataclass_asdict",
        "episode_metrics": serialized_metrics, "terminal": terminal,
        "action_summary": _action_summary(trace_rows), "step_count": len(trace_rows),
        "trace_path": str(trace_path.resolve()), "trace_sha256": file_sha256(trace_path),
        "source_commit": plan["source_commit"], "heldout_access": False,
        "scientific_evidence": False, "training_started": False,
    }
    summary["sha256"] = canonical_sha256(summary)
    return summary


def _verify_episode_summary(path: Path) -> dict:
    summary = _load_json(path)
    _verify_self_hash(summary)
    rows = _load_trace(Path(summary["trace_path"]), summary["trace_sha256"])
    if len(rows) != summary["step_count"]:
        raise ValueError(f"step count mismatch for {path.name}")
    terminal = summary["terminal"]
    last = rows[-1]
    if (terminal["finished"] != last["finished"] or terminal["simulation_timeout"] != last["truncated"]
            or terminal["next_checkpoint_index"] != last["next_checkpoint_index"]):
        raise ValueError(f"terminal summary mismatch for {path.name}")
    if terminal["wall_timeout"] is not False:
        raise ValueError("wall timeout cannot be inferred from a completed episode summary")
    return summary


def _run_plan(plan_path: Path) -> dict:
    plan = _load_json(plan_path)
    _verify_plan(plan)
    output = Path(plan["artifact_root"])
    output.mkdir(parents=True, exist_ok=True)
    episode_dir = output / "episodes"
    episode_dir.mkdir(exist_ok=True)
    reuse = {(row["track_id"], row["episode"], row["mode"]): row
             for row in plan["prior_v2_trace_source"]["reused_cases"]}
    cases = []
    progress_path = output / "progress.jsonl"
    for seed in plan["seed_roster"]:
        for mode in plan["inference_modes"]:
            case_id = _case_id(seed["track_id"], seed["episode"], mode)
            summary_path = episode_dir / f"{case_id}.json"
            key = (seed["track_id"], seed["episode"], mode)
            if key in reuse:
                summary = _reused_summary(plan, reuse[key])
                summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2,
                                                    allow_nan=False) + "\n", encoding="utf-8")
            elif summary_path.exists():
                summary = _verify_episode_summary(summary_path)
            else:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           str(plan_path.resolve()), seed["track_id"], str(seed["episode"]), mode]
                try:
                    child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                           timeout=plan["worker_wall_timeout_seconds"], check=False)
                except subprocess.TimeoutExpired:
                    case = {"case_id": case_id, "status": "wall_timeout",
                            "wall_timeout": True, "simulation_timeout": False,
                            "timeout_seconds": plan["worker_wall_timeout_seconds"]}
                    cases.append(case)
                    with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(case, sort_keys=True) + "\n")
                    continue
                if child.returncode:
                    case = {"case_id": case_id, "status": "worker_failure",
                            "returncode": child.returncode,
                            "stderr_tail": child.stderr[-3000:]}
                    cases.append(case)
                    with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(case, sort_keys=True) + "\n")
                    continue
                summary = _verify_episode_summary(summary_path)
            case = {
                "case_id": case_id, "status": summary["status"],
                "summary_sha256": file_sha256(summary_path),
                "trace_sha256": summary["trace_sha256"],
                "finished": summary["terminal"]["finished"],
                "simulation_timeout": summary["terminal"]["simulation_timeout"],
                "wall_timeout": summary["terminal"]["wall_timeout"],
                "progress": summary["episode_metrics"].get("final_normalized_route_progress"),
                "collision_count": summary["episode_metrics"].get("collision_count"),
                "steps": summary["step_count"],
            }
            cases.append(case)
            with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(case, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps({"completed": len(cases), "total": 40, **case}, allow_nan=False), flush=True)
    summaries = [_verify_episode_summary(episode_dir / f"{case['case_id']}.json")
                 for case in cases if case["status"] in ("complete", "complete_reused_trace")]
    grouped = {}
    for track_id in plan["tracks"]:
        for mode in plan["inference_modes"]:
            cell = [row for row in summaries if row["track_id"] == track_id and row["mode"] == mode]
            grouped[f"{track_id}:{mode}"] = {
                "episodes": len(cell),
                "finishes": sum(row["terminal"]["finished"] for row in cell),
                "finish_rate": sum(row["terminal"]["finished"] for row in cell) / len(cell) if cell else None,
                "simulation_timeouts": sum(row["terminal"]["simulation_timeout"] for row in cell),
                "wall_timeouts": sum(row["terminal"]["wall_timeout"] for row in cell),
                "collisions": sum(row["episode_metrics"]["collision_count"] for row in cell),
                "mean_progress": sum(row["episode_metrics"]["final_normalized_route_progress"]
                                      for row in cell) / len(cell) if cell else None,
                "mean_steering": sum(row["action_summary"]["mean_steering"] for row in cell) / len(cell)
                                  if cell else None,
                "mean_throttle": sum(row["action_summary"]["mean_throttle"] for row in cell) / len(cell)
                                  if cell else None,
                "mean_brake": sum(row["action_summary"]["mean_brake"] for row in cell) / len(cell)
                              if cell else None,
                "final_checkpoint_indices": sorted({row["terminal"]["next_checkpoint_index"] for row in cell}),
            }
    report = {
        "schema_version": "f1rl_m5_c0_pure_ppo_diagnostic_report_v1",
        "plan_sha256": plan["sha256"], "source_commit": plan["source_commit"],
        "source_file_sha256": plan["source_file_sha256"], "cases": cases,
        "cell_summary": grouped,
        "complete_cases": len(summaries), "wall_timeout_cases": sum(c["status"] == "wall_timeout" for c in cases),
        "worker_failures": sum(c["status"] == "worker_failure" for c in cases),
        "reused_prior_v2_trace_cases": len(reuse),
        "heldout_access": False, "scientific_evidence": False, "training_started": False,
    }
    report["sha256"] = canonical_sha256(report)
    report_path = output / "M5-C0-10-SEED-REPORT.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-plan", type=Path)
    actions.add_argument("--run-plan", type=Path)
    actions.add_argument("--worker", nargs=4, metavar=("PLAN", "TRACK", "EPISODE", "MODE"))
    args = parser.parse_args()
    if args.freeze_plan:
        plan_path = args.freeze_plan.resolve()
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan = _make_plan(plan_path.parent)
        plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8", newline="\n")
        print(json.dumps({"plan": str(plan_path), "sha256": plan["sha256"],
                          "reused_cases": plan["prior_v2_trace_source"]["reused_case_count"]}))
    elif args.run_plan:
        report = _run_plan(args.run_plan.resolve())
        print(json.dumps({"report_sha256": report["sha256"],
                          "complete_cases": report["complete_cases"],
                          "cell_summary": report["cell_summary"]}, sort_keys=True, allow_nan=False))
        if report["complete_cases"] != 40 or report["wall_timeout_cases"] or report["worker_failures"]:
            raise SystemExit(2)
    else:
        plan_path, track_id, episode, mode = args.worker
        plan = _load_json(Path(plan_path))
        _verify_plan(plan)
        seed = next(row for row in plan["seed_roster"]
                    if row["track_id"] == track_id and row["episode"] == int(episode))
        if mode not in MODES:
            raise ValueError("unknown inference mode")
        output = Path(plan["artifact_root"]) / "episodes" / f"{_case_id(track_id, int(episode), mode)}.json"
        if output.exists():
            raise FileExistsError(output)
        summary = _run_episode(plan, seed, mode)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
