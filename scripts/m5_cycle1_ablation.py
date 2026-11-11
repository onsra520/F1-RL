from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M4_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m4-m5-development-v2")
CHECKPOINT = M4_ROOT / "runs/C-seed0/dev-0-14f3f2a9cc2e/model-final.zip"
M5_EPISODES = M4_ROOT / "evaluation/episodes.jsonl"
SOURCE_FILES = (
    "scripts/m5_cycle1_ablation.py",
    "scripts/m5_c0_diagnostic.py",
    "scripts/wp7_dual_turn_v4_feasibility.py",
    "f1rl/learning/action.py",
    "f1rl/learning/algorithms/ppo.py",
    "f1rl/learning/episode.py",
    "f1rl/learning/evaluation/evaluator.py",
    "f1rl/learning/gym_environment.py",
    "f1rl/learning/training/env_factory.py",
    "f1rl/research/wp3_eval_only_coordinator.py",
    "f1rl/research/wp7_protocol.py",
    "f1rl/runtime/physics.py",
)
TRACK_MAP = {"ramp-v1": "ramp-v1"}
STD_SCALE = 0.5
STEP_LIMIT = 500
WALL_TIMEOUT_SECONDS = 120


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def scale_policy_std(model: object, scale: float) -> None:
    import torch

    if (type(scale) not in (int, float) or not math.isfinite(scale)
            or not 0.0 < scale < 1.0):
        raise ValueError("stochastic policy standard-deviation scale must be in (0, 1)")
    policy = getattr(getattr(model, "_model", None), "policy", None)
    log_std = getattr(policy, "log_std", None)
    if not isinstance(log_std, torch.nn.Parameter) or log_std.ndim != 1 or not log_std.numel():
        raise ValueError("PPO policy must expose a nonempty learned log_std parameter")
    with torch.no_grad():
        log_std.add_(math.log(scale))
    if not torch.isfinite(log_std).all():
        raise ValueError("scaled PPO log_std contains a non-finite value")


def _self_hash(document: dict) -> None:
    if document.get("sha256") != canonical_sha256(
        {key: value for key, value in document.items() if key != "sha256"}
    ):
        raise ValueError("canonical plan/report self-hash mismatch")


def _source_identity() -> tuple[str, str, dict[str, str]]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT,
                          check=True, capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout
    if dirty or len(commit) != 40 or len(tree) != 40:
        raise ValueError("ablation requires a clean committed feature-branch source tree")
    return commit, sha256(bytes.fromhex(tree)).hexdigest(), {
        name: file_sha256(ROOT / name) for name in SOURCE_FILES
    }


def make_plan(output: Path) -> dict:
    commit, tree_sha, source_hashes = _source_identity()
    m5_rows = [json.loads(line) for line in M5_EPISODES.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    roster = []
    for track_id, seed_track in TRACK_MAP.items():
        rows = sorted((row for row in m5_rows if row.get("model_id") == "C-seed0"
                       and row.get("track_id") == seed_track), key=lambda row: row["episode"])
        if len(rows) != 10 or [row["episode"] for row in rows] != list(range(10)):
            raise ValueError(f"expected ten predeclared seeds for {seed_track}")
        env_seeds = [row["environment_seed"] for row in rows]
        policy_seeds = [row["policy_seed"] for row in rows]
        if len(set(env_seeds)) != 10 or len(set(policy_seeds)) != 10:
            raise ValueError("environment and policy seeds must both vary")
        for row in rows:
            roster.append({"track_id": track_id, "seed_source_track_id": seed_track,
                           "episode": row["episode"],
                           "environment_seed": row["environment_seed"],
                           "policy_seed": row["policy_seed"]})
    plan = {
        "schema_version": "f1rl_m5_cycle1_ramp_std_ablation_plan_v1",
        "status": "TRAIN_DEVELOPMENT_ABLATION_ONLY",
        "source_commit": commit,
        "source_tree_sha256": tree_sha,
        "source_file_sha256": source_hashes,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "checkpoint_timesteps": 65536,
        "m5_seed_roster_sha256": file_sha256(M5_EPISODES),
        "track_sha256": {track: file_sha256(ROOT / "assets/tracks/training" / f"{track}.json")
                         for track in TRACK_MAP},
        "physics_profile_sha256": "c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37",
        "action_contract": "steering-throttle-brake-v2",
        "mode": "stochastic Pure PPO; no action override, Auto Drive, or Hybrid",
        "ablation_factor": "multiply learned policy standard deviation by 0.5 via log_std += log(0.5)",
        "unchanged_factors": ["policy mean/network", "reward", "curriculum", "track", "Physics V4",
                              "observation/action contracts", "environment/policy seeds", "500-step limit"],
        "baseline_evidence": {
            "report_sha256": file_sha256(Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-engineering-fix-20261111\attempt-2\M5-C0-10-SEED-REPORT.json")),
            "track_id": "ramp-v1",
            "stochastic_finish_count": 1,
            "deterministic_finish_count": 10,
        },
        "seed_roster": roster,
        "inference_mode": "stochastic",
        "episodes": 10,
        "simulation_step_limit": STEP_LIMIT,
        "worker_wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "artifact_root": str(output.resolve()),
        "heldout_access": False,
        "scientific_evidence": False,
        "training_started": False,
    }
    plan["sha256"] = canonical_sha256(plan)
    return plan


def verify_plan(plan: dict) -> None:
    _self_hash(plan)
    commit, tree_sha, source_hashes = _source_identity()
    if (plan.get("source_commit") != commit or plan.get("source_tree_sha256") != tree_sha
            or plan.get("source_file_sha256") != source_hashes):
        raise ValueError("ablation source identity changed after plan freeze")
    if file_sha256(CHECKPOINT) != plan["checkpoint_sha256"]:
        raise ValueError("C-seed0 source checkpoint SHA-256 changed")
    if file_sha256(M5_EPISODES) != plan["m5_seed_roster_sha256"]:
        raise ValueError("predeclared M5 seed roster changed")
    for track, digest in plan["track_sha256"].items():
        if file_sha256(ROOT / "assets/tracks/training" / f"{track}.json") != digest:
            raise ValueError(f"TRAIN track asset changed: {track}")


def run_episode(plan: dict, seed: dict, output: Path) -> dict:
    import numpy as np

    from f1rl.learning.action import BrakingPolicyAction, map_policy_action
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.episode import EpisodeConfig
    from f1rl.learning.evaluation.evaluator import evaluate_episode
    from f1rl.learning.training.env_factory import F1RLEnvFactory
    from scripts.m5_c0_diagnostic import (
        serialize_episode_metrics, terminal_summary,
    )
    from f1rl.research.wp3_eval_only_coordinator import FULL06_RUNTIME_PROFILE, apply_runtime_profile
    from f1rl.research.wp7_protocol import build_protocol, prepare_evaluation_factory

    track_id = seed["track_id"]
    protocol = build_protocol(ROOT, profile="spatial_rotating_owner_v3")
    factory, _ = prepare_evaluation_factory(
        ROOT, protocol, "multitrack-seed0", track_id, "nominal", seed["episode"]
    )
    factory, _ = apply_runtime_profile(factory, FULL06_RUNTIME_PROFILE)
    factory = replace(factory, episode_config=EpisodeConfig(plan["simulation_step_limit"]))
    model = PPOModel.load(plan["checkpoint_path"], device="cpu")
    if model.num_timesteps != plan["checkpoint_timesteps"]:
        raise ValueError("C-seed0 checkpoint timestep identity changed")
    scale_policy_std(model, STD_SCALE)
    scaled_checkpoint = output.parent / "C-seed0-logstd-scale-0.5.zip"
    if not scaled_checkpoint.exists():
        model.save(scaled_checkpoint)
    PPOModel.seed_policy_rng(seed["policy_seed"])
    features = list(factory.observation_encoder.feature_names)
    latest_observation = None

    class Policy:
        def predict(self, observation, *, deterministic=True):
            nonlocal latest_observation
            latest_observation = observation.copy()
            return model.predict(observation, deterministic=deterministic)

    case_id = f"C-seed0__{track_id}__ep{seed['episode']:02d}__stochastic__std050"
    trace = output / "steps" / f"{case_id}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    old_create = F1RLEnvFactory.create

    def create_with_telemetry(self, *, reset_rank=0, reset_sequence=0):
        env = old_create(self, reset_rank=reset_rank, reset_sequence=reset_sequence)
        env.set_training_telemetry_enabled(True)
        return env

    F1RLEnvFactory.create = create_with_telemetry
    try:
        with trace.open("x", encoding="utf-8", newline="\n", buffering=1) as stream:
            def observe(event):
                raw = [float(value) for value in event["action"]]
                mapped = map_policy_action(BrakingPolicyAction(*raw))
                info = event["info"]
                telemetry = info.get("telemetry", {})
                race = info.get("race", {})
                row = {
                    "schema_version": "f1rl_m5_cycle1_ablation_step_v1",
                    "case_id": case_id, "track_id": track_id,
                    "episode": seed["episode"], "environment_seed": seed["environment_seed"],
                    "policy_seed": seed["policy_seed"], "step": event["step"],
                    "sim_time_s": event["elapsed_time"],
                    "observation": [float(value) for value in latest_observation],
                    "observation_features": features,
                    "raw_policy_action": raw,
                    "applied_vehicle_control": {"steering": float(mapped.steering),
                                                 "throttle": float(mapped.throttle),
                                                 "brake": float(mapped.brake)},
                    "reward": float(event["reward"]),
                    "reward_components": info.get("reward_components", {}),
                    "next_checkpoint_index": int(event["next_checkpoint_index"]),
                    "completed_checkpoint_ids": list(race.get("completed_checkpoint_ids", ())),
                    "collision_obstacle_ids": list(telemetry.get("obstacle_hit_ids", ())),
                    "vehicle_position": telemetry.get("vehicle_position"),
                    "vehicle_velocity": telemetry.get("vehicle_velocity"),
                    "vehicle_speed": telemetry.get("vehicle_speed"),
                    "finished": bool(event["finished"]),
                    "terminated": bool(event["terminated"]),
                    "truncated": bool(event["truncated"]),
                }
                if not np.allclose(raw, [mapped.steering, mapped.throttle, mapped.brake], atol=0, rtol=0):
                    raise ValueError("action map changed raw PPO action")
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False) + "\n")
                trace_rows.append(row)
            metrics = evaluate_episode(
                track_id=track_id, factory=factory, policy=Policy(),
                reset_seed=seed["environment_seed"], deterministic=False,
                step_observer=observe,
            )
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        F1RLEnvFactory.create = old_create
    summary = {
        "schema_version": "f1rl_m5_cycle1_ablation_episode_v1",
        "case_id": case_id, "track_id": track_id, "episode": seed["episode"],
        "environment_seed": seed["environment_seed"], "policy_seed": seed["policy_seed"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "candidate_checkpoint_sha256": file_sha256(scaled_checkpoint),
        "candidate_checkpoint_path": str(scaled_checkpoint), "std_scale": STD_SCALE,
        "mode": "stochastic", "episode_metrics": serialize_episode_metrics(metrics),
        "terminal": terminal_summary(metrics, trace_rows[-1]),
        "raw_trace_sha256": file_sha256(trace), "trace_path": str(trace.resolve()),
        "trace_steps": len(trace_rows), "source_commit": plan["source_commit"],
        "training_started": False, "heldout_access": False,
    }
    summary["sha256"] = canonical_sha256(summary)
    return summary


def run_plan(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_plan(plan)
    output = Path(plan["artifact_root"])
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    (output / "steps").mkdir(parents=True, exist_ok=True)
    cases = []
    for seed in plan["seed_roster"]:
        case_id = f"C-seed0__{seed['track_id']}__ep{seed['episode']:02d}__stochastic__std050"
        summary_path = output / "episodes" / f"{case_id}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   str(plan_path), seed["track_id"], str(seed["episode"])]
        child = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=plan["worker_wall_timeout_seconds"], check=False)
        if child.returncode != 0 or not summary_path.is_file():
            raise RuntimeError(f"ablation worker failed ({child.returncode}): {child.stderr[-2000:]}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("sha256") != canonical_sha256(
            {key: value for key, value in summary.items() if key != "sha256"}
        ):
            raise ValueError("ablation episode summary self-hash mismatch")
        cases.append(summary)
    report = {
        "schema_version": "f1rl_m5_cycle1_ramp_std_ablation_report_v1",
        "plan_sha256": plan["sha256"], "source_commit": plan["source_commit"],
        "candidate_checkpoint_sha256": cases[0]["candidate_checkpoint_sha256"],
        "cases": cases,
        "finish_count": sum(row["terminal"]["finished"] for row in cases),
        "simulation_timeouts": sum(row["terminal"]["simulation_timeout"] for row in cases),
        "wall_timeouts": 0,
        "mean_progress": sum(row["episode_metrics"]["final_normalized_route_progress"]
                              for row in cases) / len(cases),
        "training_started": False, "heldout_access": False,
    }
    report["sha256"] = canonical_sha256(report)
    report_path = output / "RAMP-STD-ABLATION-REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-plan", type=Path)
    group.add_argument("--run-plan", type=Path)
    group.add_argument("--worker", nargs=3, metavar=("PLAN", "TRACK", "EPISODE"))
    args = parser.parse_args()
    if args.freeze_plan:
        path = args.freeze_plan.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        plan = make_plan(path.parent)
        path.write_text(json.dumps(plan, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"plan_sha256": plan["sha256"], "episodes": len(plan["seed_roster"])}))
    elif args.run_plan:
        report = run_plan(args.run_plan.resolve())
        print(json.dumps({"report_sha256": report["sha256"],
                          "finish_count": report["finish_count"],
                          "simulation_timeouts": report["simulation_timeouts"]}))
        if report["finish_count"] < 8:
            raise SystemExit(2)
    else:
        plan_path, track_id, episode = args.worker
        plan_path = Path(plan_path).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        verify_plan(plan)
        seed = next(row for row in plan["seed_roster"]
                    if row["track_id"] == track_id and row["episode"] == int(episode))
        output = Path(plan["artifact_root"])
        case_id = f"C-seed0__{track_id}__ep{int(episode):02d}__stochastic__std050"
        summary_path = output / "episodes" / f"{case_id}.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        summary = run_episode(plan, seed, output)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                           allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
