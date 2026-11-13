from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M4_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m4-m5-development-v2")
M5_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-engineering-fix-20261111\attempt-2")
ABLATION_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-improvement-cycle1-20261111\std-ablation")
CHECKPOINT = ABLATION_ROOT.parent / "C-seed0-logstd-scale-0.65.zip"
TRACK_MAP = {
    "slalom-v1": "slalom-v1",
    "ramp-v1": "ramp-v1",
}
STD_SCALE = 0.65
MODES = ("deterministic", "stochastic")
STEP_LIMIT = 500
WALL_TIMEOUT_SECONDS = 120
SOURCE_FILES = (
    "scripts/m5_cycle1_validate.py",
    "scripts/m5_cycle1_ablation.py",
    "scripts/m5_c0_diagnostic.py",
    "scripts/wp7_dual_turn_v4_feasibility.py",
    "f1rl/learning/action.py",
    "f1rl/learning/algorithms/ppo.py",
    "f1rl/learning/episode/__init__.py",
    "f1rl/learning/evaluation/evaluator.py",
    "f1rl/learning/gym_environment.py",
    "f1rl/learning/training/env_factory.py",
    "f1rl/research/wp3_eval_only_coordinator.py",
    "f1rl/research/wp7_protocol.py",
    "f1rl/runtime/physics.py",
)


def summarize_actions(trace_rows: list[dict]) -> dict:
    if not trace_rows:
        raise ValueError("episode action telemetry cannot be empty")
    count = len(trace_rows)
    controls = [row["applied_vehicle_control"] for row in trace_rows]
    return {
        "mean_steering": sum(row["steering"] for row in controls) / count,
        "mean_throttle": sum(row["throttle"] for row in controls) / count,
        "mean_brake": sum(row["brake"] for row in controls) / count,
        "checkpoint_indices": sorted({r["next_checkpoint_index"] for r in trace_rows}),
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def source_identity() -> tuple[str, str, dict[str, str]]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout
    if dirty or len(commit) != 40 or len(tree) != 40:
        raise ValueError("cycle-1 validation requires a clean committed feature branch")
    return commit, sha256(bytes.fromhex(tree)).hexdigest(), {
        name: file_sha256(ROOT / name) for name in SOURCE_FILES
    }


def make_plan(output: Path) -> dict:
    commit, tree_sha, source_hashes = source_identity()
    prior = json.loads((ABLATION_ROOT / "RAMP-STD-ABLATION-REPORT.json").read_text())
    if prior["finish_count"] != 10:
        raise ValueError("std 0.5 ablation did not produce 10 Ramp finishes")
    seed_rows = [json.loads(line) for line in (M4_ROOT / "evaluation/episodes.jsonl").read_text().splitlines()
                 if line.strip()]
    roster = []
    for track_id, seed_track_id in TRACK_MAP.items():
        rows = sorted((row for row in seed_rows if row.get("model_id") == "C-seed0"
                       and row.get("track_id") == seed_track_id), key=lambda row: row["episode"])
        if len(rows) != 10 or [row["episode"] for row in rows] != list(range(10)):
            raise ValueError(f"expected the same ten predeclared seeds for {seed_track_id}")
        if (len({r["environment_seed"] for r in rows}) != 10
                or len({r["policy_seed"] for r in rows}) != 10):
            raise ValueError(f"M5 seed roster for {seed_track_id} is not varied")
        for row in rows:
            roster.append({"track_id": track_id, "seed_source_track_id": seed_track_id,
                           "episode": row["episode"], "environment_seed": row["environment_seed"],
                           "policy_seed": row["policy_seed"]})
    plan = {
        "schema_version": "f1rl_m5_cycle1_scaled_std_validation_plan_v2",
        "status": "TRAIN_DEVELOPMENT_PURE_PPO_VALIDATION",
        "source_commit": commit,
        "source_tree_sha256": tree_sha,
        "source_file_sha256": source_hashes,
        "candidate_checkpoint_path": str(CHECKPOINT),
        "candidate_checkpoint_sha256": file_sha256(CHECKPOINT),
        "parent_checkpoint_sha256": "600e6eb3a781af66e34ea56f8f65e20f46d4a4318a77aeebceceacef3ffa7fd4",
        "policy_change": f"only PPO learned log_std shifted by log({STD_SCALE}); no gradient updates",
        "reference_std050_checkpoint_sha256": prior["candidate_checkpoint_sha256"],
        "tracks": list(TRACK_MAP),
        "track_file_sha256": {track: file_sha256(ROOT / "assets/tracks/training" / f"{track}.json")
                              for track in TRACK_MAP},
        "m5_seed_roster_sha256": file_sha256(M4_ROOT / "evaluation/episodes.jsonl"),
        "ramp_ablation_report_sha256": file_sha256(ABLATION_ROOT / "RAMP-STD-ABLATION-REPORT.json"),
        "physics_profile_sha256": "c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37",
        "action_contract": "steering-throttle-brake-v2",
        "mode": "Pure PPO; no Auto Drive, Hybrid, action override or action clipping outside policy contract",
        "seed_roster": roster,
        "inference_modes": list(MODES),
        "episodes_per_track_per_mode": 10,
        "simulation_step_limit": STEP_LIMIT,
        "worker_wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "artifact_root": str(output.resolve()),
        "training_started": False,
        "heldout_access": False,
        "scientific_evidence": False,
    }
    plan["sha256"] = canonical_sha256(plan)
    return plan


def verify_plan(plan: dict) -> None:
    if plan.get("sha256") != canonical_sha256(
        {key: value for key, value in plan.items() if key != "sha256"}
    ):
        raise ValueError("cycle-1 validation plan self-hash mismatch")
    commit, tree_sha, sources = source_identity()
    if (plan["source_commit"] != commit or plan["source_tree_sha256"] != tree_sha
            or plan["source_file_sha256"] != sources):
        raise ValueError("cycle-1 source identity changed after plan freeze")
    if file_sha256(CHECKPOINT) != plan["candidate_checkpoint_sha256"]:
        raise ValueError("scaled-std candidate checkpoint changed")
    if file_sha256(M4_ROOT / "evaluation/episodes.jsonl") != plan["m5_seed_roster_sha256"]:
        raise ValueError("M5 fixed seed roster changed")
    for track, digest in plan["track_file_sha256"].items():
        if file_sha256(ROOT / "assets/tracks/training" / f"{track}.json") != digest:
            raise ValueError(f"track changed after plan freeze: {track}")


def run_episode(plan: dict, seed: dict, mode: str, output: Path) -> dict:
    import numpy as np

    from f1rl.learning.action import BrakingPolicyAction, map_policy_action
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.episode import EpisodeConfig
    from f1rl.learning.evaluation.evaluator import evaluate_episode
    from f1rl.learning.training.env_factory import F1RLEnvFactory
    from scripts.m5_c0_diagnostic import serialize_episode_metrics, terminal_summary
    from scripts.wp7_dual_turn_v4_feasibility import build_factory as build_turn_factory

    track = seed["track_id"]
    if track in ("right-turn-train-v2", "left-turn-train-v2"):
        factory, _ = build_turn_factory(track)
    else:
        from f1rl.research.wp3_eval_only_coordinator import (
            FULL06_RUNTIME_PROFILE, apply_runtime_profile,
        )
        from f1rl.research.wp7_protocol import build_protocol, prepare_evaluation_factory

        protocol = build_protocol(ROOT, profile="spatial_rotating_owner_v3")
        factory, _ = prepare_evaluation_factory(
            ROOT, protocol, "multitrack-seed0", track, "nominal", seed["episode"]
        )
        factory, _ = apply_runtime_profile(factory, FULL06_RUNTIME_PROFILE)
    factory = replace(factory, episode_config=EpisodeConfig(plan["simulation_step_limit"]))
    if (factory.physics_config.sha256 != plan["physics_profile_sha256"]
            or factory.action_contract.version != plan["action_contract"]):
        raise ValueError("Physics V4 or action contract differs from frozen validation plan")

    model = PPOModel.load(plan["candidate_checkpoint_path"], device="cpu")
    if (model.num_timesteps != 65536
            or file_sha256(CHECKPOINT) != plan["candidate_checkpoint_sha256"]):
        raise ValueError("candidate policy/checkpoint identity is invalid")
    PPOModel.seed_policy_rng(seed["policy_seed"])
    latest_observation = None
    feature_names = list(factory.observation_encoder.feature_names)

    class Policy:
        def predict(self, observation, *, deterministic=True):
            nonlocal latest_observation
            latest_observation = observation.copy()
            return model.predict(observation, deterministic=deterministic)

    case_id = f"C-seed0__{track}__ep{seed['episode']:02d}__{mode}__std065"
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
                if not np.allclose(raw, [mapped.steering, mapped.throttle, mapped.brake], atol=0, rtol=0):
                    raise ValueError("raw Pure PPO action differs from applied contract action")
                info = event["info"]
                telemetry = info.get("telemetry", {})
                race = info.get("race", {})
                row = {
                    "schema_version": "f1rl_m5_cycle1_validation_step_v1",
                    "case_id": case_id, "track_id": track, "episode": seed["episode"],
                    "mode": mode, "environment_seed": seed["environment_seed"],
                    "policy_seed": seed["policy_seed"], "step": event["step"],
                    "sim_time_s": event["elapsed_time"],
                    "observation": [float(x) for x in latest_observation],
                    "observation_features": feature_names,
                    "raw_policy_action": raw,
                    "applied_vehicle_control": {"steering": float(mapped.steering),
                                                 "throttle": float(mapped.throttle),
                                                 "brake": float(mapped.brake)},
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
                    "terminated": bool(event["terminated"]),
                    "truncated": bool(event["truncated"]),
                }
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False) + "\n")
                trace_rows.append(row)
            metrics = evaluate_episode(
                track_id=track, factory=factory, policy=Policy(),
                reset_seed=seed["environment_seed"], deterministic=(mode == "deterministic"),
                step_observer=observe,
            )
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        F1RLEnvFactory.create = old_create

    summary = {
        "schema_version": "f1rl_m5_cycle1_validation_episode_v1",
        "case_id": case_id, "track_id": track, "episode": seed["episode"],
        "mode": mode, "environment_seed": seed["environment_seed"],
        "policy_seed": seed["policy_seed"],
        "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
        "candidate_checkpoint_sha256": plan["candidate_checkpoint_sha256"],
        "episode_metrics": serialize_episode_metrics(metrics),
        "action_summary": summarize_actions(trace_rows),
        "terminal": terminal_summary(metrics, trace_rows[-1]),
        "trace_path": str(trace.resolve()), "trace_sha256": file_sha256(trace),
        "step_count": len(trace_rows), "source_commit": plan["source_commit"],
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
        for mode in MODES:
            case_id = f"C-seed0__{seed['track_id']}__ep{seed['episode']:02d}__{mode}__std065"
            summary_path = output / "episodes" / f"{case_id}.json"
            child = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(plan_path),
                 seed["track_id"], str(seed["episode"]), mode],
                cwd=ROOT, capture_output=True, text=True,
                timeout=plan["worker_wall_timeout_seconds"], check=False,
            )
            if child.returncode != 0 or not summary_path.is_file():
                raise RuntimeError(f"validation worker failed for {case_id}: {child.stderr[-2000:]}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("sha256") != canonical_sha256(
                {key: value for key, value in summary.items() if key != "sha256"}
            ):
                raise ValueError(f"episode summary hash mismatch: {case_id}")
            cases.append(summary)
    cells = {}
    for track in plan["tracks"]:
        for mode in MODES:
            group = [c for c in cases if c["track_id"] == track and c["mode"] == mode]
            cells[f"{track}:{mode}"] = {
                "episodes": len(group), "finishes": sum(c["terminal"]["finished"] for c in group),
                "mean_progress": sum(c["episode_metrics"]["final_normalized_route_progress"]
                                      for c in group) / len(group),
                "simulation_timeouts": sum(c["terminal"]["simulation_timeout"] for c in group),
                "collisions": sum(c["episode_metrics"]["collision_count"] for c in group),
                "mean_steps": sum(c["step_count"] for c in group) / len(group),
                "mean_speed_mps": sum(c["episode_metrics"]["mean_speed"] for c in group) / len(group),
                "mean_throttle": sum(c["action_summary"]["mean_throttle"] for c in group) / len(group),
                "mean_brake": sum(c["action_summary"]["mean_brake"] for c in group) / len(group),
                "mean_steering": sum(c["action_summary"]["mean_steering"] for c in group) / len(group),
            }
    report = {
        "schema_version": "f1rl_m5_cycle1_scaled_std_validation_report_v2",
        "plan_sha256": plan["sha256"], "source_commit": plan["source_commit"],
        "candidate_checkpoint_sha256": plan["candidate_checkpoint_sha256"],
        "cases": cases, "cell_summary": cells,
        "complete_cases": len(cases), "wall_timeouts": 0, "worker_failures": 0,
        "training_started": False, "heldout_access": False,
    }
    report["sha256"] = canonical_sha256(report)
    path = output / "CYCLE1-VALIDATION-REPORT.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-plan", type=Path)
    group.add_argument("--run-plan", type=Path)
    group.add_argument("--worker", nargs=4, metavar=("PLAN", "TRACK", "EPISODE", "MODE"))
    args = parser.parse_args()
    if args.freeze_plan:
        path = args.freeze_plan.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        plan = make_plan(path.parent)
        path.write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        print(json.dumps({"plan_sha256": plan["sha256"], "cases": len(plan["seed_roster"]) * 2}))
    elif args.run_plan:
        report = run_plan(args.run_plan.resolve())
        print(json.dumps({"report_sha256": report["sha256"],
                          "complete_cases": report["complete_cases"],
                          "cell_summary": report["cell_summary"]}, sort_keys=True, allow_nan=False))
    else:
        plan_path, track, episode, mode = args.worker
        plan_path = Path(plan_path).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        verify_plan(plan)
        seed = next(row for row in plan["seed_roster"]
                    if row["track_id"] == track and row["episode"] == int(episode))
        output = Path(plan["artifact_root"])
        case_id = f"C-seed0__{track}__ep{int(episode):02d}__{mode}__std065"
        summary_path = output / "episodes" / f"{case_id}.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        summary = run_episode(plan, seed, mode, output)
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2,
                                           allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
