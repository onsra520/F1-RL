"""Paired Pure PPO development evaluation for M5 Cycle 2 checkpoints."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from m5_multicandidate_screening import _build_factory
from m5_c0_diagnostic import canonical_sha256, file_sha256

TRACKS = ("straight-sprint-v1", "slalom-v1", "right-turn-train-v2", "ramp-v1",
          "left-turn-train-v2")
STEP_LIMIT = 500
WORKER_TIMEOUT = 120


def freeze_plan(output: Path, campaign_plan_path: Path) -> dict:
    campaign = json.loads(campaign_plan_path.read_text(encoding="utf-8"))
    roster = campaign["shared_seed_roster"]
    if tuple(roster) != TRACKS:
        if set(roster) != set(TRACKS):
            raise ValueError("campaign seed roster differs from the registered five-track set")
    for track in TRACKS:
        rows = roster[track]
        if (len(rows) != 15 or [row["episode"] for row in rows] != list(range(15))
                or len({row["environment_seed"] for row in rows}) != 15
                or len({row["policy_seed"] for row in rows}) != 15):
            raise ValueError(f"paired seed roster is incomplete or duplicated: {track}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    plan = {
        "schema_version": "f1rl_m5_cycle2_paired_evaluation_plan_v1",
        "source_commit": commit, "source_tree": tree,
        "campaign_plan_path": str(campaign_plan_path.resolve()),
        "campaign_plan_sha256": file_sha256(campaign_plan_path),
        "tracks": list(TRACKS), "track_asset_sha256": {
            track: file_sha256(ROOT / "assets/tracks/training" / f"{track}.json")
            for track in TRACKS},
        "seed_roster": {track: roster[track] for track in TRACKS},
        "selection_indices": {"fixed": list(range(10)), "fresh_development": list(range(10, 15))},
        "slalom_deterministic_indices": list(range(10)),
        "inference_modes": ["deterministic_slalom", "stochastic_all_tracks"],
        "step_limit": STEP_LIMIT, "worker_wall_timeout_seconds": WORKER_TIMEOUT,
        "heldout_access": False, "scientific_evidence": False,
    }
    plan["sha256"] = canonical_sha256(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return plan


def verify_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("sha256") != canonical_sha256({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("paired evaluation plan self-hash mismatch")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    if plan["source_commit"] != commit or plan["source_tree"] != tree:
        raise ValueError("paired evaluation source changed after plan freeze")
    if file_sha256(Path(plan["campaign_plan_path"])) != plan["campaign_plan_sha256"]:
        raise ValueError("screening seed source changed after plan freeze")
    for track, digest in plan["track_asset_sha256"].items():
        if track not in TRACKS or file_sha256(ROOT / "assets/tracks/training" / f"{track}.json") != digest:
            raise ValueError(f"paired evaluation track asset changed: {track}")
    return plan


def run_episode(checkpoint: Path, plan: dict, track: str, seed: dict, mode: str,
                output: Path) -> dict:
    import numpy as np
    from f1rl.learning.action import BrakingPolicyAction, map_policy_action
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.episode import EpisodeConfig
    from f1rl.learning.evaluation.evaluator import evaluate_episode
    from f1rl.learning.training.env_factory import F1RLEnvFactory

    checkpoint_sha = file_sha256(checkpoint)
    factory, _ = _build_factory(track, seed["episode"])
    from dataclasses import replace
    factory = replace(factory, episode_config=EpisodeConfig(STEP_LIMIT))
    expected_physics = "c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37"
    if (factory.physics_config.sha256 != expected_physics
            or factory.action_contract.version != "steering-throttle-brake-v2"
            or len(factory.observation_encoder.feature_names) != 16):
        raise ValueError("Cycle 2 evaluation environment contract changed")
    model = PPOModel.load(checkpoint, device="cpu")
    if not 65536 <= model.num_timesteps <= 98304:
        raise ValueError("evaluation checkpoint timestep is outside C0 plus Cycle 2 budget")
    PPOModel.seed_policy_rng(seed["policy_seed"])
    latest_observation = None
    rows = []
    features = list(factory.observation_encoder.feature_names)
    max_steering = float(factory.vehicle_parameters.max_steering_angle)
    fixed_dt = float(factory.simulation_config.fixed_dt)
    previous_angle = None

    class Policy:
        def predict(self, observation, *, deterministic=True):
            nonlocal latest_observation
            latest_observation = np.asarray(observation).copy()
            return model.predict(observation, deterministic=deterministic)

    original_create = F1RLEnvFactory.create
    def create_with_telemetry(self, *, reset_rank=0, reset_sequence=0):
        env = original_create(self, reset_rank=reset_rank, reset_sequence=reset_sequence)
        env.set_training_telemetry_enabled(True)
        return env

    def observe(event):
        nonlocal previous_angle
        raw = [float(x) for x in np.asarray(event["action"]).reshape(-1)]
        mapped = map_policy_action(BrakingPolicyAction(*raw))
        applied = {"steering": float(mapped.steering), "throttle": float(mapped.throttle),
                   "brake": float(mapped.brake)}
        if not np.allclose(raw, list(applied.values()), atol=1e-7, rtol=0):
            raise ValueError("Pure PPO raw action differs from the registered action mapping")
        angle = applied["steering"] * max_steering
        info = event["info"]
        telemetry = info.get("telemetry", {})
        state = info.get("training_telemetry", {})
        row = {
            "step": event["step"], "observation": [float(x) for x in latest_observation],
            "observation_features": features, "raw_action": raw, "applied_action": applied,
            "commanded_steering_angle_rad": angle,
            "commanded_steering_rate_radps": 0.0 if previous_angle is None else (angle-previous_angle)/fixed_dt,
            "reward": float(event["reward"]), "reward_components": info.get("reward_components", {}),
            "checkpoint_index": int(event["next_checkpoint_index"]),
            "completed_checkpoint_ids": info.get("race", {}).get("completed_checkpoint_ids", []),
            "collision_obstacle_ids": telemetry.get("obstacle_hit_ids", []),
            "position": telemetry.get("vehicle_position"), "velocity": telemetry.get("vehicle_velocity"),
            "speed_mps": telemetry.get("vehicle_speed"), "heading_rad": state.get("heading_rad"),
            "yaw_rate_radps": state.get("yaw_rate_radps"),
            "lateral_route_deviation_m": state.get("lateral_route_deviation_m"),
            "route_progress_m": state.get("route_progress_m"),
            "route_highwater_m": state.get("route_highwater_m"),
            "physical_crash": info.get("physical_crash_terminal"),
            "research_outcome": info.get("research_outcome"),
            "finished": bool(event["finished"]), "terminated": bool(event["terminated"]),
            "truncated": bool(event["truncated"]),
        }
        json.dumps(row, allow_nan=False)
        rows.append(row)
        previous_angle = angle

    F1RLEnvFactory.create = create_with_telemetry
    try:
        metrics = evaluate_episode(track_id=track, factory=factory, policy=Policy(),
                                   reset_seed=seed["environment_seed"],
                                   deterministic=(mode == "deterministic"), step_observer=observe)
    finally:
        F1RLEnvFactory.create = original_create
    candidate_id = checkpoint.stem
    case = f"{candidate_id}__{track}__ep{seed['episode']:02d}__{mode}"
    trace_path = output / "steps" / f"{case}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    metric_dict = asdict(metrics) if is_dataclass(metrics) else vars(metrics)
    terminal_row = rows[-1]
    result = {"case_id": case, "checkpoint_sha256": checkpoint_sha,
              "track_id": track, "episode": seed["episode"], "mode": mode,
              "environment_seed": seed["environment_seed"], "policy_seed": seed["policy_seed"],
              "metrics": metric_dict,
              "terminal": {"finished": bool(metrics.success), "simulation_timeout": bool(metrics.truncated),
                           "reason": ("finish" if metrics.success else
                                     terminal_row["research_outcome"].get("reason")
                                     if isinstance(terminal_row["research_outcome"], dict) else
                                     "physical_crash" if terminal_row["physical_crash"] else
                                     "step_limit" if metrics.truncated else "terminal_failure"),
                           "physical_crash": terminal_row["physical_crash"],
                           "collision_obstacle_ids": terminal_row["collision_obstacle_ids"],
                           "next_checkpoint_index": terminal_row["checkpoint_index"]},
              "action_summary": {
                  "mean_steering": float(np.mean([r["applied_action"]["steering"] for r in rows])),
                  "mean_throttle": float(np.mean([r["applied_action"]["throttle"] for r in rows])),
                  "mean_brake": float(np.mean([r["applied_action"]["brake"] for r in rows])),
                  "mean_speed_mps": float(np.mean([r["speed_mps"] or 0.0 for r in rows])),
              },
              "trace_path": str(trace_path), "trace_sha256": file_sha256(trace_path),
              "step_count": len(rows), "heldout_access": False, "scientific_evidence": False}
    result["sha256"] = canonical_sha256(result)
    return result


def evaluate_checkpoint(plan_path: Path, checkpoint: Path, output: Path,
                        *, deterministic_slalom_only: bool = False) -> dict:
    plan = verify_plan(plan_path)
    checkpoint = checkpoint.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases = []
    roster = plan["seed_roster"]
    selected = []
    for track in TRACKS:
        if deterministic_slalom_only and track != "slalom-v1":
            continue
        for seed in roster[track]:
            modes = ["deterministic"] if deterministic_slalom_only else ["stochastic"]
            for mode in modes:
                selected.append((track, seed, mode))
    if not deterministic_slalom_only:
        for seed in roster["slalom-v1"][:10]:
            selected.append(("slalom-v1", seed, "deterministic"))
    for index, (track, seed, mode) in enumerate(selected):
        mode_arg = mode
        case = f"{checkpoint.stem}__{track}__ep{seed['episode']:02d}__{mode_arg}"
        summary_path = output / "episodes" / f"{case}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, str(Path(__file__).resolve()), "--worker", str(plan_path.resolve()),
                str(checkpoint), track, str(seed["episode"]), mode_arg, str(output.resolve())]
        child = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                               timeout=WORKER_TIMEOUT, check=False)
        if child.returncode != 0 or not summary_path.is_file():
            (output / "failure-diagnostic.txt").write_text(
                f"case={case}\nreturncode={child.returncode}\nstdout={child.stdout[-2000:]}\nstderr={child.stderr[-4000:]}\n",
                encoding="utf-8")
            raise RuntimeError(f"evaluation worker failed: {case}")
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        if row.get("sha256") != canonical_sha256({k: v for k, v in row.items() if k != "sha256"}):
            raise ValueError(f"episode summary hash mismatch: {case}")
        cases.append(row)
        print(json.dumps({"index": index + 1, "count": len(selected), "track": track,
                          "mode": mode, "episode": seed["episode"],
                          "finished": row["terminal"]["finished"]}), flush=True)
    grouped = {}
    for track in TRACKS:
        for mode in (("deterministic",) if deterministic_slalom_only else ("stochastic",)):
            rows = [row for row in cases if row["track_id"] == track and row["mode"] == mode]
            if rows:
                grouped[f"{track}:{mode}"] = {
                    "episodes": len(rows), "finishes": sum(row["terminal"]["finished"] for row in rows),
                    "crashes": sum(row["terminal"]["reason"] == "physical_crash" for row in rows),
                    "step_limits": sum(row["terminal"]["reason"] == "step_limit" for row in rows),
                    "mean_progress": float(np.mean([row["metrics"]["final_normalized_route_progress"] for row in rows])),
                }
    report = {"schema_version": "f1rl_m5_cycle2_checkpoint_evaluation_v1",
              "source_commit": plan["source_commit"], "plan_sha256": plan["sha256"],
              "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
              "deterministic_slalom_only": deterministic_slalom_only,
              "cases": cases, "grouped": grouped, "complete_cases": len(cases),
              "heldout_access": False, "scientific_evidence": False}
    report["sha256"] = canonical_sha256(report)
    (output / "EVALUATION-REPORT.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--freeze-plan", nargs=2, metavar=("CAMPAIGN_PLAN", "OUTPUT"))
    group.add_argument("--evaluate", nargs=3, metavar=("PLAN", "CHECKPOINT", "OUTPUT"))
    group.add_argument("--deterministic-slalom", nargs=3, metavar=("PLAN", "CHECKPOINT", "OUTPUT"))
    parser.add_argument("--worker", nargs=6,
                        metavar=("PLAN", "CHECKPOINT", "TRACK", "EPISODE", "MODE", "OUTPUT"))
    args = parser.parse_args()
    if args.freeze_plan:
        plan = freeze_plan(Path(args.freeze_plan[1]), Path(args.freeze_plan[0]))
        print(json.dumps({"plan_sha256": plan["sha256"]}))
    elif args.evaluate or args.deterministic_slalom:
        values = args.evaluate or args.deterministic_slalom
        report = evaluate_checkpoint(Path(values[0]), Path(values[1]), Path(values[2]),
                                     deterministic_slalom_only=bool(args.deterministic_slalom))
        print(json.dumps({"report_sha256": report["sha256"], "complete_cases": report["complete_cases"],
                          "grouped": report["grouped"]}, sort_keys=True))
    elif args.worker:
        plan_path, checkpoint, track, episode, mode, output = args.worker
        plan = verify_plan(Path(plan_path))
        seed = next(row for row in plan["seed_roster"][track] if row["episode"] == int(episode))
        result = run_episode(Path(checkpoint), plan, track, seed, mode, Path(output))
        path = Path(output) / "episodes" / f"{Path(checkpoint).stem}__{track}__ep{int(episode):02d}__{mode}.json"
        path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        parser.error("choose a plan, evaluation, or worker operation")


if __name__ == "__main__":
    main()
