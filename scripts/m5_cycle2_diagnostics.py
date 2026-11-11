"""Reproduce Pure PPO Slalom behavior with source-bound step evidence."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from math import atan2, hypot, pi
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from m5_multicandidate_screening import _build_factory
from m5_c0_diagnostic import canonical_sha256, file_sha256

PARENT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m4-m5-development-v2\runs\C-seed0\dev-0-14f3f2a9cc2e\model-final.zip")
STD050 = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-improvement-cycle1-20261111\C-seed0-logstd-scale-0.5.zip")
EXPECTED = {
    "parent": "600e6eb3a781af66e34ea56f8f65e20f46d4a4318a77aeebceceacef3ffa7fd4",
    "std050": "b6de69382523b9dc5309a6cffb230afd31f9b0206d3381dff723abc69ebffa86",
}


def signed_path_error(position, navigation, checkpoint_index: int) -> tuple[float, float]:
    points = navigation.points
    stop = (navigation.checkpoint_point_indices[checkpoint_index]
            if checkpoint_index < len(navigation.checkpoint_point_indices) else len(points) - 1)
    best = None
    for index in range(stop):
        first, second = points[index], points[index + 1]
        dx, dz = second.x - first.x, second.z - first.z
        denominator = dx * dx + dz * dz
        fraction = max(0.0, min(1.0, ((position[0] - first.x) * dx
                                     + (position[2] - first.z) * dz) / denominator))
        nearest_x, nearest_z = first.x + fraction * dx, first.z + fraction * dz
        distance = hypot(position[0] - nearest_x, position[2] - nearest_z)
        candidate = (distance, index, first, dx, dz, nearest_x, nearest_z)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return 0.0, 0.0
    _, _, first, dx, dz, nearest_x, nearest_z = best
    lateral = (dx * (position[2] - nearest_z) - dz * (position[0] - nearest_x)) / hypot(dx, dz)
    desired_heading = atan2(dx, dz)
    return lateral, (desired_heading)


def load_roster(path: Path, count: int = 10) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = sorted((row for row in rows if row.get("model_id") == "C-seed0"
                       and row.get("track_id") == "slalom-v1"), key=lambda row: row["episode"])
    if len(selected) < count:
        raise ValueError("registered Slalom seed roster is incomplete")
    selected = selected[:count]
    pairs = {(row["environment_seed"], row["policy_seed"]) for row in selected}
    if (len(pairs) != count or len({row["environment_seed"] for row in selected}) != count
            or len({row["policy_seed"] for row in selected}) != count):
        raise ValueError("Slalom reset and policy seeds must be distinct")
    return [{"episode": index, "environment_seed": row["environment_seed"],
             "policy_seed": row["policy_seed"]} for index, row in enumerate(selected)]


def run_episode(checkpoint: Path, label: str, seed: dict, deterministic: bool,
                output: Path, index: int) -> dict:
    import numpy as np
    from f1rl.learning.action import BrakingPolicyAction, map_policy_action
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.evaluation.evaluator import evaluate_episode
    from f1rl.learning.training.env_factory import F1RLEnvFactory

    digest = file_sha256(checkpoint)
    if digest != EXPECTED[label]:
        raise ValueError(f"{label} baseline SHA mismatch")
    factory, _ = _build_factory("slalom-v1", index)
    model = PPOModel.load(checkpoint, device="cpu")
    if model.num_timesteps != 65536:
        raise ValueError("checkpoint timestep provenance mismatch")
    PPOModel.seed_policy_rng(seed["policy_seed"])
    latest_observation = None
    features = list(factory.observation_encoder.feature_names)
    max_steering_angle = float(factory.vehicle_parameters.max_steering_angle)
    fixed_dt = float(factory.simulation_config.fixed_dt)
    rows = []

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
        raw = [float(value) for value in np.asarray(event["action"]).reshape(-1)]
        mapped = map_policy_action(BrakingPolicyAction(*raw))
        info = event["info"]
        telemetry = info.get("telemetry", {})
        race = info.get("race", {})
        training_state = info.get("training_telemetry", {})
        position = telemetry.get("vehicle_position")
        signed_lateral, desired_heading = signed_path_error(
            position, factory.navigation, int(event["next_checkpoint_index"]))
        heading = training_state.get("heading_rad")
        yaw_error = None if heading is None else (heading - desired_heading + pi) % (2*pi) - pi
        steering_angle = float(mapped.steering) * max_steering_angle
        previous_angle = rows[-1].get("commanded_steering_angle_rad") if rows else None
        row = {
            "step": event["step"], "observation": [float(x) for x in latest_observation],
            "observation_features": features, "raw_action": raw,
            "applied_action": {"steering": float(mapped.steering),
                               "throttle": float(mapped.throttle), "brake": float(mapped.brake)},
            "commanded_steering_angle_rad": steering_angle,
            "commanded_steering_rate_radps": (0.0 if previous_angle is None else
                                               (steering_angle - previous_angle) / fixed_dt),
            "reward": float(event["reward"]), "reward_components": info.get("reward_components", {}),
            "checkpoint_index": int(event["next_checkpoint_index"]),
            "completed_checkpoint_ids": list(race.get("completed_checkpoint_ids", ())),
            "collision_obstacle_ids": list(telemetry.get("obstacle_hit_ids", ())),
            "position": position,
            "velocity": telemetry.get("vehicle_velocity"),
            "speed_mps": telemetry.get("vehicle_speed"),
            "heading_rad": training_state.get("heading_rad"),
            "desired_heading_rad": desired_heading,
            "yaw_error_rad": yaw_error,
            "yaw_rate_radps": training_state.get("yaw_rate_radps"),
            "lateral_route_deviation_m": training_state.get("lateral_route_deviation_m"),
            "signed_lateral_path_error_m": signed_lateral,
            "terminal": {"finished": bool(event["finished"]),
                         "terminated": bool(event["terminated"]),
                         "truncated": bool(event["truncated"]),
                         "physical_crash": info.get("physical_crash_terminal"),
                         "research_outcome": info.get("research_outcome")},
        }
        json.dumps(row, allow_nan=False)
        rows.append(row)

    F1RLEnvFactory.create = create_with_telemetry
    try:
        metrics = evaluate_episode(track_id="slalom-v1", factory=factory, policy=Policy(),
                                   reset_seed=seed["environment_seed"],
                                   deterministic=deterministic, step_observer=observe)
    finally:
        F1RLEnvFactory.create = original_create
    trace = output / "steps" / f"{label}__slalom__{index:02d}__{'det' if deterministic else 'stoch'}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    summary = asdict(metrics) if is_dataclass(metrics) else vars(metrics)
    result = {"checkpoint_label": label, "checkpoint_sha256": digest,
              "environment_seed": seed["environment_seed"], "policy_seed": seed["policy_seed"],
              "deterministic": deterministic, "metrics": summary,
              "terminal": rows[-1]["terminal"], "trace": str(trace),
              "trace_sha256": file_sha256(trace), "step_count": len(rows)}
    result["sha256"] = canonical_sha256(result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--checkpoint", choices=("parent", "std050", "both"), default="both")
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="deterministic")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roster = load_roster(args.seeds, args.count)
    checkpoint_map = {"parent": PARENT, "std050": STD050}
    labels = list(checkpoint_map) if args.checkpoint == "both" else [args.checkpoint]
    modes = [True, False] if args.mode == "both" else [args.mode == "deterministic"]
    cases = []
    for label in labels:
        for deterministic in modes:
            for index, seed in enumerate(roster):
                cases.append(run_episode(checkpoint_map[label], label, seed, deterministic,
                                         args.output, index))
                print(json.dumps({"checkpoint": label, "mode": "det" if deterministic else "stoch",
                                  "index": index, "finished": cases[-1]["metrics"].get("success"),
                                  "steps": cases[-1]["step_count"]}), flush=True)
    report = {"schema_version": "f1rl_m5_cycle2_slalom_diagnostic_v1",
              "source_commit": __import__("subprocess").run(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                  capture_output=True, text=True).stdout.strip(),
              "seed_roster": roster, "cases": cases,
              "heldout_access": False, "scientific_evidence": False}
    report["sha256"] = canonical_sha256(report)
    (args.output / "DIAGNOSTIC-REPORT.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "report_sha256": report["sha256"]}))


if __name__ == "__main__":
    main()
