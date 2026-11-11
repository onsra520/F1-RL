from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACT_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-multicandidate-screening-20261111")
FULL06_ARCHIVE = Path(r"D:\Projects\.archives\F1-RL\pre-cleanup-consolidation-818eedc\evidence-wp3-87face7-raw.zip")
FULL06_ARCHIVE_SHA256 = "144a41a5e2b2f393103321487265a6ba0f98ec1075196e4f796aecf72b3ef2ba"
M4_ROOT = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m4-m5-development-v2")
C0_PARENT = M4_ROOT / "runs/C-seed0/dev-0-14f3f2a9cc2e/model-final.zip"
C0_STD050 = Path(r"D:\Projects\.archives\F1-RL\demo-completion\m5-improvement-cycle1-20261111\C-seed0-logstd-scale-0.5.zip")
TRACKS = (
    "straight-sprint-v1",
    "slalom-v1",
    "right-turn-train-v2",
    "ramp-v1",
    "left-turn-train-v2",
)
HISTORICAL_MODELS = (
    "A-seed0", "A-seed1", "A-seed2", "B-seed0", "B-seed1", "B-seed2",
    "C-seed0", "C-seed1", "C-seed2", "D-seed0", "D-seed1", "D-seed2",
)
FULL06_RESULTS = "docs/research/evidence/wp3-issue51-full06-traindev-20261027/WP3-TRAIN-EVALUATION.json"
FULL06_RUN_MANIFEST = "docs/research/evidence/wp3-issue51-full06-traindev-20261027/RUN-MANIFEST.json"
FULL06_REPORT_PATH = "docs/research/evidence/wp3-issue51-full06-traindev-20261027/WP3-TRAIN-EVALUATION.json"
EXPECTED_M4_C0_SHA256 = "600e6eb3a781af66e34ea56f8f65e20f46d4a4318a77aeebceceacef3ffa7fd4"
EXPECTED_STD050_SHA256 = "b6de69382523b9dc5309a6cffb230afd31f9b0206d3381dff723abc69ebffa86"
EXPECTED_PHYSICS_SHA256 = "c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37"
ACTION_CONTRACT = "steering-throttle-brake-v2"
OBSERVATION_DIM = 16
ACTION_DIM = 3
STEP_LIMIT = 500
WALL_TIMEOUT_SECONDS = 120
WORKER_COUNTS = (1, 2, 4, 8)
CPU_THREADS_PER_WORKER = 1
RESERVED_MEMORY_GB = 8


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
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout
    if dirty or len(commit) != 40 or len(tree) != 40:
        raise ValueError("screening plan requires a clean committed feature branch")
    relevant = (
        "scripts/m5_multicandidate_screening.py",
        "scripts/m5_c0_diagnostic.py",
        "scripts/wp7_dual_turn_v4_feasibility.py",
        "f1rl/learning/action.py",
        "f1rl/learning/algorithms/ppo.py",
        "f1rl/learning/episode/__init__.py",
        "f1rl/learning/evaluation/evaluator.py",
        "f1rl/learning/gym_environment.py",
        "f1rl/learning/observation.py",
        "f1rl/learning/training/env_factory.py",
        "f1rl/learning/rewards/core.py",
        "f1rl/learning/rewards/terms.py",
        "f1rl/research/wp3_eval_only_coordinator.py",
        "f1rl/research/wp7_protocol.py",
        "f1rl/runtime/physics.py",
        "f1rl/sandbox/vehicle.py",
        "configs/experiments/ppo-generalization-five-track-paired-v2.yml",
        "configs/recovery/wp7-c2-five-track-paired-v2.yml",
        "assets/tracks/manifest-five-track-paired-v2.json",
        *(f"assets/tracks/training/{track}.json" for track in TRACKS),
    )
    sources = {name: file_sha256(ROOT / name) for name in relevant}
    return commit, sha256(bytes.fromhex(tree)).hexdigest(), sources


def _historical_evidence() -> tuple[dict, dict, str]:
    if file_sha256(FULL06_ARCHIVE) != FULL06_ARCHIVE_SHA256:
        raise ValueError("historical FULL-06 evidence archive SHA-256 mismatch")
    with ZipFile(FULL06_ARCHIVE) as archive:
        run_manifest = json.loads(archive.read(FULL06_RUN_MANIFEST))
        report_bytes = archive.read(FULL06_REPORT_PATH)
        evaluation = json.loads(report_bytes)
    for document in (run_manifest, evaluation):
        if document.get("sha256") != canonical_sha256(
                {key: value for key, value in document.items() if key != "sha256"}):
            raise ValueError("FULL-06 embedded evidence self-hash mismatch")
    if (run_manifest["run_id"] != "WP3-OWNER-TRAINDEV-20261027-FULL-06"
            or evaluation["status"] != "COMPLETE_96_CELLS_TRAIN_DEVELOPMENT_NOT_SCIENTIFICALLY_SIGNED"
            or evaluation["valid_episodes"] != 5760
            or evaluation["invalid_episodes"] != 0
            or run_manifest["physics_version"] != "spatial_road_edge_partial_lifecycle_v4"):
        raise ValueError("historical FULL-06 provenance is incomplete or incompatible")
    expected = run_manifest["original_final_model_sha256"]
    if set(expected) != set(HISTORICAL_MODELS):
        raise ValueError("FULL-06 does not bind all 12 expected historical checkpoints")
    evaluation["_exact_file_sha256"] = sha256(report_bytes).hexdigest()
    return run_manifest, evaluation, file_sha256(FULL06_ARCHIVE)


def candidate_inventory() -> tuple[list[dict], dict, dict]:
    historical, historical_report, archive_sha = _historical_evidence()
    per_track = {}
    for model_id in HISTORICAL_MODELS:
        per_track[model_id] = {}
        for track in ("straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"):
            rows = [row for row in historical_report["per_cell"]
                    if row["model_id"] == model_id and row["track"] == track]
            per_track[model_id][track] = {
                "attempts": sum(row["attempted"] for row in rows),
                "finishes": sum(row["finishes"] for row in rows),
                "historical_finish_rate": sum(row["finishes"] for row in rows)
                    / max(1, sum(row["attempted"] for row in rows)),
                "physical_crashes": sum(row["outcomes"].get("physical_crash", 0) for row in rows),
                "step_limits": sum(row["outcomes"].get("step_limit", 0) for row in rows),
            }
    candidates = []
    for model_id in HISTORICAL_MODELS:
        path = ROOT / "models" / model_id / "model-final.zip"
        digest = file_sha256(path)
        if digest != historical["original_final_model_sha256"][model_id]:
            raise ValueError(f"historical checkpoint does not match FULL-06: {model_id}")
        candidates.append({
            "candidate_id": f"FULL06-{model_id}", "identity": "historical_full06",
            "source_model_id": model_id, "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": digest, "provenance": "WP3-OWNER-TRAINDEV-20261027-FULL-06",
            "training_source": historical["training_source"],
            "source_head": historical["source_head"],
            "historical_results": historical_report["by_model"][model_id],
            "historical_per_track": per_track[model_id],
        })
    for candidate_id, path, expected, provenance in (
        ("M4-C-seed0-original", C0_PARENT, EXPECTED_M4_C0_SHA256, "M4 65,536-step C-seed0"),
        ("M4-C-seed0-std050", C0_STD050, EXPECTED_STD050_SHA256,
         "M5 Cycle 1 log_std ablation of M4-C-seed0; no gradient update"),
    ):
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(f"new M4 candidate checkpoint SHA-256 mismatch: {candidate_id}")
        candidates.append({
            "candidate_id": candidate_id, "identity": "new_m4_candidate",
            "source_model_id": "C-seed0" if candidate_id == "M4-C-seed0-original" else "C-seed0-std050",
            "checkpoint_path": str(path.resolve()), "checkpoint_sha256": digest,
            "provenance": provenance, "training_source": "M4 C-seed0 parent",
            "source_head": "e3b7f749186d676cbd1a63796e68d0930866cbfe",
            "historical_results": None,
        })
    if len(candidates) != 14 or len({row["checkpoint_sha256"] for row in candidates}) != 14:
        raise ValueError("candidate inventory must contain 14 distinct model identities/checkpoints")
    return candidates, historical, {"archive_sha256": archive_sha,
        "historical_report_sha256": historical_report["_exact_file_sha256"]}


def _build_factory(track_id: str, episode: int):
    from dataclasses import replace as dc_replace

    from f1rl.learning.episode import EpisodeConfig
    from scripts.wp7_dual_turn_v4_feasibility import build_factory as build_turn_factory

    if track_id in ("right-turn-train-v2", "left-turn-train-v2"):
        factory, provenance = build_turn_factory(track_id)
    else:
        from f1rl.research.wp3_eval_only_coordinator import (
            FULL06_RUNTIME_PROFILE, apply_runtime_profile,
        )
        from f1rl.research.wp7_protocol import build_protocol, prepare_evaluation_factory

        protocol = build_protocol(ROOT, profile="spatial_rotating_owner_v3")
        factory, _ = prepare_evaluation_factory(
            ROOT, protocol, "multitrack-seed0", track_id, "nominal", episode
        )
        factory, provenance = apply_runtime_profile(factory, FULL06_RUNTIME_PROFILE)
    factory = dc_replace(factory, episode_config=EpisodeConfig(STEP_LIMIT))
    if factory.physics_config.sha256 != EXPECTED_PHYSICS_SHA256:
        raise ValueError(f"unexpected Physics V4 profile for {track_id}: {factory.physics_config.sha256}")
    if factory.action_contract.version != ACTION_CONTRACT:
        raise ValueError(f"unexpected action contract for {track_id}")
    if len(factory.observation_encoder.feature_names) != OBSERVATION_DIM:
        raise ValueError(f"unexpected observation contract for {track_id}")
    if provenance is None:
        raise ValueError("runtime profile provenance is missing")
    return factory, provenance


def _compatibility_preflight(candidates: list[dict]) -> dict:
    import numpy as np

    from f1rl.learning.action import BrakingPolicyAction, map_policy_action
    from f1rl.learning.algorithms.ppo import PPOModel

    track_rows = {}
    factories = {}
    for track_id in TRACKS:
        factory, provenance = _build_factory(track_id, 0)
        env = factory.create()
        try:
            observation, info = env.reset(seed=20261111)
            track_rows[track_id] = {
                "track_asset_sha256": file_sha256(ROOT / "assets/tracks/training" / f"{track_id}.json"),
                "observation_feature_names": list(factory.observation_encoder.feature_names),
                "observation_space": {"shape": list(env.observation_space.shape),
                                      "dtype": str(env.observation_space.dtype)},
                "action_space": {"shape": list(env.action_space.shape),
                                 "low": env.action_space.low.tolist(),
                                 "high": env.action_space.high.tolist(),
                                 "dtype": str(env.action_space.dtype)},
                "action_contract": factory.action_contract.version,
                "physics_profile_sha256": factory.physics_config.sha256,
                "runtime_profile": provenance,
                "vehicle_config": asdict(factory.vehicle_parameters),
                "vehicle_config_sha256": canonical_sha256(asdict(factory.vehicle_parameters)),
                "reward_terms": [term.name for term in factory.reward_composer.terms],
                "reward_config_sha256": canonical_sha256(asdict(factory.reward_composer)),
                "simulation_config_sha256": canonical_sha256(asdict(factory.simulation_config)),
                "episode_step_limit": factory.episode_config.max_steps,
                "reset_info_physics": info.get("physics", {}).get("configuration", {}),
            }
            factories[track_id] = (factory, observation)
        finally:
            env.close()
    shared = {key: {row[key] for row in track_rows.values()}
              for key in ("vehicle_config_sha256", "reward_config_sha256",
                          "simulation_config_sha256", "physics_profile_sha256", "action_contract")}
    if any(len(values) != 1 for values in shared.values()):
        raise ValueError(f"development track contracts are not uniform: {shared}")

    model_contracts = []
    for candidate in candidates:
        model = PPOModel.load(candidate["checkpoint_path"], device="cpu")
        if model.num_timesteps != 65536:
            raise ValueError(f"unexpected PPO timestep provenance: {candidate['candidate_id']}")
        per_track = []
        for track_id, (factory, observation) in factories.items():
            obs_shape = tuple(model._model.observation_space.shape)
            action_shape = tuple(model._model.action_space.shape)
            compatible = (obs_shape == (OBSERVATION_DIM,) and action_shape == (ACTION_DIM,)
                          and obs_shape == (factory.observation_encoder.size,)
                          and np.array_equal(model._model.action_space.low, np.asarray(factory.action_contract.low))
                          and np.array_equal(model._model.action_space.high, np.asarray(factory.action_contract.high)))
            if not compatible:
                raise ValueError(f"checkpoint space mismatch: {candidate['candidate_id']} / {track_id}")
            PPOModel.seed_policy_rng(431 + len(per_track))
            action = model.predict(observation, deterministic=True)
            raw = np.asarray(action, dtype=np.float32).reshape(-1)
            if raw.shape != (ACTION_DIM,) or not np.isfinite(raw).all():
                raise ValueError(f"checkpoint gives invalid action: {candidate['candidate_id']} / {track_id}")
            mapped = map_policy_action(BrakingPolicyAction(*map(float, raw)))
            applied = np.asarray([mapped.steering, mapped.throttle, mapped.brake], dtype=np.float32)
            if not np.array_equal(raw, applied):
                raise ValueError(f"action map mismatch: {candidate['candidate_id']} / {track_id}")
            per_track.append({"track_id": track_id, "compatible": True,
                              "observation_shape": list(obs_shape), "action_shape": list(action_shape),
                              "finite_action_probe": raw.tolist(), "raw_applied_equal": True})
        model_contracts.append({
            "candidate_id": candidate["candidate_id"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "num_timesteps": model.num_timesteps,
            "observation_shape": list(model._model.observation_space.shape),
            "action_shape": list(model._model.action_space.shape),
            "policy_class": type(model._model.policy).__name__,
            "tracks": per_track,
        })
    return {"tracks": track_rows, "shared_contracts": {
        key: next(iter(values)) for key, values in shared.items()},
        "candidates": model_contracts,
        "right_turn_v1_episode_count": 0,
        "heldout_access": False, "scientific_evidence": False}


def _seed_roster() -> dict:
    rng = random.Random(20261111)
    roster = {track: [] for track in TRACKS}
    used_environment = set()
    used_policy = set()
    for track in TRACKS:
        for episode in range(15):
            env_seed = rng.randrange(1, 2**31 - 1)
            while env_seed in used_environment:
                env_seed = rng.randrange(1, 2**31 - 1)
            policy_seed = rng.randrange(1, 2**31 - 1)
            while policy_seed in used_policy:
                policy_seed = rng.randrange(1, 2**31 - 1)
            used_environment.add(env_seed)
            used_policy.add(policy_seed)
            roster[track].append({"episode": episode, "environment_seed": env_seed,
                                  "policy_seed": policy_seed})
    return roster


def freeze_plan(output: Path) -> dict:
    commit, tree_sha, source_hashes = source_identity()
    candidates, historical, archive_metadata = candidate_inventory()
    compatibility = _compatibility_preflight(candidates)
    roster = _seed_roster()
    plan = {
        "schema_version": "f1rl_m5_multicandidate_screening_plan_v1",
        "campaign_id": "M5-MULTICANDIDATE-SCREENING-20261111",
        "status": "TRAIN_DEVELOPMENT_ENGINEERING_SCREENING",
        "source_commit": commit, "source_tree_sha256": tree_sha,
        "source_file_sha256": source_hashes,
        "artifact_root": str(output.resolve()),
        "historical_full06": {
            "archive_path": str(FULL06_ARCHIVE), "archive_sha256": archive_metadata["archive_sha256"],
            "run_id": historical["run_id"], "source_head": historical["source_head"],
            "training_source": historical["training_source"],
            "physics_version": historical["physics_version"],
            "report_sha256": archive_metadata["historical_report_sha256"],
        },
        "candidates": candidates,
        "tracks": list(TRACKS), "track_compatibility": compatibility,
        "seed_root": 20261111, "shared_seed_roster": roster,
        "stage_a": {"episodes_per_candidate_per_track": 4,
                    "indices": [0, 1, 2, 3], "mode": "stochastic",
                    "purpose": "broad pilot; not a standalone rejection threshold"},
        "worker_benchmark": {"candidate_id": "M4-C-seed0-std050",
                             "episodes_per_track": 4, "worker_counts": list(WORKER_COUNTS),
                             "cpu_threads_per_worker": CPU_THREADS_PER_WORKER,
                             "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
                             "memory_reserve_gb": RESERVED_MEMORY_GB},
        "stage_b": {"fixed_stochastic_indices": list(range(10)),
                    "fresh_development_indices": list(range(10, 15)),
                    "deterministic_diagnostic_indices": [0],
                    "shortlist_rule": {
                        "always_include": ["FULL06-A-seed2", "FULL06-B-seed1", "FULL06-C-seed1",
                                           "M4-C-seed0-original", "M4-C-seed0-std050"],
                        "also_include_top_n": 3,
                        "rank_by": ["minimum_track_finish_rate", "mean_track_finish_rate",
                                    "lower_physical_crash_rate", "lower_step_limit_rate",
                                    "mean_route_progress"],
                        "pilot_is_not_a_final_rejection_gate": True,
                    },
                    "workers": "fastest stable worker count selected by preregistered benchmark"},
        "episode_step_limit": STEP_LIMIT, "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
        "inference": "Pure PPO stochastic; no Auto Drive, Hybrid, action override or reward mutation",
        "thread_limits": {"torch": 1, "omp": 1, "mkl": 1, "openblas": 1},
        "right_turn_v1_used_for_new_episodes": False,
        "heldout_access": False, "final_acceptance_tracks_used": False,
        "ppo_training_started": False,
    }
    plan["sha256"] = canonical_sha256(plan)
    return plan


def verify_plan(plan: dict) -> None:
    if plan.get("sha256") != canonical_sha256(
            {key: value for key, value in plan.items() if key != "sha256"}):
        raise ValueError("screening plan self-hash mismatch")
    commit, tree, sources = source_identity()
    if (plan["source_commit"] != commit or plan["source_tree_sha256"] != tree
            or plan["source_file_sha256"] != sources):
        raise ValueError("source identity changed after campaign plan freeze")
    if file_sha256(FULL06_ARCHIVE) != plan["historical_full06"]["archive_sha256"]:
        raise ValueError("FULL-06 archive changed after plan freeze")
    for candidate in plan["candidates"]:
        if file_sha256(Path(candidate["checkpoint_path"])) != candidate["checkpoint_sha256"]:
            raise ValueError(f"candidate checkpoint changed: {candidate['candidate_id']}")
    for track in TRACKS:
        path = ROOT / "assets/tracks/training" / f"{track}.json"
        digest = plan["track_compatibility"]["tracks"][track]["track_asset_sha256"]
        if file_sha256(path) != digest:
            raise ValueError(f"development track asset changed: {track}")


def _episode_id(stage: str, candidate: str, track: str, index: int, mode: str,
                worker_count: int | None = None) -> str:
    candidate_slug = candidate.lower().replace("-", "_")
    suffix = f"_w{worker_count}" if worker_count is not None else ""
    return f"{stage}__{candidate_slug}__{track}__ep{index:02d}__{mode}{suffix}"


def _worker_episode(plan: dict, candidate: dict, track_id: str, seed: dict,
                    mode: str, output_dir: Path, case_id: str) -> dict:
    import numpy as np
    import torch

    from f1rl.learning.action import BrakingPolicyAction, map_policy_action
    from f1rl.learning.algorithms.ppo import PPOModel
    from f1rl.learning.episode import EpisodeConfig
    from f1rl.learning.evaluation.evaluator import evaluate_episode
    from f1rl.learning.training.env_factory import F1RLEnvFactory
    from scripts.m5_c0_diagnostic import serialize_episode_metrics, terminal_summary

    torch.set_num_threads(CPU_THREADS_PER_WORKER)
    model_load_started = time.monotonic()
    model = PPOModel.load(candidate["checkpoint_path"], device="cpu")
    model_load_seconds = time.monotonic() - model_load_started
    if (model.num_timesteps != 65536 or file_sha256(Path(candidate["checkpoint_path"]))
            != candidate["checkpoint_sha256"]):
        raise ValueError("loaded checkpoint provenance changed")
    PPOModel.seed_policy_rng(seed["policy_seed"])
    factory, runtime_profile = _build_factory(track_id, seed["episode"])
    factory = replace(factory, episode_config=EpisodeConfig(STEP_LIMIT))
    feature_names = list(factory.observation_encoder.feature_names)
    latest_observation = None

    class RecordingPolicy:
        def predict(self, observation, *, deterministic=True):
            nonlocal latest_observation
            latest_observation = observation.copy()
            return model.predict(observation, deterministic=deterministic)

    trace_path = output_dir / "steps" / f"{case_id}.jsonl"
    summary_path = output_dir / "episodes" / f"{case_id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    old_create = F1RLEnvFactory.create

    def create_with_telemetry(self, *, reset_rank=0, reset_sequence=0):
        env = old_create(self, reset_rank=reset_rank, reset_sequence=reset_sequence)
        env.set_training_telemetry_enabled(True)
        return env

    F1RLEnvFactory.create = create_with_telemetry
    episode_started = time.monotonic()
    try:
        with trace_path.open("x", encoding="utf-8", newline="\n", buffering=1) as stream:
            def observe(event):
                raw = [float(value) for value in event["action"]]
                mapped = map_policy_action(BrakingPolicyAction(*raw))
                applied = {"steering": float(mapped.steering), "throttle": float(mapped.throttle),
                           "brake": float(mapped.brake)}
                if not np.array_equal(np.asarray(raw, dtype=np.float32),
                                      np.asarray([applied["steering"], applied["throttle"],
                                                  applied["brake"]], dtype=np.float32)):
                    raise ValueError("raw policy action differs from applied contract action")
                info = event["info"]
                telemetry = info.get("telemetry", {})
                race = info.get("race", {})
                row = {
                    "schema_version": "f1rl_m5_multicandidate_step_v1",
                    "case_id": case_id, "candidate_id": candidate["candidate_id"],
                    "track_id": track_id, "episode": seed["episode"], "mode": mode,
                    "environment_seed": seed["environment_seed"],
                    "policy_seed": seed["policy_seed"], "step": event["step"],
                    "sim_time_s": event["elapsed_time"],
                    "observation": [float(x) for x in latest_observation],
                    "observation_features": feature_names,
                    "raw_policy_action": raw, "applied_vehicle_control": applied,
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
                    "finished": bool(event["finished"]), "terminated": bool(event["terminated"]),
                    "truncated": bool(event["truncated"]),
                }
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False) + "\n")
                trace_rows.append(row)

            metrics = evaluate_episode(
                track_id=track_id, factory=factory, policy=RecordingPolicy(),
                reset_seed=seed["environment_seed"], deterministic=(mode == "deterministic"),
                step_observer=observe,
            )
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        F1RLEnvFactory.create = old_create

    if not trace_rows:
        raise ValueError("episode completed without any action trace")
    controls = [row["applied_vehicle_control"] for row in trace_rows]
    count = len(controls)
    steering = [row["steering"] for row in controls]
    summary = {
        "schema_version": "f1rl_m5_multicandidate_episode_v1",
        "case_id": case_id, "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["checkpoint_sha256"],
        "track_id": track_id, "episode": seed["episode"], "mode": mode,
        "environment_seed": seed["environment_seed"], "policy_seed": seed["policy_seed"],
        "episode_metrics": serialize_episode_metrics(metrics),
        "terminal": terminal_summary(metrics, trace_rows[-1]),
        "action_summary": {
            "mean_steering": sum(steering) / count,
            "mean_throttle": sum(c["throttle"] for c in controls) / count,
            "mean_brake": sum(c["brake"] for c in controls) / count,
            "throttle_brake_overlap_fraction": sum(
                c["throttle"] > 0.05 and c["brake"] > 0.05 for c in controls) / count,
            "steering_sign_flips": sum(a * b < 0 for a, b in zip(steering, steering[1:])),
            "checkpoint_indices": sorted({row["next_checkpoint_index"] for row in trace_rows}),
            "max_speed_mps": max(float(row.get("vehicle_speed") or 0.0) for row in trace_rows),
            "raw_applied_equivalent": True,
        },
        "runtime_profile": runtime_profile,
        "model_load_seconds": model_load_seconds,
        "episode_seconds": time.monotonic() - episode_started,
        "step_count": len(trace_rows), "trace_path": str(trace_path.resolve()),
        "trace_sha256": file_sha256(trace_path),
        "source_commit": plan["source_commit"], "heldout_access": False,
        "scientific_evidence": False, "ppo_training_started": False,
    }
    summary["sha256"] = canonical_sha256(summary)
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _memory_snapshot() -> dict:
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"memory_load_pct": int(status.dwMemoryLoad),
            "total_physical_bytes": int(status.ullTotalPhys),
            "available_physical_bytes": int(status.ullAvailPhys)}


def _cpu_times() -> tuple[int, int, int]:
    import ctypes
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                                  ctypes.byref(user)):
        raise OSError("GetSystemTimes failed")
    conv = lambda item: (item.dwHighDateTime << 32) | item.dwLowDateTime
    return conv(idle), conv(kernel), conv(user)


def _process_rss_bytes(pid: int) -> int:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return 0
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    try:
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def run_workload(plan: dict, tasks: list[dict], output_dir: Path, workers: int,
                 timeout: int) -> tuple[list[dict], dict]:
    if workers not in WORKER_COUNTS:
        raise ValueError(f"worker count must be one of {WORKER_COUNTS}")
    available = _memory_snapshot()["available_physical_bytes"]
    if available < RESERVED_MEMORY_GB * 1024**3:
        raise MemoryError(f"less than {RESERVED_MEMORY_GB} GiB RAM available before run")
    lock = threading.Lock()
    active_pids: set[int] = set()
    started = time.monotonic()
    last_cpu = _cpu_times()
    samples = []
    results = []

    def save_failure(task: dict, status: str, detail: dict) -> dict:
        row = {"schema_version": "f1rl_m5_multicandidate_episode_failure_v1",
               "case_id": task["case_id"], "candidate_id": task["candidate_id"],
               "track_id": task["track_id"], "mode": task["mode"],
               "seed": task["seed"], "status": status, **detail,
               "source_commit": plan["source_commit"], "heldout_access": False,
               "ppo_training_started": False}
        row["sha256"] = canonical_sha256(row)
        path = output_dir / task["output_group"] / "episodes" / f"{task['case_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
        return row

    def run_task(task: dict) -> dict:
        candidate = next(row for row in plan["candidates"]
                         if row["candidate_id"] == task["candidate_id"])
        case_id = task["case_id"]
        case_dir = output_dir / task["output_group"]
        summary_path = case_dir / "episodes" / f"{case_id}.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        child_env = os.environ.copy()
        child_env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                          "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                          "F1RL_TASK_LAUNCHED_MONOTONIC": repr(time.monotonic())})
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   str(plan_path_for(plan)), case_id, task["candidate_id"], task["track_id"],
                   str(task["seed"]["episode"]), task["mode"], str(case_dir)]
        launch_time = time.monotonic()
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        with lock:
            active_pids.add(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode != 0 or not summary_path.is_file():
                failure_path = save_failure(task, "worker_failure", {
                    "returncode": process.returncode, "wall_seconds": time.monotonic() - launch_time,
                    "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-4000:]})
                return {"case_id": case_id, "candidate_id": task["candidate_id"],
                        "track_id": task["track_id"], "mode": task["mode"],
                        "status": "worker_failure", "returncode": process.returncode,
                        "wall_seconds": time.monotonic() - launch_time,
                        "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-4000:],
                        "summary_path": str((case_dir / "episodes" / f"{case_id}.json").resolve()),
                        "trace_path": None, "failure_sha256": failure_path["sha256"]}
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("sha256") != canonical_sha256(
                    {key: value for key, value in summary.items() if key != "sha256"}):
                raise ValueError(f"episode summary self-hash mismatch: {case_id}")
            return {"case_id": case_id, "candidate_id": task["candidate_id"],
                    "track_id": task["track_id"], "mode": task["mode"],
                    "status": "complete", "finished": summary["terminal"]["finished"],
                    "worker_startup_seconds": summary["worker_startup_seconds"],
                    "wall_seconds": time.monotonic() - launch_time,
                    "step_count": summary["step_count"],
                    "trace_sha256": summary["trace_sha256"],
                    "summary_sha256": summary["sha256"],
                    "summary_path": str(summary_path.resolve()),
                    "trace_path": summary["trace_path"], "stdout_tail": stdout[-1000:]}
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            failure_path = save_failure(task, "wall_timeout", {
                "timeout_seconds": timeout, "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-4000:]})
            return {"case_id": case_id, "candidate_id": task["candidate_id"],
                    "track_id": task["track_id"], "mode": task["mode"],
                    "status": "wall_timeout", "returncode": None,
                    "wall_seconds": time.monotonic() - launch_time,
                    "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-4000:],
                    "summary_path": str((case_dir / "episodes" / f"{case_id}.json").resolve()),
                    "trace_path": None, "failure_sha256": failure_path["sha256"]}
        finally:
            with lock:
                active_pids.discard(process.pid)

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for task in tasks:
            futures[executor.submit(run_task, task)] = task
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    results.append(future.result())
                except Exception as exc:
                    task = futures[future]
                    failure = save_failure(task, "worker_failure", {"error": repr(exc)})
                    results.append({"case_id": task["case_id"],
                                    "candidate_id": task["candidate_id"],
                                    "track_id": task["track_id"], "mode": task["mode"],
                                    "status": "worker_failure", "error": repr(exc),
                                    "failure_sha256": failure["sha256"],
                                    "summary_path": str((output_dir / task["output_group"] / "episodes"
                                                          / f"{task['case_id']}.json").resolve()),
                                    "trace_path": None})
            now_cpu = _cpu_times()
            idle = now_cpu[0] - last_cpu[0]
            total = sum(now_cpu[1:]) - sum(last_cpu[1:])
            cpu_pct = 0.0 if total <= 0 else 100.0 * (total - idle) / total
            last_cpu = now_cpu
            memory = _memory_snapshot()
            with lock:
                rss = sum(_process_rss_bytes(pid) for pid in active_pids)
            samples.append({"elapsed_seconds": time.monotonic() - started,
                            "host_cpu_utilization_pct": cpu_pct,
                            "active_worker_rss_bytes": rss, **memory})
    results.sort(key=lambda row: row["case_id"])
    duration = time.monotonic() - started
    transitions = sum(row.get("step_count", 0) for row in results)
    complete = sum(row["status"] == "complete" for row in results)
    report = {
        "workers": workers, "episodes": len(tasks), "complete_episodes": complete,
        "worker_crashes": sum(row["status"] == "worker_failure" for row in results),
        "wall_timeouts": sum(row["status"] == "wall_timeout" for row in results),
        "wall_seconds": duration,
        "episodes_per_second": complete / duration if duration else 0.0,
        "transitions": transitions,
        "transitions_per_second": transitions / duration if duration else 0.0,
        "time_to_first_result_seconds": min((r["wall_seconds"] for r in results
                                               if "wall_seconds" in r), default=None),
        "mean_worker_startup_seconds": (
            sum(r["worker_startup_seconds"] for r in results if "worker_startup_seconds" in r)
            / max(1, sum("worker_startup_seconds" in r for r in results))),
        "mean_host_cpu_utilization_pct": (
            sum(r["host_cpu_utilization_pct"] for r in samples) / max(1, len(samples))),
        "peak_active_worker_rss_bytes": max((r["active_worker_rss_bytes"] for r in samples), default=0),
        "min_available_physical_bytes": min((r["available_physical_bytes"] for r in samples),
                                             default=available),
        "mean_memory_load_pct": sum(r["memory_load_pct"] for r in samples) / max(1, len(samples)),
        "resource_samples": samples, "results": results,
    }
    return results, report


def plan_path_for(plan: dict) -> Path:
    return Path(plan["plan_path"]).resolve()


def _task(case_id: str, candidate_id: str, track: str, seed: dict,
          mode: str, output_group: str) -> dict:
    return {"case_id": case_id, "candidate_id": candidate_id, "track_id": track,
            "seed": seed, "mode": mode, "output_group": output_group}


def benchmark(plan: dict) -> dict:
    verify_plan(plan)
    candidate_id = plan["worker_benchmark"]["candidate_id"]
    output = Path(plan["artifact_root"]) / "worker-benchmark"
    rows = []
    for workers in WORKER_COUNTS:
        tasks = []
        for track in TRACKS:
            for seed in plan["shared_seed_roster"][track][:4]:
                case_id = _episode_id("benchmark", candidate_id, track,
                                      seed["episode"], "stochastic", workers)
                tasks.append(_task(case_id, candidate_id, track, seed, "stochastic",
                                   f"workers-{workers}"))
        results, run = run_workload(plan, tasks, output, workers, WALL_TIMEOUT_SECONDS)
        row = {key: value for key, value in run.items() if key != "results"}
        row["results"] = results
        rows.append(row)
    keys = {r["case_id"].rsplit("_w", 1)[0] for r in rows[0]["results"]}
    hashes_by_worker = {}
    for row in rows:
        hashes_by_worker[str(row["workers"])] = {
            item["case_id"].rsplit("_w", 1)[0]: item.get("trace_sha256")
            for item in row["results"] if item["status"] == "complete"
        }
    reference = hashes_by_worker[str(WORKER_COUNTS[0])]
    reproducible = all(hashes_by_worker[str(worker)] == reference for worker in WORKER_COUNTS)
    eligible = [row for row in rows if row["complete_episodes"] == 20
                and row["worker_crashes"] == 0 and row["wall_timeouts"] == 0
                and row["min_available_physical_bytes"] >= RESERVED_MEMORY_GB * 1024**3]
    if not eligible:
        raise RuntimeError("no worker configuration completed stably within memory reserve")
    selected = max(eligible, key=lambda row: (row["episodes_per_second"], -row["workers"]))
    report = {"schema_version": "f1rl_m5_multicandidate_worker_benchmark_v1",
              "runs": rows, "reproducible_trace_hashes_across_worker_counts": reproducible,
              "reproducibility_case_count": len(keys), "selected_worker_count": selected["workers"],
              "selection_rule": "fastest throughput among complete/no-crash/no-timeout runs preserving 8GiB available RAM",
              "torch_threads_per_worker": CPU_THREADS_PER_WORKER,
              "heldout_access": False, "ppo_training_started": False}
    report["sha256"] = canonical_sha256(report)
    path = output / "WORKER-BENCHMARK-REPORT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return report


def stage_a(plan: dict, workers: int) -> dict:
    verify_plan(plan)
    tasks = []
    for candidate in plan["candidates"]:
        for track in TRACKS:
            for seed in plan["shared_seed_roster"][track][:4]:
                case_id = _episode_id("stagea", candidate["candidate_id"], track,
                                      seed["episode"], "stochastic")
                tasks.append(_task(case_id, candidate["candidate_id"], track, seed,
                                   "stochastic", "stage-a"))
    output = Path(plan["artifact_root"])
    results, run = run_workload(plan, tasks, output, workers, WALL_TIMEOUT_SECONDS)
    report = {"schema_version": "f1rl_m5_multicandidate_stage_a_report_v1",
              "plan_sha256": plan["sha256"], "workers": workers,
              "run": {key: value for key, value in run.items() if key != "results"},
              "results": results, "cell_summary": _summarize_cells(results, "stage-a"),
              "historical_full06_context": {c["candidate_id"]: c["historical_results"]
                                            for c in plan["candidates"] if c["historical_results"]},
              "small_pilot_warning": "Four stochastic episodes per candidate/track are broad screening only; do not reject historic candidates solely on this sample.",
              "heldout_access": False, "ppo_training_started": False}
    report["sha256"] = canonical_sha256(report)
    path = output / "stage-a" / "STAGE-A-REPORT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return report


def _summarize_cells(results: list[dict], output_group: str) -> dict:
    complete = []
    for result in results:
        if result.get("status") == "complete":
            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            complete.append(summary)
    cells = {}
    candidate_ids = sorted({row["candidate_id"] for row in results})
    for candidate in candidate_ids:
        for track in TRACKS:
            rows = [row for row in complete if row["candidate_id"] == candidate
                    and row["track_id"] == track and row["mode"] == "stochastic"]
            cells[f"{candidate}:{track}"] = {
                "episodes": len(rows), "planned_episodes": 4,
                "finishes": sum(r["terminal"]["finished"] for r in rows),
                "finish_rate": sum(r["terminal"]["finished"] for r in rows) / 4,
                "worker_failures": sum(r["candidate_id"] == candidate and r["track_id"] == track
                                        and r["status"] == "worker_failure" for r in results),
                "wall_timeouts": sum(r["candidate_id"] == candidate and r["track_id"] == track
                                     and r["status"] == "wall_timeout" for r in results),
                "physical_crashes": sum(r["episode_metrics"]["collision_count"] > 0 for r in rows),
                "step_limits": sum(r["terminal"]["simulation_timeout"] for r in rows),
                "mean_progress": sum(r["episode_metrics"]["final_normalized_route_progress"]
                                      for r in rows) / max(1, len(rows)),
                "mean_steps": sum(r["step_count"] for r in rows) / max(1, len(rows)),
                "mean_speed_mps": sum(r["episode_metrics"]["mean_speed"] for r in rows)
                                  / max(1, len(rows)),
                "mean_steering": sum(r["action_summary"]["mean_steering"] for r in rows)
                                 / max(1, len(rows)),
                "mean_throttle": sum(r["action_summary"]["mean_throttle"] for r in rows)
                                 / max(1, len(rows)),
                "mean_brake": sum(r["action_summary"]["mean_brake"] for r in rows)
                              / max(1, len(rows)),
                "mean_throttle_brake_overlap": sum(
                    r["action_summary"]["throttle_brake_overlap_fraction"] for r in rows)
                    / max(1, len(rows)),
                "correct_checkpoint_final": bool(any(
                    r["terminal"]["finished"] for r in rows) and all(
                    r["terminal"]["finished"] and r["terminal"]["next_checkpoint_index"] >= 2
                    for r in rows if r["terminal"]["finished"])),
                "raw_applied_equivalent": all(r["action_summary"]["raw_applied_equivalent"]
                                               for r in rows),
            }
    return cells


def select_shortlist(plan: dict, stage_a_report: dict) -> dict:
    if stage_a_report.get("plan_sha256") != plan["sha256"]:
        raise ValueError("Stage A report does not match frozen plan")
    summaries = _summarize_cells(stage_a_report["results"], "stage-a")
    metrics = {}
    for candidate in [row["candidate_id"] for row in plan["candidates"]]:
        values = [summaries[f"{candidate}:{track}"] for track in TRACKS]
        metrics[candidate] = {
            "minimum_track_finish_rate": min(value["finish_rate"] for value in values),
            "mean_track_finish_rate": sum(value["finish_rate"] for value in values) / len(values),
            "physical_crash_rate": sum(value["physical_crashes"] for value in values)
                                  / max(1, sum(value["episodes"] for value in values)),
            "step_limit_rate": sum(value["step_limits"] for value in values)
                               / max(1, sum(value["episodes"] for value in values)),
            "mean_route_progress": sum(value["mean_progress"] for value in values) / len(values),
        }
    top = sorted(metrics, key=lambda model: (
        -metrics[model]["minimum_track_finish_rate"],
        -metrics[model]["mean_track_finish_rate"],
        metrics[model]["physical_crash_rate"], metrics[model]["step_limit_rate"],
        -metrics[model]["mean_route_progress"], model,
    ))[:3]
    always = plan["stage_b"]["shortlist_rule"]["always_include"]
    selected = list(dict.fromkeys([*always, *top]))
    result = {"schema_version": "f1rl_m5_multicandidate_shortlist_v1",
              "plan_sha256": plan["sha256"], "stage_a_report_sha256": stage_a_report["sha256"],
              "selection_rule": plan["stage_b"]["shortlist_rule"],
              "stage_a_metrics": metrics, "pilot_top_three": top,
              "finalists": selected, "finalist_count": len(selected),
              "small_pilot_warning": "Shortlist only. Four-seed pilot is not a rejection or acceptance test.",
              "heldout_access": False, "ppo_training_started": False}
    result["sha256"] = canonical_sha256(result)
    path = Path(plan["artifact_root"]) / "stage-b" / "STAGE-B-SHORTLIST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return result


def stage_b(plan: dict, shortlist: dict, workers: int) -> dict:
    verify_plan(plan)
    if shortlist.get("plan_sha256") != plan["sha256"]:
        raise ValueError("shortlist does not match frozen plan")
    tasks = []
    roster = plan["shared_seed_roster"]
    for candidate in shortlist["finalists"]:
        for track in TRACKS:
            for index in plan["stage_b"]["fixed_stochastic_indices"]:
                seed = roster[track][index]
                case_id = _episode_id("stageb_fixed", candidate, track, index, "stochastic")
                tasks.append(_task(case_id, candidate, track, seed, "stochastic", "stage-b"))
            for index in plan["stage_b"]["fresh_development_indices"]:
                seed = roster[track][index]
                case_id = _episode_id("stageb_fresh", candidate, track, index, "stochastic")
                tasks.append(_task(case_id, candidate, track, seed, "stochastic", "stage-b"))
            index = plan["stage_b"]["deterministic_diagnostic_indices"][0]
            seed = roster[track][index]
            case_id = _episode_id("stageb_diag", candidate, track, index, "deterministic")
            tasks.append(_task(case_id, candidate, track, seed, "deterministic", "stage-b"))
    output = Path(plan["artifact_root"])
    results, run = run_workload(plan, tasks, output, workers, WALL_TIMEOUT_SECONDS)
    stochastic = _summarize_stage_b(results, plan)
    ranking = _rank_candidates(stochastic)
    report = {"schema_version": "f1rl_m5_multicandidate_stage_b_report_v1",
              "plan_sha256": plan["sha256"], "shortlist_sha256": shortlist["sha256"],
              "workers": workers, "run": {key: value for key, value in run.items() if key != "results"},
              "results": results, "scorecard": stochastic, "ranking": ranking,
              "deterministic_diagnostics_are_not_acceptance": True,
              "heldout_access": False, "ppo_training_started": False}
    report["sha256"] = canonical_sha256(report)
    path = output / "stage-b" / "STAGE-B-REPORT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return report


def _summarize_stage_b(results: list[dict], plan: dict) -> dict:
    complete = []
    for result in results:
        if result.get("status") == "complete":
            complete.append(json.loads(Path(result["summary_path"]).read_text(encoding="utf-8")))
    scorecard = {}
    candidates = sorted({row["candidate_id"] for row in results})
    for candidate in candidates:
        scorecard[candidate] = {}
        for track in TRACKS:
            all_rows = [r for r in complete if r["candidate_id"] == candidate
                        and r["track_id"] == track]
            fixed = [r for r in all_rows if r["case_id"].startswith("stageb_fixed")
                     and r["mode"] == "stochastic"]
            fresh = [r for r in all_rows if r["case_id"].startswith("stageb_fresh")
                     and r["mode"] == "stochastic"]
            deterministic = [r for r in all_rows if r["mode"] == "deterministic"]
            def stats(rows: list[dict], planned: int, prefix: str) -> dict:
                failed = sum(r["candidate_id"] == candidate and r["track_id"] == track
                             and r["case_id"].startswith(prefix)
                             and r["status"] in ("worker_failure", "wall_timeout") for r in results)
                return {"episodes": len(rows), "planned_episodes": planned,
                        "finishes": sum(r["terminal"]["finished"] for r in rows),
                        "finish_rate": sum(r["terminal"]["finished"] for r in rows) / planned,
                        "worker_failures_or_timeouts": failed,
                        "physical_crashes": sum(r["episode_metrics"]["collision_count"] > 0 for r in rows),
                        "step_limits": sum(r["terminal"]["simulation_timeout"] for r in rows),
                        "mean_progress": sum(r["episode_metrics"]["final_normalized_route_progress"]
                                              for r in rows) / max(1, len(rows)),
                        "mean_steps": sum(r["step_count"] for r in rows) / max(1, len(rows)),
                        "mean_speed_mps": sum(r["episode_metrics"]["mean_speed"] for r in rows)
                                          / max(1, len(rows)),
                        "mean_throttle": sum(r["action_summary"]["mean_throttle"] for r in rows)
                                         / max(1, len(rows)),
                        "mean_brake": sum(r["action_summary"]["mean_brake"] for r in rows)
                                      / max(1, len(rows)),
                        "mean_steering": sum(r["action_summary"]["mean_steering"] for r in rows)
                                         / max(1, len(rows)),
                        "mean_pedal_overlap": sum(r["action_summary"]["throttle_brake_overlap_fraction"]
                                                   for r in rows) / max(1, len(rows)),
                        "raw_applied_equivalent": all(r["action_summary"]["raw_applied_equivalent"]
                                                       for r in rows)}
            scorecard[candidate][track] = {
                "fixed_stochastic_10": stats(fixed, 10, "stageb_fixed"),
                "fresh_stochastic_5": stats(fresh, 5, "stageb_fresh"),
                "deterministic_diagnostic": stats(deterministic, 1, "stageb_diag"),
            }
    return scorecard


def _rank_candidates(scorecard: dict) -> list[dict]:
    rows = []
    for candidate, tracks in scorecard.items():
        fixed = [value["fixed_stochastic_10"] for value in tracks.values()]
        fresh = [value["fresh_stochastic_5"] for value in tracks.values()]
        combined = [
            {"finish_rate": (a["finishes"] + b["finishes"]) / max(1, a["episodes"] + b["episodes"]),
             "crashes": a["physical_crashes"] + b["physical_crashes"],
             "timeouts": a["step_limits"] + b["step_limits"],
             "episodes": a["episodes"] + b["episodes"],
             "progress": (a["mean_progress"] + b["mean_progress"]) / 2}
            for a, b in zip(fixed, fresh)
        ]
        rates = [row["finish_rate"] for row in combined]
        total_episodes = sum(row["episodes"] for row in combined)
        total_crashes = sum(row["crashes"] for row in combined)
        total_timeouts = sum(row["timeouts"] for row in combined)
        rows.append({"candidate_id": candidate,
                     "minimum_track_finish_rate": min(rates),
                     "mean_track_finish_rate": sum(rates) / len(rates),
                     "physical_crash_rate": total_crashes / max(1, total_episodes),
                     "step_limit_rate": total_timeouts / max(1, total_episodes),
                     "mean_route_progress": sum(r["progress"] for r in combined) / len(combined),
                     "per_track_combined_finish_rate": {track: round(item["finish_rate"], 4)
                                                        for track, item in zip(TRACKS, combined)},
                     "fixed_and_fresh_reported_separately": True})
    return sorted(rows, key=lambda row: (
        -row["minimum_track_finish_rate"], -row["mean_track_finish_rate"],
        row["physical_crash_rate"], row["step_limit_rate"],
        -row["mean_route_progress"], row["candidate_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-plan", type=Path)
    group.add_argument("--benchmark", type=Path)
    group.add_argument("--stage-a", type=Path)
    group.add_argument("--select-shortlist", type=Path)
    group.add_argument("--stage-b", type=Path)
    group.add_argument("--worker", nargs=7, metavar=("PLAN", "CASE", "CANDIDATE", "TRACK",
                                                      "EPISODE", "MODE", "OUTPUT"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.freeze_plan:
        path = args.freeze_plan.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        plan = freeze_plan(path.parent)
        plan["plan_path"] = str(path)
        plan["sha256"] = canonical_sha256({k: v for k, v in plan.items() if k != "sha256"})
        path.write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
        inv = {"schema_version": "f1rl_m5_multicandidate_inventory_v1",
               "candidates": plan["candidates"], "compatibility": plan["track_compatibility"],
               "sha256": canonical_sha256({"candidates": plan["candidates"],
                                            "compatibility": plan["track_compatibility"]})}
        (path.parent / "CANDIDATE-INVENTORY-AND-COMPATIBILITY.json").write_text(
            json.dumps(inv, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"plan_sha256": plan["sha256"], "candidates": len(plan["candidates"]),
                          "tracks": len(plan["tracks"]), "cases_stage_a": len(plan["candidates"]) * 20,
                          "right_turn_v1_episodes": 0}))
    elif args.worker:
        plan_path, case_id, candidate_id, track, episode, mode, output = args.worker
        plan_path = Path(plan_path).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        verify_plan(plan)
        candidate = next(row for row in plan["candidates"] if row["candidate_id"] == candidate_id)
        seed = next(row for row in plan["shared_seed_roster"][track]
                    if row["episode"] == int(episode))
        started = time.monotonic()
        summary = _worker_episode(plan, candidate, track, seed, mode, Path(output), case_id)
        launch = float(os.environ["F1RL_TASK_LAUNCHED_MONOTONIC"])
        summary["worker_startup_seconds"] = max(0.0, started - launch)
        summary["sha256"] = canonical_sha256({k: v for k, v in summary.items() if k != "sha256"})
        summary_path = Path(output) / "episodes" / f"{case_id}.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2,
                                           allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"case_id": case_id, "sha256": summary["sha256"],
                          "trace_sha256": summary["trace_sha256"]}))
    else:
        plan_path = (args.benchmark or args.stage_a or args.select_shortlist or args.stage_b).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["plan_path"] = str(plan_path)
        if args.benchmark:
            report = benchmark(plan)
            print(json.dumps({"selected_worker_count": report["selected_worker_count"],
                              "reproducible": report["reproducible_trace_hashes_across_worker_counts"]}))
        elif args.stage_a:
            report = stage_a(plan, args.workers or 1)
            print(json.dumps({"complete": report["run"]["complete_episodes"],
                              "total": report["run"]["episodes"]}))
        elif args.select_shortlist:
            report_path = Path(plan["artifact_root"]) / "stage-a" / "STAGE-A-REPORT.json"
            report = select_shortlist(plan, json.loads(report_path.read_text(encoding="utf-8")))
            print(json.dumps({"finalists": report["finalists"], "sha256": report["sha256"]}))
        else:
            shortlist_path = Path(plan["artifact_root"]) / "stage-b" / "STAGE-B-SHORTLIST.json"
            report = stage_b(plan, json.loads(shortlist_path.read_text(encoding="utf-8")),
                             args.workers or 1)
            print(json.dumps({"ranking": report["ranking"], "sha256": report["sha256"]}))


if __name__ == "__main__":
    main()
