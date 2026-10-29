"""Validation for individual WP3 campaign run artifacts."""
from __future__ import annotations

import json
import random
import struct
from hashlib import sha256
from pathlib import Path

from ..rl.pilot_preflight import TRAIN_IDS
from ._artifacts import _canonical, _inside, _sha256, _strict_json


def _iter_jsonl(path: Path, max_bytes: int):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate telemetry JSON key: {key}")
            result[key] = value
        return result
    with path.open("rb") as stream:
        while True:
            line = stream.readline(max_bytes + 1)
            if not line:
                break
            if len(line) > max_bytes or not line.endswith(b"\n"):
                raise ValueError("training telemetry has an oversized or incomplete JSONL record")
            record = json.loads(line.decode("utf-8"), object_pairs_hook=unique,
                                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            if not isinstance(record, dict):
                raise ValueError("training telemetry JSONL records must be objects")
            yield record


def _validate_training_contact_terminal(row: dict, expected_demo_reward: dict) -> None:
    demo = row.get("demo_training")
    detail = demo.get("physics_failure_detail") if isinstance(demo, dict) else None
    try:
        vector_reward = struct.unpack("=f", struct.pack("=f", float(demo["reward"])))[0]
    except (KeyError, TypeError, ValueError, OverflowError, struct.error):
        vector_reward = None
    if (not isinstance(demo, dict)
            or row.get("terminated") is not True or row.get("truncated") is not False
            or demo.get("physics_failure_code") not in {"unsupported_contact", "verified_obstacle_contact"}
            or not isinstance(detail, str) or not detail.strip()
            or row.get("termination_reason") != detail
            or demo.get("version") != expected_demo_reward.get("version")
            or demo.get("training_corridor_m") != expected_demo_reward.get("training_corridor_m")
            or demo.get("terminal_penalty") != expected_demo_reward.get("terminal_penalty")
            or demo.get("registered_reward") != 0.0
            or demo.get("curriculum_terminal") is not False
            or demo.get("scientific_evidence") is not False
            or not np_isfinite(demo.get("lateral_penalty"))
            or demo.get("lateral_penalty", -1) < 0
            or not np_isfinite(demo.get("reward"))
            or not np_isfinite(row.get("reward"))
            or demo.get("reward") != -demo["lateral_penalty"] - demo["terminal_penalty"]
            or row.get("reward") != vector_reward):
        raise ValueError("training contact terminal transition violates its frozen demo contract")


def validate_training_run(root: Path, run_dir: Path, expected: dict,
                          campaign_manifest: dict, config: dict) -> dict:
    run_manifest, _ = _strict_json(run_dir / "run-manifest.json")
    state, _ = _strict_json(run_dir / "state.json")
    if run_manifest.get("sha256") != sha256(_canonical({k: v for k, v in run_manifest.items() if k != "sha256"})).hexdigest():
        raise ValueError("run manifest self-checksum mismatch")
    if (run_manifest.get("source_commit") != campaign_manifest["source_commit"]
            or run_manifest.get("source_tree_sha256") != campaign_manifest["source_tree_sha256"]
            or run_manifest.get("experiment_manifest_sha256") != campaign_manifest["simulation_only_config_sha256"]
            or run_manifest.get("registration_sha256") != campaign_manifest["registration_sha256"]
            or run_manifest.get("track_manifest_sha256") != campaign_manifest["track_manifest_sha256"]
            or run_manifest.get("training_profile_sha256") != campaign_manifest["training_profile_sha256"]):
        raise ValueError("run source or registered configuration identity differs from campaign")
    if (run_manifest.get("candidate", {}).get("candidate_id") != expected["candidate_id"]
            or any(run_manifest.get("candidate", {}).get(key) != value
                   for key, value in expected["candidate_factors"].items())
            or run_manifest.get("training_seed_bundle") != expected["seed_bundle"]
            or run_manifest.get("split") != "TRAIN" or run_manifest.get("heldout_access") is not False
            or run_manifest.get("scientific_training_authorized") is not False
            or run_manifest.get("training_failure_policy") != config["training"]["physics_failure_policy"]):
        raise ValueError("run candidate, seed, or scientific isolation identity mismatch")
    factors = expected["candidate_factors"]
    corridor = factors["training_corridor_m"]
    expected_demo_reward = {
        "version": "straight_route_stability_demo_v1" if corridor == 2.0
                   else "straight_route_stability_demo_v2",
        "training_corridor_m": corridor,
        "lateral_penalty_per_m2_s": factors["lateral_penalty_per_m2_s"],
        "terminal_penalty": 25.0,
    }
    if run_manifest.get("demo_reward") != expected_demo_reward:
        raise ValueError("run demo reward differs from the frozen candidate contract")
    ppo = run_manifest.get("ppo_config", {})
    if (ppo.get("seed") != expected["seed_bundle"]["ppo"] or ppo.get("device") != "cpu"
            or ppo.get("n_steps") != 256 or ppo.get("batch_size") != 64 or ppo.get("n_epochs") != 10):
        raise ValueError("run PPO policy or hyperparameters differ from the frozen training profile")
    limits = run_manifest.get("limits", {})
    target_steps = config["training"]["timesteps_per_model"]
    if (limits.get("learning_timesteps") != target_steps
            or limits.get("checkpoint_every_timesteps") != config["training"]["checkpoint_every_timesteps"]
            or limits.get("max_wall_seconds") != config["training"]["max_wall_seconds_per_model"]
            or limits.get("device") != "cpu" or limits.get("n_envs") != 1):
        raise ValueError("run budget or resource limits differ from the frozen campaign")
    if (state.get("status") != "learning_completed" or state.get("phase") != "complete"
            or state.get("learning_timesteps") != target_steps
            or state.get("optimizer_steps", 0) <= 0 or state.get("scientific_evidence") is not False):
        raise ValueError("training execution did not complete the exact finite PPO budget")
    telemetry_path = run_dir / "training-telemetry.jsonl"
    telemetry_manifest, _ = _strict_json(run_dir / "training-telemetry-manifest.json")
    records = list(_iter_jsonl(telemetry_path, config["resources"]["max_telemetry_record_bytes"]))
    digest = _sha256(telemetry_path)
    if (telemetry_manifest.get("complete") is not True
            or telemetry_manifest.get("sha256") != digest
            or telemetry_manifest.get("bytes") != telemetry_path.stat().st_size
            or telemetry_manifest.get("events") != len(records)
            or telemetry_manifest.get("step_rows") != target_steps
            or telemetry_manifest.get("update_rows") != target_steps // 256
            or telemetry_manifest.get("optimizer_steps") != state["optimizer_steps"]):
        raise ValueError("training telemetry is incomplete or fails its byte/count/hash manifest")
    runs = [r for r in records if r.get("event_type") == "run"]
    starts = [r for r in records if r.get("event_type") == "training_start"]
    steps = [r for r in records if r.get("event_type") == "step"]
    updates = [r for r in records if r.get("event_type") == "ppo_update"]
    checkpoints = [r for r in records if r.get("event_type") == "checkpoint"]
    if (len(runs) != 1 or len(starts) != 1 or len(steps) != target_steps
            or len(updates) != target_steps // 256):
        raise ValueError("training telemetry event counts do not match the full rollout budget")
    run_event = runs[0]
    if (run_event.get("source_commit") != campaign_manifest["source_commit"]
            or run_event.get("split") != "TRAIN"
            or run_event.get("seed_identity") != expected["seed_bundle"]
            or run_event.get("training_failure_policy") != config["training"]["physics_failure_policy"]
            or run_event.get("reset_contracts") != run_manifest.get("reset_contracts")
            or run_event.get("seed_identity", {}).get("training_reset") != expected["seed_bundle"]["training_reset"]
            or run_event.get("track_ids") != list(TRAIN_IDS)):
        raise ValueError("telemetry run identity is not bound to campaign source/TRAIN seed roster")
    reset_contracts = run_event.get("reset_contracts", {})
    if set(reset_contracts) != set(TRAIN_IDS):
        raise ValueError("telemetry lacks the exact registered reset contracts")
    episode_tracks: dict[int, str] = {}
    track_timestep_counts = {track: 0 for track in TRAIN_IDS}
    episode_summary: dict[int, dict] = {}
    reset_count = 0
    for timestep, row in enumerate(steps, 1):
        if (row.get("schema_version") != "f1rl_training_telemetry_v1"
                or row.get("source_commit") != campaign_manifest["source_commit"]
                or row.get("run_id") != run_event.get("run_id")
                or row.get("global_timestep") != timestep or row.get("env_index") != 0):
            raise ValueError("training transition identity/order/rank/source mismatch")
        transition_seeds = row.get("seed_identity", {})
        if (transition_seeds.get("run") != expected["seed_bundle"]
                or not isinstance(row.get("observation"), list)
                or len(row["observation"]) != len(run_manifest["observation_schema"])
                or not all(np_isfinite(value) for value in row["observation"])
                or not isinstance(row.get("action_sent_to_environment"), list)
                or len(row["action_sent_to_environment"]) != 3
                or not all(np_isfinite(value) for value in row["action_sent_to_environment"])
                or any(value < low or value > high for value, low, high in zip(
                    row["action_sent_to_environment"], run_manifest["action_space"]["low"],
                    run_manifest["action_space"]["high"]))
                or type(row.get("terminated")) is not bool
                or type(row.get("truncated")) is not bool
                or not np_isfinite(row.get("reward"))):
            raise ValueError("training transition observation/action/reward/seed contract is invalid")
        track = row.get("track_id")
        if track not in track_timestep_counts:
            raise ValueError("training transition uses a non-TRAIN track")
        track_timestep_counts[track] += 1
        episode = row.get("episode_index")
        reset = row.get("seed_identity", {}).get("train_reset", {})
        contract = reset_contracts.get(track, {})
        if (type(episode) is not int or row.get("episode_id") != f"{run_event['run_id']}/env-0/episode-{episode}"
                or reset.get("seed") != expected["seed_bundle"]["training_reset"]
                or reset.get("rank") != 0 or reset.get("reset_index") != episode
                or reset.get("track_id") != track or reset.get("version") != contract.get("version")
                or reset.get("configuration_sha256") != contract.get("configuration_sha256")
                or reset.get("world_sha256") != contract.get("world_sha256")):
            raise ValueError("training episode reset provenance differs from declared TRAIN contract")
        from ..rl.train_reset import derive_train_reset_draw_seed
        if reset.get("draw_seed") != derive_train_reset_draw_seed(
                version=contract["version"], seed=contract["seed"], rank=0,
                track_id=track, reset_index=episode):
            raise ValueError("training reset draw seed does not match authoritative derivation")
        previous = episode_tracks.setdefault(episode, track)
        if previous != track:
            raise ValueError("track identity changed inside a training episode")
        data = episode_summary.setdefault(episode, {"track_id": track, "steps": 0,
                                                     "termination_reason": None,
                                                     "terminated": False, "truncated": False,
                                                     "curriculum_terminal": False})
        data["steps"] += 1
        done = row["terminated"] or row["truncated"]
        if done:
            autoreset = row.get("autoreset_identity")
            next_obs = row.get("autoreset_observation")
            next_episode_id = row.get("autoreset_episode_id")
            next_track = (autoreset or {}).get("track_id")
            next_contract = reset_contracts.get(next_track)
            expected_next_index = episode + 1
            from ..rl.train_reset import derive_train_reset_draw_seed
            if (not isinstance(autoreset, dict) or next_contract is None
                    or not isinstance(next_obs, list)
                    or len(next_obs) != len(run_manifest["observation_schema"])
                    or not all(np_isfinite(value) for value in next_obs)
                    or next_episode_id != f"{run_event['run_id']}/env-0/episode-{expected_next_index}"
                    or autoreset.get("seed") != expected["seed_bundle"]["training_reset"]
                    or autoreset.get("rank") != 0
                    or autoreset.get("reset_index") != expected_next_index
                    or autoreset.get("track_id") != next_track
                    or autoreset.get("version") != next_contract.get("version")
                    or autoreset.get("configuration_sha256") != next_contract.get("configuration_sha256")
                    or autoreset.get("world_sha256") != next_contract.get("world_sha256")
                    or autoreset.get("draw_seed") != derive_train_reset_draw_seed(
                        version=next_contract["version"], seed=next_contract["seed"],
                        rank=0, track_id=next_track, reset_index=expected_next_index)):
                raise ValueError("terminal transition lacks a valid automatic reset identity/observation")
        elif any(row.get(key) is not None for key in
                 ("autoreset_identity", "autoreset_observation", "autoreset_episode_id")):
            raise ValueError("nonterminal training transition contains an automatic reset")
        if done:
            if data["termination_reason"] is not None:
                raise ValueError("multiple terminal transitions were recorded for one episode")
            data["termination_reason"] = row.get("termination_reason") or (
                "truncated" if row.get("truncated") else "terminated")
            data["terminated"] = bool(row.get("terminated"))
            data["truncated"] = bool(row.get("truncated"))
            demo = row.get("demo_training") or {}
            data["curriculum_terminal"] = bool(demo.get("curriculum_terminal"))
            if demo.get("physics_abort_terminal") is True:
                _validate_training_contact_terminal(row, expected_demo_reward)
                data["physics_abort_terminal"] = True
            elif demo.get("physics_abort_terminal") not in (None, False):
                raise ValueError("physics-abort terminal marker is malformed")
            reset_count += 1
        elif row.get("termination_reason") is not None:
            raise ValueError("nonterminal transition has a terminal reason")
    for index, row in enumerate(steps[:-1]):
        if row["terminated"] or row["truncated"]:
            next_row = steps[index + 1]
            if (next_row.get("episode_index") != row["episode_index"] + 1
                    or next_row.get("episode_id") != row.get("autoreset_episode_id")
                    or next_row.get("track_id") != row.get("autoreset_identity", {}).get("track_id")
                    or next_row.get("seed_identity", {}).get("train_reset") != row.get("autoreset_identity")
                    or next_row.get("observation") != row.get("autoreset_observation")):
                raise ValueError("terminal transition does not link to the next autoreset episode observation")
    episode_ids = sorted(episode_summary)
    if episode_ids != list(range(len(episode_ids))):
        raise ValueError("training episode indices are not contiguous")
    sampling_seed = run_event["seed_identity"].get("sampling")
    sampler_seed = int.from_bytes(sha256(
        f"f1rl/multi-track-sampling/v1/{sampling_seed}/0".encode("utf-8")
    ).digest(), "big")
    sampler = random.Random(sampler_seed)
    sampled_tracks = [sampler.choice(list(TRAIN_IDS)) for _ in episode_ids]
    if [episode_summary[i]["track_id"] for i in episode_ids] != sampled_tracks:
        raise ValueError("observed episode tracks differ from the registered deterministic scenario sampler")
    for episode in episode_ids[:-1]:
        if episode_summary[episode]["termination_reason"] is None:
            raise ValueError("next training episode began without a terminal transition association")
    automatic_resets_verified = max(0, len(episode_ids) - 1)
    if automatic_resets_verified:
        for episode in episode_ids[:-1]:
            if not episode_summary[episode]["terminated"] and not episode_summary[episode]["truncated"]:
                raise ValueError("episode boundary is missing a terminal/truncation transition")
    initial_hash = starts[0].get("policy_weights_sha256")
    final_hash = updates[-1].get("policy_weights_sha256")
    if (not isinstance(initial_hash, str) or len(initial_hash) != 64
            or not isinstance(final_hash, str) or len(final_hash) != 64
            or initial_hash == final_hash):
        raise ValueError("initial and final PPO policy parameter hashes do not prove learning")
    if len(checkpoints) != 17:
        raise ValueError("checkpoint telemetry must include 16 periodic checkpoints plus final archive")
    periodic = [r for r in checkpoints if r.get("name", "").startswith("checkpoint-")]
    final_events = [r for r in checkpoints if r.get("name") == "model-final.zip"]
    expected_times = list(range(4096, target_steps + 1, 4096))
    if ([r.get("timestep") for r in periodic] != expected_times
            or len(final_events) != 1 or final_events[0].get("timestep") != target_steps):
        raise ValueError("checkpoint event schedule differs from the frozen 4,096-step schedule")
    checkpoint_manifest, _ = _strict_json(run_dir / "checkpoints.json")
    rows = checkpoint_manifest.get("checkpoints", [])
    if [r.get("timesteps") for r in rows] != expected_times:
        raise ValueError("periodic checkpoint manifest is missing or reordered")
    archive_records = []
    for row in rows:
        path = _inside(run_dir, row["path"])
        if path.name.endswith(".pending.zip") or _sha256(path) != row["sha256"]:
            raise ValueError(f"periodic checkpoint SHA or completion state failed: {path.name}")
        archive_records.append({"path": row["path"], "timesteps": row["timesteps"],
                                "sha256": row["sha256"], "size_bytes": path.stat().st_size})
    final_info = state.get("final_model", {})
    final_path = _inside(run_dir, final_info.get("path", ""))
    if (final_path.name != "model-final.zip" or _sha256(final_path) != final_info.get("sha256")
            or final_events[0].get("archive_sha256") != final_info.get("sha256")):
        raise ValueError("final model archive hash does not match the run state/telemetry")
    archive_records.append({"path": "model-final.zip", "timesteps": target_steps,
                            "sha256": final_info["sha256"], "size_bytes": final_path.stat().st_size})
    if list(run_dir.rglob("*.pending.zip")):
        raise ValueError("incomplete pending checkpoint archive exists")
    integrity, _ = _strict_json(run_dir / "training-integrity.json")
    gradient_rows = integrity.get("optimizer_gradient_activity", [])
    if (integrity.get("schema_version") != "f1rl_local_demo_training_integrity_v1"
            or integrity.get("optimizer_steps") != state["optimizer_steps"]
            or len(gradient_rows) != state["optimizer_steps"]
            or integrity.get("finite_gradient_steps") != len(gradient_rows)
            or integrity.get("nonzero_gradient_steps") != len(gradient_rows)
            or integrity.get("policy_weights_finite_at_every_optimizer_step") is not True
            or integrity.get("weights_changed") is not True):
        raise ValueError("per-optimizer gradient/weight integrity evidence is incomplete")
    if (integrity.get("initial_policy_weights_sha256") != starts[0].get("policy_weights_sha256")
            or integrity.get("final_policy_weights_sha256") != updates[-1].get("policy_weights_sha256")):
        raise ValueError("integrity weight hashes differ from the source-bound telemetry")
    for item in gradient_rows:
        if (item.get("finite") is not True or item.get("gradient_l2_norm", 0) <= 0
                or item.get("nonzero_gradient_tensors", 0) <= 0
                or not np_isfinite(item.get("gradient_l2_norm"))):
            raise ValueError("an optimizer step has non-finite or zero gradient evidence")
    checkpoint_policy_hashes = {r["name"]: r["policy_weights_sha256"] for r in checkpoints}
    update_optimizer_steps = [r.get("optimizer_steps") for r in updates]
    if (not all(isinstance(value, int) and value >= 0 for value in update_optimizer_steps)
            or any(b <= a for a, b in zip(update_optimizer_steps, update_optimizer_steps[1:]))
            or update_optimizer_steps[-1] != state["optimizer_steps"]):
        raise ValueError("PPO optimizer telemetry did not increase monotonically to the final count")
    for update in updates:
        metrics = update.get("metrics", {})
        for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss"):
            if not isinstance(metrics.get(key), (int, float)) or not np_isfinite(metrics[key]):
                raise ValueError(f"PPO training metric is missing or non-finite: {key}")
    from ..rl.algorithms.ppo import PPOModel
    for row in archive_records:
        model_path = _inside(run_dir, row["path"])
        loaded = PPOModel.load(model_path, device="cpu")
        if loaded.num_timesteps != row["timesteps"]:
            raise ValueError(f"checkpoint timestep identity mismatch: {row['path']}")
        if not loaded.policy_weights_finite():
            raise ValueError(f"checkpoint contains a non-finite policy parameter: {row['path']}")
        expected_weight_hash = checkpoint_policy_hashes.get(Path(row["path"]).name)
        if expected_weight_hash is None:
            expected_weight_hash = final_events[0]["policy_weights_sha256"]
        if loaded.weights_sha256() != expected_weight_hash:
            raise ValueError(f"checkpoint policy hash differs from telemetry: {row['path']}")
        if row["timesteps"] == target_steps:
            if loaded._model.observation_space.shape != tuple(run_manifest["observation_space"]["shape"]):
                raise ValueError("final checkpoint observation space differs from the registered contract")
            if loaded._model.action_space.shape != tuple(run_manifest["action_space"]["shape"]):
                raise ValueError("final checkpoint action space differs from the registered contract")
    track_episodes = {track: sum(data["track_id"] == track for data in episode_summary.values())
                      for track in TRAIN_IDS}
    lifecycle = {
        "episode_count": len(episode_summary), "terminal_transitions": reset_count,
        "automatic_resets_verified": automatic_resets_verified,
        "natural_terminations": sum(data["terminated"] and not data["curriculum_terminal"]
                                     and not data.get("physics_abort_terminal")
                                     for data in episode_summary.values()),
        "physics_abort_terminations": sum(data.get("physics_abort_terminal", False)
                                           for data in episode_summary.values()),
        "curriculum_terminations": sum(data["curriculum_terminal"] for data in episode_summary.values()),
        "truncations": sum(data["truncated"] for data in episode_summary.values()),
        "track_episode_counts": track_episodes,
        "track_timestep_counts": track_timestep_counts,
    }
    return {
        "schema_version": "f1rl_wp3_model_engineering_validation_v1",
        "status": "ENGINEERING_PASS", "model_id": expected["model_id"],
        "candidate_id": expected["candidate_id"], "seed_index": expected["seed_index"],
        "source_commit": campaign_manifest["source_commit"], "run_id": run_event["run_id"],
        "learning_timesteps": target_steps, "ppo_update_rows": len(updates),
        "optimizer_steps": state["optimizer_steps"],
        "initial_policy_weights_sha256": initial_hash,
        "final_policy_weights_sha256": final_hash,
        "weights_changed": initial_hash != final_hash,
        "policy_weights_finite_at_all_checkpoints": True,
        "telemetry": {"path": "training-telemetry.jsonl", "bytes": telemetry_path.stat().st_size,
                      "sha256": digest, "events": len(records), "step_rows": len(steps),
                      "update_rows": len(updates), "optimizer_steps": state["optimizer_steps"]},
        "checkpoints": archive_records, "lifecycle": lifecycle,
        "episodes": len(episode_summary), "run_manifest_sha256": _sha256(run_dir / "run-manifest.json"),
        "scientific_evidence": False, "heldout_access": False,
    }


def np_isfinite(value: object) -> bool:
    import math
    return isinstance(value, (int, float)) and math.isfinite(value)
