"""Bounded local Windows PPO demo; output is never WP7 scientific evidence."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from ..learning.algorithms.ppo import PPOModel, configure_cpu_runtime
from ..learning.pilot_preflight import TRAIN_IDS
from ..learning.training.config import TrainingRunConfig, VectorEnvConfig
from ..learning.training.telemetry import TrainingTelemetryRecorder
from ..learning.training.vector_env import make_multi_track_vector_env
from ..research.simulation_only_protocol import load_simulation_only_manifest
from ..research.wp7_protocol import load_registration, prepare_scenarios
from ..exploratory.reward import apply_training_curriculum, validate as validate_demo_reward
from .admission import LocalDemoStopped, _issue, _revoke, _source_identity


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _save_checkpoint(model: PPOModel, pending: Path, final: Path) -> None:
    try:
        model.save(pending)
        os.replace(pending, final)
    except BaseException as error:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            error.add_note(
                f"Could not remove incomplete checkpoint {pending.name}: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def _action_policy_weights_sha256(model: PPOModel) -> str:
    selected = ((name, value) for name, value in model._model.policy.state_dict().items()
                if name.startswith(("mlp_extractor.policy_net.", "action_net."))
                or name == "log_std")
    digest = sha256()
    names = []
    for name, value in sorted(selected):
        names.append(name)
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    if not names:
        raise RuntimeError("PPO action policy has no fingerprintable parameters")
    return digest.hexdigest()


def _assert_finite_policy_weights(model: PPOModel) -> None:
    import torch

    for name, tensor in model._model.policy.state_dict().items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"non-finite PPO policy weight: {name}")


def _load_smoke_profile(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
        profile = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("LOCAL_DEMO smoke profile could not be loaded") from error
    expected = {
        "schema_version": "f1rl_local_demo_ppo_smoke_v1",
        "candidate": "C", "tuning_seed_index": 0, "split": "TRAIN",
        "timesteps": 256, "rollout_length": 256, "n_envs": 1,
        "rollout_batches": 1, "device": "cpu",
        "tracks": ["straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"],
        "checkpoint_interval": 0, "max_wall_seconds": 1800,
        "final_checkpoint_required": True, "heldout_access": False,
        "scientific_evaluation": False,
    }
    if (not isinstance(profile, dict) or set(profile) != set(expected)
            or any(type(profile[key]) is not type(value) or profile[key] != value
                   for key, value in expected.items())):
        raise ValueError("LOCAL_DEMO smoke profile is invalid or exceeds its fixed scope")
    return profile, sha256(raw_bytes).hexdigest()


def _validate_smoke_telemetry(path: Path, run_id: str) -> dict:
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")))
            json.dumps(event, allow_nan=False)
            if event.get("schema_version") != "f1rl_local_demo_smoke_telemetry_v1":
                raise ValueError("smoke telemetry schema changed")
            events.append(event)
    steps = [event for event in events if event.get("event_type") == "step"]
    starts = [event for event in events if event.get("event_type") == "training_start"]
    updates = [event for event in events if event.get("event_type") == "ppo_update"]
    if (not events or events[0].get("event_type") != "run"
            or events[0].get("run_id") != run_id or len(starts) != 1
            or len(steps) != 256 or len(updates) != 1
            or updates[0].get("ppo_epoch_updates", 0) <= 0):
        raise ValueError("smoke telemetry must contain one identified run, one rollout and one PPO update")
    expected_episode = 0
    expected_autoreset = None
    expected_following_observation = None
    episode_tracks = {}
    observed_tracks = set()
    for index, row in enumerate(steps, start=1):
        if row.get("global_timestep") != index or row.get("env_index") != 0:
            raise ValueError("smoke telemetry timestep order or environment rank changed")
        episode = row.get("episode_index")
        reset = row.get("seed_identity", {}).get("train_reset", {})
        track = row.get("track_id")
        if (episode != expected_episode or reset.get("reset_index") != episode
                or reset.get("seed") != 1000 or reset.get("track_id") != track
                or track not in TRAIN_IDS
                or row.get("episode_id") != f"{run_id}/env-0/episode-{episode}"):
            raise ValueError("smoke transition is attributed to the wrong reset episode")
        if expected_autoreset is not None and (
                reset != expected_autoreset or track != expected_autoreset.get("track_id")
                or row.get("observation") != expected_following_observation):
            raise ValueError("smoke autoreset identity does not match the following transition")
        elif expected_autoreset is None and expected_following_observation is not None and (
                row.get("observation") != expected_following_observation):
            raise ValueError("smoke consecutive transition observation continuity failed")
        expected_autoreset = None
        expected_following_observation = None
        if episode in episode_tracks and episode_tracks[episode] != track:
            raise ValueError("smoke track identity changed within an episode")
        episode_tracks[episode] = track
        observed_tracks.add(track)
        done = bool(row.get("terminated")) or bool(row.get("truncated"))
        auto_identity = row.get("autoreset_identity")
        auto_observation = row.get("autoreset_observation")
        if done:
            expected_episode += 1
            if (not isinstance(auto_identity, dict)
                    or auto_identity.get("reset_index") != expected_episode
                    or auto_identity.get("seed") != 1000
                    or row.get("autoreset_episode_id") != f"{run_id}/env-0/episode-{expected_episode}"
                    or not isinstance(auto_observation, list)):
                raise ValueError("smoke terminal step lost its automatic-reset episode identity")
            expected_autoreset = auto_identity
            expected_following_observation = auto_observation
        elif auto_identity is not None or auto_observation is not None or row.get("autoreset_episode_id") is not None:
            raise ValueError("smoke telemetry reported an autoreset for a nonterminal transition")
        else:
            expected_following_observation = row.get("next_observation")
        if (len(row.get("observation", [])) != 16
                or len(row.get("next_observation", [])) != 16
                or len(row.get("raw_ppo_action", [])) != 3
                or len(row.get("action_sent_to_environment", [])) != 3):
            raise ValueError("smoke transition observation/action schema changed")
    metrics = updates[0].get("metrics", {})
    for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss"):
        if not isinstance(metrics.get(key), (int, float)) or not np.isfinite(metrics[key]):
            raise ValueError(f"smoke PPO metric is missing or non-finite: {key}")
    return {"step_rows": len(steps), "ppo_update_rows": len(updates),
            "initial_policy_weights_sha256": starts[0]["policy_weights_sha256"],
            "updated_policy_weights_sha256": updates[0]["policy_weights_sha256"],
            "ppo_epoch_updates": updates[0]["ppo_epoch_updates"],
            "episodes_started": expected_episode + (1 if not steps[-1].get("terminated")
                                                       and not steps[-1].get("truncated") else 0),
            "episodes_terminated_or_truncated": expected_episode,
            "observed_track_ids": sorted(observed_tracks),
            "automatic_resets_verified": sum(
                bool(row.get("terminated") or row.get("truncated")) for row in steps),
            "finite_loss_metrics": True}


def _candidate_reward(manifest: dict, candidate_id: str) -> tuple[dict, dict]:
    row = next((item for item in manifest["candidates"] if item["candidate_id"] == candidate_id), None)
    if row is None:
        raise ValueError("candidate must be A, B, C or D")
    reward = validate_demo_reward({
        "version": row["demo_contract"],
        "training_corridor_m": row["training_corridor_m"],
        "lateral_penalty_per_m2_s": row["lateral_penalty_per_m2_s"],
        "terminal_penalty": 25.0,
    })
    return row, reward


def _preflight(root: Path, candidate_id: str, tuning_seed_index: int, *, profile: str = "tuning"):
    if os.name != "nt":
        raise ValueError("LOCAL_DEMO is supported on local Windows only")
    if type(tuning_seed_index) is not int or not 0 <= tuning_seed_index < 3:
        raise ValueError("LOCAL_DEMO may use only one of the three reserved tuning seed bundles")
    if not isinstance(profile, str) or profile not in {"tuning", "smoke"}:
        raise ValueError("LOCAL_DEMO profile must be tuning or smoke")
    smoke_profile = smoke_profile_sha256 = None
    if profile == "smoke":
        smoke_profile, smoke_profile_sha256 = _load_smoke_profile(
            root / "configs/research/local-demo-ppo-smoke-v1.json"
        )
        if candidate_id != smoke_profile["candidate"] or tuning_seed_index != smoke_profile["tuning_seed_index"]:
            raise ValueError("LOCAL_DEMO smoke profile requires candidate C and tuning seed index 0")
    source_commit, source_tree_sha256 = _source_identity(root)
    manifest, experiment_sha256 = load_simulation_only_manifest(root)
    candidate, reward = _candidate_reward(manifest, candidate_id)
    registration = load_registration(root, profile="spatial_rotating_owner_v3")
    scenarios = prepare_scenarios(root, registration, seed_index=tuning_seed_index)
    if tuple(scenario.track_id for scenario in scenarios.scenarios) != TRAIN_IDS:
        raise ValueError("LOCAL_DEMO requires the exact four-track TRAIN roster")
    if any(scenario.factory.physics_config is None or not scenario.factory.requires_scientific_admission
           for scenario in scenarios.scenarios):
        raise ValueError("LOCAL_DEMO requires the registered research simulation profile")
    if any(scenario.factory.race_config is None
           or scenario.factory.race_config.hard_corridor_m != 4.0
           for scenario in scenarios.scenarios):
        raise ValueError("LOCAL_DEMO requires the unchanged registered 4m evaluation race contract")
    base = TrainingRunConfig.from_yaml(root / registration["training_profile"])
    seed_row = registration["seed_roster"][tuning_seed_index]
    if tuple(manifest["seeds"]["tuning"][tuning_seed_index]) != (
            seed_row["ppo"], seed_row["sampling"], seed_row["perturbation"]):
        raise ValueError("LOCAL_DEMO seed bundle differs from the simulation-only tuning roster")
    ppo = replace(base.ppo, seed=seed_row["ppo"], device="cpu")
    config = manifest["training"]
    if smoke_profile is not None:
        if (smoke_profile["tracks"] != list(TRAIN_IDS)
                or smoke_profile["rollout_length"] % smoke_profile["rollout_batches"]):
            raise ValueError("LOCAL_DEMO smoke rollout or TRAIN track roster is invalid")
        ppo = replace(ppo, n_steps=smoke_profile["rollout_length"],
                      batch_size=smoke_profile["rollout_length"] // smoke_profile["rollout_batches"],
                      device=smoke_profile["device"])
        timesteps = smoke_profile["timesteps"]
        checkpoint_every = smoke_profile["checkpoint_interval"]
        max_wall_seconds = smoke_profile["max_wall_seconds"]
    else:
        timesteps = config["timesteps_per_candidate_seed"]
        checkpoint_every = config["checkpoint_every_timesteps"]
        max_wall_seconds = config["max_wall_seconds"]
    if timesteps % (ppo.n_steps * (smoke_profile or {}).get("n_envs", config["n_envs"])):
        raise ValueError("fixed local demo budget must align to complete PPO rollout batches")
    reset_seed = seed_row["perturbation"]
    if any(scenario.factory.reset_config.seed != reset_seed for scenario in scenarios.scenarios):
        raise ValueError("TRAIN reset seed differs from the selected tuning seed bundle")
    track_rows = []
    reset_contracts = {}
    for scenario in scenarios.scenarios:
        entry = next(item for item in scenario.factory.reset_config.training_registry
                     if item.track_id == scenario.track_id)
        track_rows.append({"track_id": scenario.track_id, "world_sha256": entry.world_sha256,
                           "content_sha256": entry.content_sha256, "path": entry.path})
        reset_contracts[scenario.track_id] = scenario.factory.reset_config.identity_for_track(scenario.track_id)
    if (scenarios.action_space.shape != (3,) or len(scenarios.observation_space.shape) != 1):
        raise ValueError("LOCAL_DEMO action/observation preflight failed")
    return {
        "source_commit": source_commit, "source_tree_sha256": source_tree_sha256,
        "experiment_manifest_sha256": experiment_sha256, "candidate": candidate,
        "demo_reward": reward, "scenarios": scenarios, "ppo": ppo,
        "timesteps": timesteps,
        "checkpoint_every": checkpoint_every,
        "max_wall_seconds": max_wall_seconds,
        "profile": profile, "smoke_profile": smoke_profile,
        "smoke_profile_sha256": smoke_profile_sha256,
        "threads": config["torch_threads"], "interop_threads": config["torch_interop_threads"],
        "sampling_seed": seed_row["sampling"], "reset_seed": reset_seed,
        "track_rows": track_rows, "reset_contracts": reset_contracts,
        "registration_sha256": _hash_file(root / "configs/research/wp7-rotating-owner-protocol-v3.json"),
        "track_manifest_sha256": _hash_file(root / "assets/tracks/manifest.json"),
        "training_profile_sha256": _hash_file(root / registration["training_profile"]),
        "physics_sha256": scenarios.scenarios[0].factory.physics_config.sha256,
        "observation_schema": list(scenarios.scenarios[0].factory.observation_encoder.feature_names),
        "action_contract_version": scenarios.scenarios[0].factory.action_contract.version,
        "ppo_config": asdict(ppo),
        "action_space": {"shape": list(scenarios.action_space.shape),
                         "low": scenarios.action_space.low.tolist(),
                         "high": scenarios.action_space.high.tolist()},
        "observation_space": {"shape": list(scenarios.observation_space.shape),
                              "low": scenarios.observation_space.low.tolist(),
                              "high": scenarios.observation_space.high.tolist()},
        "scientific_evidence": False, "training_authorized": False,
    }


def _install_demo_reward(vector_env, reward: dict) -> None:
    for sampled_env in vector_env.envs:
        original_step = sampled_env.step

        def step(action, *, _sampled_env=sampled_env, _original=original_step):
            result = _original(action)
            if _sampled_env._active_env is None:
                raise RuntimeError("TRAIN scenario did not create an active environment")
            return apply_training_curriculum(_sampled_env._active_env, result, reward)

        sampled_env.step = step


def _finalize_run(output: Path, state: dict, *, telemetry, model, vector, session,
                  learning_completed: bool) -> None:
    errors = []
    if telemetry is not None:
        try:
            telemetry.close()
            _write_json(output / "training-telemetry-manifest.json",
                        telemetry.manifest(complete=learning_completed))
        except BaseException as error:
            errors.append(error)
    try:
        if model is not None:
            model._model.get_env().close()
        elif vector is not None:
            vector.close()
    except BaseException as error:
        errors.append(error)
    finally:
        if session is not None:
            try:
                _revoke(vector)
            except BaseException as error:
                errors.append(error)
    if errors:
        state.update(status="failed", phase="finalization_failed",
                     finalization_error="; ".join(
                         f"{type(error).__name__}: {error}" for error in errors
                     ), scientific_evidence=False, scientific_training_authorized=False)
        try:
            _write_json(output / "state.json", state)
        except BaseException as error:
            errors[0].add_note(
                f"Could not persist LOCAL_DEMO finalization failure: {type(error).__name__}: {error}"
            )
        for error in errors[1:]:
            errors[0].add_note(f"Additional LOCAL_DEMO finalization failure: {type(error).__name__}: {error}")
        raise errors[0]


def run_local_demo(root: str | Path, *, candidate_id: str, tuning_seed_index: int,
                   output_directory: str | Path, dry_run: bool = False,
                   progress_observer=None, profile: str = "tuning") -> dict:
    """Run the existing bounded tuning profile, isolated smoke profile, or dry-run."""
    root = Path(root).resolve(strict=True)
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be bool")
    if progress_observer is not None and not callable(progress_observer):
        raise ValueError("progress_observer must be callable or None")
    started = time.monotonic()
    preflight = _preflight(root, candidate_id, tuning_seed_index, profile=profile)
    output = Path(output_directory)
    if not output.is_absolute():
        raise ValueError("LOCAL_DEMO output directory must be absolute")
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("LOCAL_DEMO evidence must be stored outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    manifest = {
        "schema_version": "f1rl_local_demo_run_v1", "run_id": run_id,
        "status": "prepared_not_scientific", "classification": "LOCAL_DEMO_NOT_SCIENTIFIC",
        "split": "TRAIN", "source_commit": preflight["source_commit"],
        "source_tree_sha256": preflight["source_tree_sha256"],
        "experiment_manifest_sha256": preflight["experiment_manifest_sha256"],
        "registration_sha256": preflight["registration_sha256"],
        "track_manifest_sha256": preflight["track_manifest_sha256"],
        "physics_sha256": preflight["physics_sha256"],
        "observation_schema": preflight["observation_schema"],
        "action_contract_version": preflight["action_contract_version"],
        "ppo_config": preflight["ppo_config"],
        "candidate": preflight["candidate"], "demo_reward": preflight["demo_reward"],
        "training_seed_bundle": {"ppo": preflight["ppo"].seed,
                                  "sampling": preflight["sampling_seed"],
                                  "training_reset": preflight["reset_seed"]},
        "track_roster": preflight["track_rows"],
        "reset_contracts": preflight["reset_contracts"],
        "action_space": preflight["action_space"], "observation_space": preflight["observation_space"],
        "limits": {"learning_timesteps": preflight["timesteps"],
                   "checkpoint_every_timesteps": preflight["checkpoint_every"],
                   "max_wall_seconds": preflight["max_wall_seconds"], "device": "cpu",
                   "n_envs": 1, "torch_threads": preflight["threads"],
                   "torch_interop_threads": preflight["interop_threads"]},
        "dry_run": dry_run, "scientific_evidence": False,
        "authorization_context": "explicit_local_demo_command",
        "scientific_training_authorized": False,
        "heldout_access": False,
    }
    if profile == "smoke":
        manifest.update(schema_version="f1rl_local_demo_smoke_run_v1",
                        classification="LOCAL_DEMO_SMOKE_NOT_SCIENTIFIC",
                        profile="smoke", smoke_profile_sha256=preflight["smoke_profile_sha256"],
                        training_profile_sha256=preflight["training_profile_sha256"])
    manifest["sha256"] = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    _write_json(output / "run-manifest.json", manifest)
    state = {"status": "running", "phase": "environment_preflight", "learning_timesteps": 0,
             "optimizer_steps": 0, "scientific_training_authorized": False,
             "scientific_evidence": False}
    _write_json(output / "state.json", state)
    vector = None
    model = None
    telemetry = None
    session = None
    primary_error = None
    learning_completed = False
    deadline = started + (min(60, preflight["max_wall_seconds"]) if dry_run
                          else preflight["max_wall_seconds"])
    try:
        configure_cpu_runtime(preflight["threads"], preflight["interop_threads"])
        if time.monotonic() >= deadline:
            raise LocalDemoStopped("LOCAL_DEMO time budget exhausted during preflight")
        vector = make_multi_track_vector_env(
            preflight["scenarios"], VectorEnvConfig(n_envs=1, backend="dummy"),
            preflight["sampling_seed"],
        )
        _install_demo_reward(vector, preflight["demo_reward"])
        session = _issue(root, vector, commit=preflight["source_commit"],
                         tree_sha256=preflight["source_tree_sha256"], deadline=deadline,
                         max_timesteps=preflight["timesteps"])
        model = PPOModel.create(vector, preflight["ppo"])
        if model.num_timesteps != 0:
            raise RuntimeError("fresh LOCAL_DEMO policy unexpectedly has learning timesteps")
        if dry_run:
            state.update(status="dry_run_pass", phase="complete", policy_weights_sha256=model.weights_sha256(),
                         action_space=manifest["action_space"], observation_space=manifest["observation_space"])
            _write_json(output / "state.json", state)
            return {**manifest, "state": state}

        import platform
        runtime = {"python": platform.python_version(), "system": platform.platform(),
                   "torch": __import__("torch").__version__,
                   "stable_baselines3": __import__("stable_baselines3").__version__}
        _write_json(output / "runtime.json", runtime)
        identity = {
            "run_id": run_id, "source_commit": preflight["source_commit"],
            "policy_id": (f"local-demo-smoke-{candidate_id}-seed-{tuning_seed_index}"
                          if profile == "smoke"
                          else f"local-demo-{candidate_id}-tuning-{tuning_seed_index}"),
            "split": "TRAIN", "track_ids": list(TRAIN_IDS),
            "seed_identity": {"ppo": preflight["ppo"].seed,
                              "sampling": preflight["sampling_seed"],
                              "training_reset": preflight["reset_seed"]},
            "reset_contracts": preflight["reset_contracts"],
        }
        if profile == "smoke":
            _assert_finite_policy_weights(model)
        initial_weights_sha256 = model.weights_sha256()
        initial_action_weights_sha256 = (
            _action_policy_weights_sha256(model) if profile == "smoke" else None
        )
        if profile == "smoke":
            state.update(initial_policy_weights_sha256=initial_weights_sha256,
                         initial_action_policy_weights_sha256=initial_action_weights_sha256)
            _write_json(output / "state.json", state)
        telemetry = TrainingTelemetryRecorder(
            output / ("SMOKE-TRAINING-TELEMETRY.jsonl" if profile == "smoke"
                      else "training-telemetry.jsonl"),
            run_identity=identity, max_timesteps=preflight["timesteps"],
            capture_autoreset=profile == "smoke",
        )
        checkpoint_root = output / "checkpoints"
        checkpoint_root.mkdir()
        optimizer_steps = 0
        checkpoints = []
        gradient_activity = []

        def progress(actual_timesteps: int) -> None:
            nonlocal optimizer_steps
            session.check(actual_timesteps)
            if telemetry.optimizer_steps > optimizer_steps:
                optimizer_steps = telemetry.optimizer_steps
            if profile == "smoke":
                state.update(phase="learning", learning_timesteps=actual_timesteps,
                             optimizer_steps=optimizer_steps)
                _write_json(output / "state.json", state)
                if progress_observer is not None:
                    progress_observer(dict(state))
            if (preflight["checkpoint_every"] and actual_timesteps
                    and actual_timesteps % preflight["checkpoint_every"] == 0):
                temp = checkpoint_root / f"checkpoint-{actual_timesteps}.pending.zip"
                final = checkpoint_root / f"checkpoint-{actual_timesteps}.zip"
                _save_checkpoint(model, temp, final)
                checkpoint_sha256 = _hash_file(final)
                checkpoints.append({"path": final.relative_to(output).as_posix(),
                                    "sha256": checkpoint_sha256, "timesteps": actual_timesteps})
                telemetry.record_checkpoint(final.name, timestep=actual_timesteps,
                                            archive_sha256=checkpoint_sha256,
                                            policy_weights_sha256=model.weights_sha256())
                _write_json(output / "checkpoints.json", {"checkpoints": checkpoints})
                state.update(phase="learning", learning_timesteps=actual_timesteps,
                             optimizer_steps=optimizer_steps, latest_checkpoint=checkpoints[-1])
                _write_json(output / "state.json", state)
                if progress_observer is not None:
                    progress_observer(dict(state))

        state.update(phase="learning")
        _write_json(output / "state.json", state)
        def record_gradient_activity() -> None:
            gradients = [parameter.grad.detach() for parameter in model._model.policy.parameters()
                         if parameter.grad is not None]
            import torch
            if not gradients:
                raise RuntimeError("PPO optimizer step produced no gradients")
            finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            norm = float(torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(gradient.float()) for gradient in gradients
            ])).item())
            if not finite or not np.isfinite(norm):
                raise RuntimeError("PPO optimizer step produced non-finite gradients")
            gradient_activity.append({"step": len(gradient_activity) + 1,
                                      "gradient_l2_norm": norm,
                                      "nonzero_gradient_tensors": sum(
                                          bool(torch.count_nonzero(gradient)) for gradient in gradients),
                                      "finite": finite})

        model.learn(preflight["timesteps"], progress_hook=progress,
                    telemetry_observer=telemetry,
                    optimizer_step_hook=record_gradient_activity if profile == "smoke" else None)
        if profile == "smoke":
            session.check(model.num_timesteps)
        if model.num_timesteps != preflight["timesteps"] or telemetry.optimizer_steps <= 0:
            raise RuntimeError("LOCAL_DEMO did not complete the bounded genuine PPO learning budget")
        final_weights_sha256 = model.weights_sha256()
        final_action_weights_sha256 = (
            _action_policy_weights_sha256(model) if profile == "smoke" else None
        )
        if profile == "smoke":
            _assert_finite_policy_weights(model)
            if (model.num_timesteps != 256 or len(gradient_activity) != telemetry.optimizer_steps
                    or not any(row["gradient_l2_norm"] > 0 for row in gradient_activity)
                    or initial_weights_sha256 == final_weights_sha256
                    or initial_action_weights_sha256 == final_action_weights_sha256
                    or telemetry.step_rows != 256):
                raise RuntimeError("LOCAL_DEMO smoke failed its genuine rollout/update/weight checks")
        final_tmp = output / "model-final.pending.zip"
        final_model = output / "model-final.zip"
        _save_checkpoint(model, final_tmp, final_model)
        reloaded_weights_sha256 = final_weights_sha256
        if profile == "smoke":
            reloaded = PPOModel.load(final_model, device="cpu")
            _assert_finite_policy_weights(reloaded)
            reloaded_weights_sha256 = reloaded.weights_sha256()
            reloaded_action_weights_sha256 = _action_policy_weights_sha256(reloaded)
            if (reloaded_weights_sha256 != final_weights_sha256
                    or reloaded_action_weights_sha256 != final_action_weights_sha256
                    or reloaded.num_timesteps != model.num_timesteps
                    or reloaded._model.observation_space != model._model.observation_space
                    or reloaded._model.action_space != model._model.action_space):
                raise RuntimeError("saved PPO checkpoint failed CPU identity/schema reload verification")
            _write_json(output / "SMOKE-WEIGHT-COMPARISON.json", {
                "initial_policy_weights_sha256": initial_weights_sha256,
                "final_policy_weights_sha256": final_weights_sha256,
                "reloaded_policy_weights_sha256": reloaded_weights_sha256,
                "initial_action_policy_weights_sha256": initial_action_weights_sha256,
                "final_action_policy_weights_sha256": final_action_weights_sha256,
                "reloaded_action_policy_weights_sha256": reloaded_action_weights_sha256,
                "initial_policy_weights_finite": True,
                "final_policy_weights_finite": True,
                "reloaded_policy_weights_finite": True,
                "weights_changed": initial_weights_sha256 != final_weights_sha256,
                "action_policy_weights_changed": initial_action_weights_sha256 != final_action_weights_sha256,
                "reload_matches_final": reloaded_weights_sha256 == final_weights_sha256,
                "optimizer_steps": telemetry.optimizer_steps,
                "optimizer_gradient_activity": gradient_activity,
                "nonzero_gradient_steps": sum(row["gradient_l2_norm"] > 0
                                               for row in gradient_activity),
                "ppo_epoch_updates": int(model._model._n_updates),
                "training_metrics": telemetry._metrics,
            })
            _write_json(output / "SMOKE-CHECKPOINT-VERIFICATION.json", {
                "archive_path": final_model.name,
                "archive_sha256": _hash_file(final_model),
                "persisted_policy_weights_sha256": final_weights_sha256,
                "reloaded_policy_weights_sha256": reloaded_weights_sha256,
                "action_policy_weights_sha256": reloaded_action_weights_sha256,
                "policy_weights_finite": True,
                "reloaded_num_timesteps": reloaded.num_timesteps,
                "observation_space_matches": reloaded._model.observation_space == model._model.observation_space,
                "action_space_matches": reloaded._model.action_space == model._model.action_space,
                "verified": True,
            })
        telemetry.record_checkpoint(final_model.name, timestep=model.num_timesteps,
                                    archive_sha256=_hash_file(final_model),
                                    policy_weights_sha256=final_weights_sha256)
        learning_completed = True
        state.update(status="learning_completed", phase="complete", learning_timesteps=model.num_timesteps,
                     optimizer_steps=telemetry.optimizer_steps,
                     final_model={"path": final_model.name, "sha256": _hash_file(final_model)},
                     scientific_evidence=False, scientific_training_authorized=False)
        if profile == "smoke":
            state.update(initial_policy_weights_sha256=initial_weights_sha256,
                         initial_action_policy_weights_sha256=initial_action_weights_sha256,
                         final_policy_weights_sha256=final_weights_sha256,
                         final_action_policy_weights_sha256=final_action_weights_sha256,
                         reloaded_policy_weights_sha256=reloaded_weights_sha256)
        _write_json(output / "state.json", state)
        return {**manifest, "state": state}
    except BaseException as error:
        primary_error = error
        if profile == "smoke":
            if model is not None:
                state["learning_timesteps"] = model.num_timesteps
            if telemetry is not None:
                state["optimizer_steps"] = telemetry.optimizer_steps
        state.update(status="failed", phase=state.get("phase"),
                     error=f"{type(error).__name__}: {error}",
                     scientific_evidence=False, scientific_training_authorized=False)
        try:
            _write_json(output / "state.json", state)
        except BaseException as state_error:
            error.add_note(
                f"Could not persist LOCAL_DEMO failure state: "
                f"{type(state_error).__name__}: {state_error}"
            )
        raise
    finally:
        try:
            _finalize_run(output, state, telemetry=telemetry, model=model, vector=vector,
                          session=session, learning_completed=learning_completed)
        except BaseException as finalization_error:
            state.update(status="failed", phase="finalization_failed",
                         finalization_error=f"{type(finalization_error).__name__}: {finalization_error}",
                         scientific_evidence=False, scientific_training_authorized=False)
            if primary_error is not None:
                primary_error.add_note(
                    f"LOCAL_DEMO finalization also failed: {type(finalization_error).__name__}: "
                    f"{finalization_error}"
                )
            try:
                _write_json(output / "state.json", state)
            except BaseException as state_error:
                if primary_error is not None:
                    primary_error.add_note(
                        f"Could not persist LOCAL_DEMO finalization failure: "
                        f"{type(state_error).__name__}: {state_error}"
                    )
                elif finalization_error is not state_error:
                    finalization_error.add_note(
                        f"Could not persist LOCAL_DEMO finalization failure: "
                        f"{type(state_error).__name__}: {state_error}"
                    )
            if primary_error is None:
                raise
        if profile == "smoke" and primary_error is None and state.get("status") == "learning_completed":
            try:
                telemetry_manifest = json.loads(
                    (output / "training-telemetry-manifest.json").read_text(encoding="utf-8")
                )
                telemetry_path = output / telemetry_manifest["path"]
                transition_accounting = _validate_smoke_telemetry(telemetry_path, manifest["run_id"])
                if (not telemetry_manifest.get("complete")
                        or telemetry_manifest.get("step_rows") != 256
                        or telemetry_manifest.get("update_rows") != 1
                        or telemetry_manifest.get("optimizer_steps") != state["optimizer_steps"]
                        or telemetry_manifest.get("bytes") != telemetry_path.stat().st_size
                        or telemetry_manifest.get("sha256") != _hash_file(telemetry_path)):
                    raise ValueError("smoke telemetry manifest integrity or counters failed")
                weights = json.loads((output / "SMOKE-WEIGHT-COMPARISON.json").read_text(encoding="utf-8"))
                checkpoint = json.loads(
                    (output / "SMOKE-CHECKPOINT-VERIFICATION.json").read_text(encoding="utf-8")
                )
                checkpoint_path = output / checkpoint["archive_path"]
                if (not checkpoint.get("verified")
                        or checkpoint.get("archive_sha256") != state["final_model"]["sha256"]
                        or _hash_file(checkpoint_path) != checkpoint.get("archive_sha256")
                        or checkpoint.get("reloaded_policy_weights_sha256")
                        != weights.get("final_policy_weights_sha256")
                        or checkpoint.get("action_policy_weights_sha256")
                        != weights.get("final_action_policy_weights_sha256")
                        or not weights.get("weights_changed")
                        or not weights.get("action_policy_weights_changed")
                        or weights.get("ppo_epoch_updates", 0) <= 0
                        or weights.get("optimizer_steps") != state["optimizer_steps"]
                        or weights.get("nonzero_gradient_steps") != state["optimizer_steps"]
                        or weights.get("initial_policy_weights_finite") is not True
                        or weights.get("final_policy_weights_finite") is not True
                        or weights.get("reloaded_policy_weights_finite") is not True
                        or checkpoint.get("policy_weights_finite") is not True):
                    raise ValueError("smoke checkpoint or optimizer evidence failed verification")
                if (transition_accounting["initial_policy_weights_sha256"]
                        != weights["initial_policy_weights_sha256"]
                        or transition_accounting["updated_policy_weights_sha256"]
                        != weights["final_policy_weights_sha256"]):
                    raise ValueError("smoke telemetry policy identities do not match weight comparison")
                _write_json(output / "SMOKE-VALIDATION.json", {
                    "schema_version": "f1rl_local_demo_smoke_validation_v1",
                    "classification": "LOCAL_DEMO_SMOKE_NOT_SCIENTIFIC",
                    "source_commit": manifest["source_commit"],
                    "source_tree_sha256": manifest["source_tree_sha256"],
                    "run_manifest_path": "run-manifest.json",
                    "run_manifest_sha256": _hash_file(output / "run-manifest.json"),
                    "smoke_profile_sha256": manifest["smoke_profile_sha256"],
                    "experiment_manifest_sha256": manifest["experiment_manifest_sha256"],
                    "registration_sha256": manifest["registration_sha256"],
                    "track_manifest_sha256": manifest["track_manifest_sha256"],
                    "training_profile_sha256": manifest["training_profile_sha256"],
                    "physics_sha256": manifest["physics_sha256"],
                    "candidate": "C", "tuning_seed_index": 0, "split": "TRAIN",
                    "track_roster": manifest["track_roster"],
                    "ppo_config": manifest["ppo_config"],
                    "observation_schema": manifest["observation_schema"],
                    "action_contract_version": manifest["action_contract_version"],
                    "learning_timesteps": state["learning_timesteps"],
                    "rollout_length": preflight["ppo"].n_steps,
                    "n_envs": 1, "batch_size": preflight["ppo"].batch_size,
                    "optimizer_steps": state["optimizer_steps"],
                    "weight_comparison": "SMOKE-WEIGHT-COMPARISON.json",
                    "checkpoint_verification": "SMOKE-CHECKPOINT-VERIFICATION.json",
                    "telemetry_path": telemetry_manifest["path"],
                    "telemetry_manifest": telemetry_manifest,
                    "telemetry_sha256": telemetry_manifest["sha256"],
                    "telemetry_bytes": telemetry_manifest["bytes"],
                    "transition_accounting": transition_accounting,
                    "initial_policy_weights_sha256": weights["initial_policy_weights_sha256"],
                    "final_policy_weights_sha256": weights["final_policy_weights_sha256"],
                    "reload_policy_weights_sha256": weights["reloaded_policy_weights_sha256"],
                    "policy_weights_finite": True,
                    "initial_action_policy_weights_sha256": weights["initial_action_policy_weights_sha256"],
                    "final_action_policy_weights_sha256": weights["final_action_policy_weights_sha256"],
                    "reload_action_policy_weights_sha256": weights["reloaded_action_policy_weights_sha256"],
                    "optimizer_gradient_steps": weights["optimizer_gradient_activity"],
                    "training_metrics": weights["training_metrics"],
                    "final_checkpoint": state["final_model"],
                    "heldout_access": False, "scientific_evaluation": False,
                    "scientific_training_authorized": False,
                    "status": "SMOKE_INTEGRATION_PASS",
                })
                state.update(status="SMOKE_INTEGRATION_PASS")
                _write_json(output / "state.json", state)
            except BaseException as error:
                state.update(status="failed", phase="smoke_validation_failed",
                             error=f"{type(error).__name__}: {error}")
                try:
                    _write_json(output / "state.json", state)
                except BaseException as state_error:
                    error.add_note(f"Could not persist smoke validation failure: {state_error}")
                raise
        elif profile == "smoke":
            failed_validation = {
                "schema_version": "f1rl_local_demo_smoke_validation_v1",
                "classification": "LOCAL_DEMO_SMOKE_NOT_SCIENTIFIC",
                "status": "SMOKE_INTEGRATION_FAILED",
                "source_commit": manifest["source_commit"],
                "smoke_profile_sha256": manifest["smoke_profile_sha256"],
                "candidate": "C", "tuning_seed_index": 0, "split": "TRAIN",
                "state": state,
                "learning_timesteps": state.get("learning_timesteps", 0),
                "optimizer_steps": telemetry.optimizer_steps if telemetry is not None else 0,
                "heldout_access": False, "scientific_evaluation": False,
                "scientific_training_authorized": False,
            }
            try:
                _write_json(output / "SMOKE-VALIDATION.json", failed_validation)
            except BaseException as error:
                if primary_error is not None:
                    primary_error.add_note(f"Could not persist failed smoke validation: {error}")
                else:
                    raise
