"""Bounded local Windows PPO demo; output is never WP7 scientific evidence."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np

from ..learning.algorithms.ppo import PPOConfig, PPOModel, configure_cpu_runtime
from ..learning.pilot_preflight import TRAIN_IDS
from ..learning.provenance import _capture_git_provenance
from ..learning.training.config import TrainingRunConfig, VectorEnvConfig
from ..learning.training.telemetry import TrainingTelemetryRecorder
from ..learning.training.vector_env import make_multi_track_vector_env
from ..research.simulation_only_protocol import load_simulation_only_manifest
from ..research.wp7_protocol import load_registration, prepare_scenarios
from ..exploratory.reward import apply_training_curriculum, validate as validate_demo_reward
from .admission import LocalDemoStopped, _issue, _revoke


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


def _source_identity(root: Path) -> tuple[str, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                           cwd=root, check=True, capture_output=True, text=True).stdout
    if dirty or branch in ("", "main") or len(commit) != 40 or len(tree) != 40:
        raise ValueError("LOCAL_DEMO requires a clean committed source tree")
    return commit, sha256(b"f1rl-local-demo-git-tree-v1\x00" + bytes.fromhex(tree)).hexdigest()


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


def _preflight(root: Path, candidate_id: str, tuning_seed_index: int):
    if os.name != "nt":
        raise ValueError("LOCAL_DEMO is supported on local Windows only")
    if type(tuning_seed_index) is not int or not 0 <= tuning_seed_index < 3:
        raise ValueError("LOCAL_DEMO may use only one of the three reserved tuning seed bundles")
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
    if config["timesteps_per_candidate_seed"] % (ppo.n_steps * config["n_envs"]):
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
        "timesteps": config["timesteps_per_candidate_seed"],
        "checkpoint_every": config["checkpoint_every_timesteps"],
        "max_wall_seconds": config["max_wall_seconds"],
        "threads": config["torch_threads"], "interop_threads": config["torch_interop_threads"],
        "sampling_seed": seed_row["sampling"], "reset_seed": reset_seed,
        "track_rows": track_rows, "reset_contracts": reset_contracts,
        "registration_sha256": _hash_file(root / "configs/research/wp7-rotating-owner-protocol-v3.json"),
        "track_manifest_sha256": _hash_file(root / "assets/tracks/manifest.json"),
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


def run_local_demo(root: str | Path, *, candidate_id: str, tuning_seed_index: int,
                   output_directory: str | Path, dry_run: bool = False,
                   progress_observer=None) -> dict:
    """Run an explicit local demo or no-learning PPO/environment preflight."""
    root = Path(root).resolve(strict=True)
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be bool")
    if progress_observer is not None and not callable(progress_observer):
        raise ValueError("progress_observer must be callable or None")
    started = time.monotonic()
    preflight = _preflight(root, candidate_id, tuning_seed_index)
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
    manifest["sha256"] = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    _write_json(output / "run-manifest.json", manifest)
    state = {"status": "running", "phase": "environment_preflight", "learning_timesteps": 0,
             "optimizer_steps": 0, "scientific_training_authorized": False,
             "scientific_evidence": False}
    _write_json(output / "state.json", state)
    configure_cpu_runtime(preflight["threads"], preflight["interop_threads"])
    vector = None
    model = None
    telemetry = None
    session = None
    learning_completed = False
    deadline = started + (min(60, preflight["max_wall_seconds"]) if dry_run
                          else preflight["max_wall_seconds"])
    try:
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
            "policy_id": f"local-demo-{candidate_id}-tuning-{tuning_seed_index}",
            "split": "TRAIN", "track_ids": list(TRAIN_IDS),
            "seed_identity": {"ppo": preflight["ppo"].seed,
                              "sampling": preflight["sampling_seed"],
                              "training_reset": preflight["reset_seed"]},
            "reset_contracts": preflight["reset_contracts"],
        }
        telemetry = TrainingTelemetryRecorder(
            output / "training-telemetry.jsonl", run_identity=identity,
            max_timesteps=preflight["timesteps"],
        )
        checkpoint_root = output / "checkpoints"
        checkpoint_root.mkdir()
        optimizer_steps = 0
        checkpoints = []

        def progress(actual_timesteps: int) -> None:
            nonlocal optimizer_steps
            session.check(actual_timesteps)
            if telemetry.optimizer_steps > optimizer_steps:
                optimizer_steps = telemetry.optimizer_steps
            if actual_timesteps and actual_timesteps % preflight["checkpoint_every"] == 0:
                temp = checkpoint_root / f"checkpoint-{actual_timesteps}.pending.zip"
                final = checkpoint_root / f"checkpoint-{actual_timesteps}.zip"
                model.save(temp)
                os.replace(temp, final)
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
        model.learn(preflight["timesteps"], progress_hook=progress, telemetry_observer=telemetry)
        if model.num_timesteps != preflight["timesteps"] or telemetry.optimizer_steps <= 0:
            raise RuntimeError("LOCAL_DEMO did not complete the bounded genuine PPO learning budget")
        final_tmp = output / "model-final.pending.zip"
        final_model = output / "model-final.zip"
        model.save(final_tmp)
        os.replace(final_tmp, final_model)
        telemetry.record_checkpoint(final_model.name, timestep=model.num_timesteps,
                                    archive_sha256=_hash_file(final_model),
                                    policy_weights_sha256=model.weights_sha256())
        learning_completed = True
        state.update(status="learning_completed", phase="complete", learning_timesteps=model.num_timesteps,
                     optimizer_steps=telemetry.optimizer_steps,
                     final_model={"path": final_model.name, "sha256": _hash_file(final_model)},
                     scientific_evidence=False, scientific_training_authorized=False)
        _write_json(output / "state.json", state)
        return {**manifest, "state": state}
    except BaseException as error:
        state.update(status="failed", phase=state.get("phase"),
                     error=f"{type(error).__name__}: {error}",
                     scientific_evidence=False, scientific_training_authorized=False)
        _write_json(output / "state.json", state)
        raise
    finally:
        if telemetry is not None:
            telemetry.close()
            _write_json(output / "training-telemetry-manifest.json",
                        telemetry.manifest(complete=learning_completed))
        if model is not None:
            model._model.get_env().close()
        elif vector is not None:
            vector.close()
        if session is not None:
            _revoke(vector)
