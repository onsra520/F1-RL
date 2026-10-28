"""Bounded, versioned observations of PPO learning; never a policy input."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np


_SCHEMA = "f1rl_training_telemetry_v1"
_SMOKE_SCHEMA = "f1rl_local_demo_smoke_telemetry_v1"
_TRAIN_TRACKS = {"straight-sprint-v1", "right-turn-v1", "slalom-v1", "ramp-v1"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MAX_RECORD_BYTES = 16 * 1024
_MAX_LOG_BYTES = 4 * 1024 * 1024 * 1024
_AUXILIARY_RESERVE_BYTES = 1024 * 1024
_METRIC_KEYS = (
    "train/approx_kl", "train/clip_fraction", "train/clip_range",
    "train/entropy_loss", "train/ent_coef", "train/explained_variance",
    "train/learning_rate", "train/loss", "train/n_updates",
    "train/policy_gradient_loss", "train/std", "train/value_loss",
)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _number(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


class TrainingTelemetryRecorder:
    """Stream read-only PPO transition and update records to a fresh JSONL file."""

    def __init__(self, path: str | Path, *, run_identity: dict, max_timesteps: int,
                 capture_autoreset: bool = False, schema_version: str = _SCHEMA) -> None:
        if type(max_timesteps) is not int or max_timesteps <= 0:
            raise ValueError("max_timesteps must be a positive integer")
        if type(capture_autoreset) is not bool:
            raise ValueError("capture_autoreset must be bool")
        if schema_version not in (_SCHEMA, _SMOKE_SCHEMA):
            raise ValueError("unsupported training telemetry schema")
        required = {"run_id", "source_commit", "policy_id", "split", "track_ids",
                    "seed_identity", "reset_contracts"}
        optional = {"training_failure_policy"}
        if (not isinstance(run_identity, dict)
                or set(run_identity) not in (required, required | optional)):
            raise ValueError("run_identity has unexpected fields")
        if ("training_failure_policy" in run_identity
                and run_identity["training_failure_policy"] not in (
                    "abort_on_physics_error", "unsupported_contact_terminal_v1",
                    "training_contact_terminal_v1")):
            raise ValueError("training failure policy is not registered")
        if (not all(isinstance(run_identity[k], str) and run_identity[k].strip()
                    for k in ("run_id", "policy_id"))
                or _COMMIT.fullmatch(run_identity["source_commit"]) is None
                or run_identity["split"] != "TRAIN"
                or not isinstance(run_identity["track_ids"], (list, tuple))
                or not run_identity["track_ids"]
                or any(not isinstance(track, str) for track in run_identity["track_ids"])
                or len(set(run_identity["track_ids"])) != len(run_identity["track_ids"])
                or any(track not in _TRAIN_TRACKS for track in run_identity["track_ids"])
                or not isinstance(run_identity["seed_identity"], dict)
                or type(run_identity["seed_identity"].get("training_reset")) is not int
                or not 0 <= run_identity["seed_identity"]["training_reset"] <= 2**32 - 1
                or not isinstance(run_identity["reset_contracts"], dict)
                or set(run_identity["reset_contracts"]) != set(run_identity["track_ids"])):
            raise ValueError("training telemetry requires an identified TRAIN run")
        for track_id, contract in run_identity["reset_contracts"].items():
            if (not isinstance(contract, dict)
                    or set(contract) != {"version", "seed", "configuration_sha256", "world_sha256"}
                    or contract["version"] not in ("wp7_train_reset_research_v1",
                                                      "wp7_train_reset_lifecycle_research_v1")
                    or type(contract["seed"]) is not int
                    or contract["seed"] != run_identity["seed_identity"]["training_reset"]
                    or not isinstance(contract["configuration_sha256"], str)
                    or _DIGEST.fullmatch(contract["configuration_sha256"]) is None
                    or not isinstance(contract["world_sha256"], str)
                    or _DIGEST.fullmatch(contract["world_sha256"]) is None):
                raise ValueError(f"declared training reset seed or contract is invalid for {track_id}")
        self.path = Path(path)
        if not self.path.parent.is_dir():
            raise ValueError("telemetry output directory must already exist")
        self._run = deepcopy(run_identity)
        self._schema = schema_version
        self._capture_autoreset = capture_autoreset
        self._max_timesteps = max_timesteps
        required_capacity = max_timesteps * _MAX_RECORD_BYTES + _AUXILIARY_RESERVE_BYTES
        if required_capacity > _MAX_LOG_BYTES:
            raise ValueError("requested telemetry timestep budget exceeds the 4 GiB capacity bound")
        self._max_bytes = required_capacity
        if shutil.disk_usage(self.path.parent).free < self._max_bytes:
            raise OSError(f"insufficient free space for telemetry capacity bound: {self._max_bytes} bytes")
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._content_hash = sha256()
        self._bytes = 0
        self._events = 0
        self.step_rows = 0
        self.update_rows = 0
        self.optimizer_steps = 0
        self._n_envs = None
        self._episodes: dict[int, int] = {}
        self._weight_sha256 = None
        self._update_index = 0
        self._metrics: dict[str, float | None] = {key: None for key in _METRIC_KEYS}
        self._checkpoint = None
        self._observation_schema = None
        self._closed = False
        self._total_timesteps = None
        try:
            self._append({"event_type": "run", "schema_version": self._schema,
                          **self._run, "max_timesteps": max_timesteps})
        except BaseException:
            self._stream.close()
            self._closed = True
            try:
                self.path.unlink()
            except OSError:
                pass
            raise

    def _append(self, record: dict) -> None:
        if self._closed:
            raise RuntimeError("telemetry recorder is closed")
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        size = len(line.encode("utf-8"))
        if size > _MAX_RECORD_BYTES or self._bytes + size > self._max_bytes:
            raise ValueError("training telemetry storage bound exceeded")
        self._stream.write(line)
        self._content_hash.update(line.encode("utf-8"))
        self._bytes += size
        self._events += 1
        if self._events % 256 == 0:
            self._sync()

    def _sync(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def on_training_start(self, model, *, total_timesteps: int, n_envs: int) -> None:
        if self._n_envs is not None:
            raise RuntimeError("training telemetry start was already recorded")
        if type(total_timesteps) is not int or not 0 < total_timesteps <= self._max_timesteps:
            raise ValueError("requested timestep budget exceeds telemetry bound")
        if type(n_envs) is not int or n_envs <= 0:
            raise ValueError("n_envs must be positive")
        capacity = (self._max_timesteps + n_envs - 1) * _MAX_RECORD_BYTES + _AUXILIARY_RESERVE_BYTES
        if capacity > _MAX_LOG_BYTES:
            raise ValueError("environment count exceeds telemetry capacity bound")
        if shutil.disk_usage(self.path.parent).free + self.path.stat().st_size < capacity:
            raise OSError(f"insufficient free space for telemetry capacity bound: {capacity} bytes")
        self._max_bytes = capacity
        self._total_timesteps = total_timesteps
        self._n_envs = n_envs
        self._episodes = {i: 0 for i in range(n_envs)}
        self._weight_sha256 = _digest(model.weights_sha256(), "policy weight identity")
        self._append({"event_type": "training_start", "schema_version": self._schema,
                      "total_timesteps": total_timesteps, "n_envs": n_envs,
                      "policy_weights_sha256": self._weight_sha256})

    def on_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def on_step(self, local_values: dict, model) -> None:
        if self._n_envs is None:
            raise RuntimeError("training start must be recorded before steps")
        obs = np.asarray(local_values["obs"])
        new_obs = np.asarray(local_values.get("new_obs")) if self._capture_autoreset else None
        raw_actions = np.asarray(local_values["actions"])
        applied_actions = np.asarray(local_values["clipped_actions"])
        rewards = np.asarray(local_values["rewards"]).reshape(-1)
        dones = np.asarray(local_values["dones"], dtype=bool).reshape(-1)
        infos = local_values["infos"]
        expected = (self._n_envs,)
        if (obs.shape[0] != self._n_envs or raw_actions.shape[0] != self._n_envs
                or applied_actions.shape != raw_actions.shape or rewards.shape != expected
                or dones.shape != expected or len(infos) != self._n_envs
                or (self._capture_autoreset and (new_obs is None or new_obs.shape != obs.shape))):
            raise ValueError("PPO callback transition batch shape changed")
        final_timestep = int(model.num_timesteps)
        if final_timestep > self._max_timesteps + self._n_envs:
            raise ValueError("training exceeded telemetry timestep bound")
        for index, info in enumerate(infos):
            state = info.get("training_telemetry")
            if not isinstance(state, dict) or state.get("schema_version") != "f1rl_training_state_v1":
                raise ValueError("environment training telemetry is missing or has an unsupported schema")
            track_id = info.get("track_id") or state.get("track_id")
            reset_identity = info.get("train_reset")
            if (not isinstance(reset_identity, dict) or not reset_identity
                    or state.get("seed_identity") != reset_identity
                    or type(reset_identity.get("seed")) is not int
                    or type(reset_identity.get("rank")) is not int
                    or type(reset_identity.get("reset_index")) is not int
                    or type(reset_identity.get("draw_seed")) is not int
                    or min(reset_identity["seed"], reset_identity["rank"],
                           reset_identity["reset_index"], reset_identity["draw_seed"]) < 0
                    or not isinstance(reset_identity.get("track_id"), str)
                    or reset_identity.get("track_id") != track_id
                    or not isinstance(reset_identity.get("configuration_sha256"), str)
                    or _DIGEST.fullmatch(reset_identity.get("configuration_sha256", "")) is None):
                raise ValueError("transition reset seed identity is missing or inconsistent")
            if track_id not in self._run["track_ids"] or track_id not in _TRAIN_TRACKS:
                raise ValueError("transition track is outside the preregistered TRAIN set")
            from src.rl.train_reset import derive_train_reset_draw_seed
            contract = self._run["reset_contracts"].get(track_id)
            if (contract is None
                    or reset_identity.get("seed") != contract["seed"]
                    or reset_identity.get("rank") != index
                    or reset_identity.get("reset_index") != self._episodes[index]
                    or reset_identity.get("track_id") != track_id
                    or reset_identity.get("version") != contract["version"]
                    or reset_identity.get("configuration_sha256") != contract["configuration_sha256"]
                    or reset_identity.get("world_sha256") != contract["world_sha256"]
                    or reset_identity.get("draw_seed") != derive_train_reset_draw_seed(
                        version=contract["version"], seed=contract["seed"], rank=index,
                        track_id=track_id, reset_index=self._episodes[index])):
                raise ValueError("declared training reset seed or identity does not match transition")
            schema = state.get("observation_schema")
            if not isinstance(schema, dict) or not schema.get("feature_names"):
                raise ValueError("observation schema identity is missing")
            schema_digest = sha256(json.dumps(schema, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode()).hexdigest()
            if self._observation_schema is None:
                self._observation_schema = schema_digest
            elif schema_digest != self._observation_schema:
                raise ValueError("observation schema changed during training")
            action = [_number(v, "applied action") for v in applied_actions[index]]
            if len(action) not in (2, 3):
                raise ValueError("unregistered action vector shape")
            reward_components = info.get("reward_components", {})
            demo = info.get("demo_training")
            research_outcome = info.get("research_outcome") or {}
            episode_index = self._episodes[index]
            global_timestep = final_timestep - self._n_envs + index + 1
            done = bool(dones[index])
            next_observation = new_obs[index] if new_obs is not None else None
            autoreset_identity = None
            if self._capture_autoreset and done:
                terminal_observation = info.get("terminal_observation")
                reset_infos = model.get_env().reset_infos
                if (terminal_observation is None or len(reset_infos) != self._n_envs
                        or not isinstance(reset_infos[index], dict)):
                    raise ValueError("smoke telemetry lacks terminal or automatic-reset state")
                next_observation = np.asarray(terminal_observation)
                reset_info = reset_infos[index]
                autoreset_identity = reset_info.get("train_reset")
                next_episode = self._episodes[index] + 1
                next_track = reset_info.get("track_id")
                next_contract = self._run["reset_contracts"].get(next_track)
                from src.rl.train_reset import derive_train_reset_draw_seed
                if (not isinstance(autoreset_identity, dict) or next_contract is None
                        or next_track not in self._run["track_ids"]
                        or type(autoreset_identity.get("seed")) is not int
                        or type(autoreset_identity.get("rank")) is not int
                        or type(autoreset_identity.get("reset_index")) is not int
                        or type(autoreset_identity.get("draw_seed")) is not int
                        or autoreset_identity.get("seed") != next_contract["seed"]
                        or autoreset_identity.get("rank") != index
                        or autoreset_identity.get("reset_index") != next_episode
                        or autoreset_identity.get("track_id") != next_track
                        or autoreset_identity.get("version") != next_contract["version"]
                        or autoreset_identity.get("configuration_sha256") != next_contract["configuration_sha256"]
                        or autoreset_identity.get("world_sha256") != next_contract["world_sha256"]
                        or autoreset_identity.get("draw_seed") != derive_train_reset_draw_seed(
                            version=next_contract["version"], seed=next_contract["seed"],
                            rank=index, track_id=next_track, reset_index=next_episode)):
                    raise ValueError("smoke automatic reset identity is inconsistent with its next episode")
            elif self._capture_autoreset:
                reset_infos = model.get_env().reset_infos
                if len(reset_infos) != self._n_envs or reset_infos[index].get("track_id") != track_id:
                    raise ValueError("smoke nonterminal transition changed active episode identity")
            encoded_next_observation = (
                [_number(v, "next observation") for v in np.asarray(next_observation).reshape(-1)]
                if self._capture_autoreset else None
            )
            encoded_autoreset_observation = (
                [_number(v, "autoreset observation") for v in new_obs[index].reshape(-1)]
                if self._capture_autoreset and done else None
            )
            truncated = bool(info.get("TimeLimit.truncated", False))
            if not done:
                termination_reason = None
            elif demo and demo.get("curriculum_terminal"):
                termination_reason = "demo_curriculum_boundary"
            elif research_outcome.get("reason") not in (None, "accepted_step"):
                termination_reason = research_outcome["reason"]
            elif truncated:
                termination_reason = info.get("truncation_reason") or "truncated"
            elif info.get("finished"):
                termination_reason = "finish"
            else:
                termination_reason = "terminated"
            record = {
                "event_type": "step", "schema_version": self._schema,
                "source_commit": self._run["source_commit"], "run_id": self._run["run_id"],
                "policy_id": self._run["policy_id"], "global_timestep": global_timestep,
                "env_index": index, "episode_id": f"{self._run['run_id']}/env-{index}/episode-{episode_index}",
                "episode_index": episode_index, "track_id": track_id,
                "seed_identity": {"run": self._run["seed_identity"],
                                  "train_reset": reset_identity},
                "observation_schema_version": schema["version"],
                "observation_feature_names": schema["feature_names"],
                "observation_schema_sha256": schema_digest,
                "observation": [_number(v, "observation") for v in obs[index].reshape(-1)],
                **({"next_observation": encoded_next_observation,
                   "autoreset_observation": encoded_autoreset_observation,
                   "autoreset_episode_id": (
                       f"{self._run['run_id']}/env-{index}/episode-{self._episodes[index] + 1}"
                       if done else None),
                   "autoreset_identity": autoreset_identity}
                  if self._capture_autoreset else {}),
                "raw_ppo_action": [_number(v, "raw PPO action") for v in raw_actions[index]],
                "action_sent_to_environment": action,
                "action_contract_version": state["action_contract_version"],
                "applied_control": {"steering": action[0], "throttle": action[1],
                                    "brake": action[2] if len(action) == 3 else 0.0},
                "vehicle_position": list(state["position"]),
                "vehicle_velocity": list(state["velocity"]), "vehicle_speed": _number(state["speed"], "speed"),
                "grounded": bool(state["grounded"]), "heading_rad": state["heading_rad"],
                "yaw_rate_radps": _number(state["yaw_rate_radps"], "yaw rate"),
                "lateral_route_deviation_m": state["lateral_route_deviation_m"],
                "route_progress_m": state["route_progress_m"],
                "route_highwater_m": state["route_highwater_m"],
                "next_checkpoint_index": info.get("next_checkpoint_index"),
                "completed_checkpoint_id": info.get("completed_checkpoint_id"),
                "directed_crossings": (info.get("race") or {}).get("directed_crossings", []),
                "finished": bool(info.get("finished", False)),
                "reward": _number(rewards[index], "reward"),
                "registered_reward": _number(demo.get("registered_reward"), "registered reward") if demo else
                                    sum(_number(v, "reward component") for v in reward_components.values()),
                "reward_components": reward_components, "demo_training": demo,
                "terminated": bool(done and not truncated), "truncated": truncated,
                "termination_reason": termination_reason,
                "ppo_epoch_updates": int(getattr(model, "_n_updates", 0)),
                "optimizer_steps": self.optimizer_steps, "latest_ppo_metrics": self._metrics,
                "policy_weights_sha256": self._weight_sha256,
                "latest_checkpoint": self._checkpoint,
            }
            self._append(record)
            self.step_rows += 1
            if dones[index]:
                self._episodes[index] += 1

    def on_update(self, model) -> None:
        backend = model._model
        metrics: dict[str, float | None] = {key: None for key in _METRIC_KEYS}
        for key, value in backend.logger.name_to_value.items():
            if key.startswith("train/"):
                metrics[key] = _number(value, key)
        self._update_index = int(backend._n_updates)
        self._weight_sha256 = _digest(model.weights_sha256(), "policy weight identity")
        self._metrics = metrics
        self._append({"event_type": "ppo_update", "schema_version": self._schema,
                      "ppo_epoch_updates": self._update_index,
                      "optimizer_steps": self.optimizer_steps, "metrics": metrics,
                      "policy_weights_sha256": self._weight_sha256})
        self.update_rows += 1

    def record_checkpoint(self, name: str, *, timestep: int,
                          archive_sha256: str, policy_weights_sha256: str) -> None:
        archive_sha256 = _digest(archive_sha256, "checkpoint archive identity")
        policy_weights_sha256 = _digest(policy_weights_sha256, "checkpoint policy identity")
        self._checkpoint = {"name": name, "archive_sha256": archive_sha256,
                            "policy_weights_sha256": policy_weights_sha256,
                            "timestep": int(timestep)}
        self._append({"event_type": "checkpoint", "schema_version": self._schema,
                      **self._checkpoint})

    def close(self) -> None:
        if not self._closed:
            error = None
            try:
                self._sync()
            except BaseException as exc:
                error = exc
            try:
                self._stream.close()
            except BaseException as exc:
                if error is None:
                    error = exc
            self._closed = True
            if error is not None:
                raise error

    def manifest(self, *, complete: bool) -> dict:
        if not self._closed:
            raise RuntimeError("close telemetry before creating its manifest")
        if type(complete) is not bool:
            raise ValueError("complete must be bool")
        digest = sha256()
        byte_count = event_count = step_rows = update_rows = training_starts = run_rows = 0
        with self.path.open("rb") as stream:
            while True:
                line = stream.readline(_MAX_RECORD_BYTES + 1)
                if not line:
                    break
                if len(line) > _MAX_RECORD_BYTES or not line.endswith(b"\n"):
                    raise ValueError("telemetry JSONL contains an oversized or incomplete record")
                try:
                    record = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"invalid JSON constant: {value}")))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError("telemetry JSONL contains a corrupt record") from exc
                if not isinstance(record, dict) or record.get("schema_version") != self._schema:
                    raise ValueError("telemetry JSONL record schema is corrupt")
                if record.get("event_type") not in {
                    "run", "training_start", "step", "ppo_update", "checkpoint",
                }:
                    raise ValueError("telemetry JSONL event type is corrupt")
                event_count += 1
                step_rows += record.get("event_type") == "step"
                update_rows += record.get("event_type") == "ppo_update"
                training_starts += record.get("event_type") == "training_start"
                run_rows += record.get("event_type") == "run"
                digest.update(line)
                byte_count += len(line)
        if (byte_count != self._bytes or event_count != self._events
                or step_rows != self.step_rows or update_rows != self.update_rows
                or digest.hexdigest() != self._content_hash.hexdigest()
                or self.path.stat().st_size != byte_count):
            raise ValueError("telemetry JSONL integrity counters or checksum do not match")
        if complete:
            if self._n_envs is None or self._total_timesteps is None or training_starts != 1 or run_rows != 1:
                raise ValueError("complete telemetry requires one run and one training-start record")
            if not self._total_timesteps <= step_rows <= self._total_timesteps + self._n_envs - 1:
                raise ValueError("complete telemetry step count does not match the declared budget")
        return {"schema_version": self._schema, "path": self.path.name,
                "sha256": digest.hexdigest(), "bytes": byte_count, "events": event_count,
                "step_rows": self.step_rows, "update_rows": self.update_rows,
                "optimizer_steps": self.optimizer_steps, "complete": complete}
