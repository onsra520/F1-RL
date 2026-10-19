"""Bounded, versioned observations of PPO learning; never a policy input."""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re

import numpy as np


_SCHEMA = "f1rl_training_telemetry_v1"
_TRAIN_TRACKS = {"straight-sprint-v1", "right-turn-v1", "slalom-v1", "ramp-v1"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MAX_RECORD_BYTES = 16 * 1024
_MAX_LOG_BYTES = 256 * 1024 * 1024


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

    def __init__(self, path: str | Path, *, run_identity: dict, max_timesteps: int) -> None:
        if type(max_timesteps) is not int or max_timesteps <= 0:
            raise ValueError("max_timesteps must be a positive integer")
        required = {"run_id", "source_commit", "policy_id", "split", "track_ids", "seed_identity"}
        if not isinstance(run_identity, dict) or set(run_identity) != required:
            raise ValueError("run_identity has unexpected fields")
        if (not all(isinstance(run_identity[k], str) and run_identity[k].strip()
                    for k in ("run_id", "policy_id"))
                or _COMMIT.fullmatch(run_identity["source_commit"]) is None
                or run_identity["split"] != "TRAIN"
                or not isinstance(run_identity["track_ids"], (list, tuple))
                or not run_identity["track_ids"]
                or any(track not in _TRAIN_TRACKS for track in run_identity["track_ids"])
                or not isinstance(run_identity["seed_identity"], dict)):
            raise ValueError("training telemetry requires an identified TRAIN run")
        self.path = Path(path)
        if not self.path.parent.is_dir():
            raise ValueError("telemetry output directory must already exist")
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._run = dict(run_identity)
        self._max_timesteps = max_timesteps
        self._max_bytes = min(_MAX_LOG_BYTES, max_timesteps * _MAX_RECORD_BYTES)
        self._bytes = 0
        self._events = 0
        self.step_rows = 0
        self.update_rows = 0
        self.optimizer_steps = 0
        self._n_envs = None
        self._episodes: dict[int, int] = {}
        self._weight_sha256 = None
        self._update_index = 0
        self._metrics: dict[str, float] = {}
        self._checkpoint = None
        self._observation_schema = None
        self._closed = False
        self._append({"event_type": "run", "schema_version": _SCHEMA,
                      **self._run, "max_timesteps": max_timesteps})

    def _append(self, record: dict) -> None:
        if self._closed:
            raise RuntimeError("telemetry recorder is closed")
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        size = len(line.encode("utf-8"))
        if size > _MAX_RECORD_BYTES or self._bytes + size > self._max_bytes:
            raise ValueError("training telemetry storage bound exceeded")
        self._stream.write(line)
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
        self._n_envs = n_envs
        self._episodes = {i: 0 for i in range(n_envs)}
        self._weight_sha256 = _digest(model.weights_sha256(), "policy weight identity")
        self._append({"event_type": "training_start", "schema_version": _SCHEMA,
                      "total_timesteps": total_timesteps, "n_envs": n_envs,
                      "policy_weights_sha256": self._weight_sha256})

    def on_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def on_step(self, local_values: dict, model) -> None:
        if self._n_envs is None:
            raise RuntimeError("training start must be recorded before steps")
        obs = np.asarray(local_values["obs"])
        raw_actions = np.asarray(local_values["actions"])
        applied_actions = np.asarray(local_values["clipped_actions"])
        rewards = np.asarray(local_values["rewards"]).reshape(-1)
        dones = np.asarray(local_values["dones"], dtype=bool).reshape(-1)
        infos = local_values["infos"]
        expected = (self._n_envs,)
        if (obs.shape[0] != self._n_envs or raw_actions.shape[0] != self._n_envs
                or applied_actions.shape != raw_actions.shape or rewards.shape != expected
                or dones.shape != expected or len(infos) != self._n_envs):
            raise ValueError("PPO callback transition batch shape changed")
        final_timestep = int(model.num_timesteps)
        if final_timestep > self._max_timesteps + self._n_envs:
            raise ValueError("training exceeded telemetry timestep bound")
        for index, info in enumerate(infos):
            state = info.get("training_telemetry")
            if not isinstance(state, dict) or state.get("schema_version") != "f1rl_training_state_v1":
                raise ValueError("environment training telemetry is missing or has an unsupported schema")
            track_id = info.get("track_id") or state.get("track_id")
            if track_id not in self._run["track_ids"] or track_id not in _TRAIN_TRACKS:
                raise ValueError("transition track is outside the preregistered TRAIN set")
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
                "event_type": "step", "schema_version": _SCHEMA,
                "source_commit": self._run["source_commit"], "run_id": self._run["run_id"],
                "policy_id": self._run["policy_id"], "global_timestep": global_timestep,
                "env_index": index, "episode_id": f"{self._run['run_id']}/env-{index}/episode-{episode_index}",
                "episode_index": episode_index, "track_id": track_id,
                "seed_identity": {"run": self._run["seed_identity"],
                                  "train_reset": info.get("train_reset")},
                "observation_schema_version": schema["version"],
                "observation_feature_names": schema["feature_names"],
                "observation_schema_sha256": schema_digest,
                "observation": [_number(v, "observation") for v in obs[index].reshape(-1)],
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
        metrics = {}
        for key, value in backend.logger.name_to_value.items():
            if key.startswith("train/"):
                metrics[key] = _number(value, key)
        self._update_index = int(backend._n_updates)
        self._weight_sha256 = _digest(model.weights_sha256(), "policy weight identity")
        self._metrics = metrics
        self._append({"event_type": "ppo_update", "schema_version": _SCHEMA,
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
        self._append({"event_type": "checkpoint", "schema_version": _SCHEMA,
                      **self._checkpoint})

    def close(self) -> None:
        if not self._closed:
            self._sync()
            self._stream.close()
            self._closed = True

    def manifest(self, *, complete: bool) -> dict:
        if not self._closed:
            raise RuntimeError("close telemetry before creating its manifest")
        if type(complete) is not bool:
            raise ValueError("complete must be bool")
        return {"schema_version": _SCHEMA, "path": self.path.name,
                "sha256": sha256(self.path.read_bytes()).hexdigest(),
                "bytes": self.path.stat().st_size, "events": self._events,
                "step_rows": self.step_rows, "update_rows": self.update_rows,
                "optimizer_steps": self.optimizer_steps, "complete": complete}
