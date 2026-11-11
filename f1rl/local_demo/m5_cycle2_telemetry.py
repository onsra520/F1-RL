from __future__ import annotations

import json
from pathlib import Path
from hashlib import sha256

import numpy as np
from ..learning.action import BrakingPolicyAction, map_policy_action


class M5Cycle2Telemetry:
    def __init__(self, path: Path, *, run_id: str, expected_steps: int) -> None:
        self.path = path
        self.run_id = run_id
        self.expected_steps = expected_steps
        self.steps = 0
        self.optimizer_steps = 0
        self.updates = 0
        self.track_exposure = {track: 0 for track in (
            "straight-sprint-v1", "slalom-v1", "right-turn-train-v2", "ramp-v1",
            "left-turn-train-v2")}
        self.gradients_finite = True
        self.nonzero_gradient_steps = 0
        self._model = None
        self._stream = None

    def _write(self, event: dict) -> None:
        json.dumps(event, allow_nan=False)
        self._stream.write(json.dumps(event, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False) + "\n")

    def on_training_start(self, model, *, total_timesteps: int, n_envs: int) -> None:
        if total_timesteps != self.expected_steps or n_envs != 1:
            raise ValueError("M5 cycle telemetry budget/vector contract mismatch")
        self._model = model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n", buffering=1)
        self._write({"event_type": "chunk_start", "run_id": self.run_id,
                     "expected_steps": total_timesteps, "n_envs": n_envs,
                     "policy_weights_sha256": model.weights_sha256()})

    def on_step(self, locals_: dict, model) -> None:
        infos = locals_.get("infos")
        actions = locals_.get("actions")
        observations = locals_.get("obs")
        rewards = locals_.get("rewards")
        dones = locals_.get("dones")
        if not isinstance(infos, (list, tuple)) or len(infos) != 1:
            raise ValueError("M5 cycle telemetry requires one environment info")
        info = infos[0]
        track_id = info.get("track_id")
        if track_id not in self.track_exposure:
            raise ValueError(f"M5 cycle sampled an unregistered track: {track_id}")
        raw_action = np.asarray(actions)[0].astype(float).tolist()
        action_sent = np.asarray(locals_.get("clipped_actions", actions))[0].astype(float).tolist()
        mapped = map_policy_action(BrakingPolicyAction(*action_sent))
        applied_action = {"steering": float(mapped.steering), "throttle": float(mapped.throttle),
                          "brake": float(mapped.brake)}
        if not np.allclose(action_sent, list(applied_action.values()), atol=1e-7, rtol=0):
            raise ValueError("Cycle 2 environment action does not match the registered action mapper")
        observation = np.asarray(observations)[0].astype(float).tolist()
        reward = float(np.asarray(rewards)[0])
        done = bool(np.asarray(dones)[0])
        self.track_exposure[track_id] += 1
        self.steps += 1
        self._write({"event_type": "transition", "run_id": self.run_id,
                     "chunk_timestep": self.steps, "global_timestep": int(model.num_timesteps),
                     "track_id": track_id, "observation": observation,
                     "raw_ppo_action": raw_action,
                     "action_sent_to_environment": action_sent,
                     "applied_action": applied_action,
                     "reward": reward, "done": done,
                     "reward_components": info.get("reward_components", {}),
                     "next_checkpoint_index": info.get("next_checkpoint_index"),
                     "collision_obstacle_ids": info.get("telemetry", {}).get("obstacle_hit_ids", []),
                     "terminal": info.get("research_outcome"),
                     "physics_sha256": info.get("physics", {}).get("sha256")})

    def on_optimizer_step(self) -> None:
        self.optimizer_steps += 1
        if self._model is not None:
            integrity = self._model.optimizer_gradient_integrity()
            self.gradients_finite = self.gradients_finite and bool(integrity["finite"])
            self.nonzero_gradient_steps += int(integrity["nonzero_gradient_elements"] > 0)

    def on_update(self, model) -> None:
        self.updates += 1
        self._write({"event_type": "ppo_update", "run_id": self.run_id,
                     "global_timestep": model.num_timesteps,
                     "policy_weights_sha256": model.weights_sha256(),
                     "gradient_integrity": {"finite": self.gradients_finite,
                                            "optimizer_steps": self.optimizer_steps,
                                            "nonzero_gradient_steps": self.nonzero_gradient_steps}})

    def close(self) -> dict:
        if self._stream is None:
            raise RuntimeError("M5 cycle telemetry was not started")
        if self.steps != self.expected_steps or self.optimizer_steps == 0 or self.updates == 0:
            raise ValueError("M5 cycle telemetry is incomplete")
        self._write({"event_type": "chunk_complete", "run_id": self.run_id,
                     "step_count": self.steps, "optimizer_steps": self.optimizer_steps,
                     "update_count": self.updates, "gradient_finite": self.gradients_finite,
                     "nonzero_gradient_steps": self.nonzero_gradient_steps,
                     "track_exposure_steps": self.track_exposure})
        self._stream.flush()
        self._stream.close()
        return {"path": str(self.path), "bytes": self.path.stat().st_size,
                "sha256": sha256(self.path.read_bytes()).hexdigest(),
                "step_count": self.steps, "optimizer_steps": self.optimizer_steps,
                "update_count": self.updates, "track_exposure_steps": self.track_exposure,
                "gradient_finite": self.gradients_finite,
                "nonzero_gradient_steps": self.nonzero_gradient_steps}
