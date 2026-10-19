import json
from types import SimpleNamespace

import numpy as np
import pytest

from f1rl.learning.training.telemetry import TrainingTelemetryRecorder


class _Model:
    num_timesteps = 2
    _n_updates = 3
    logger = SimpleNamespace(name_to_value={
        "train/policy_gradient_loss": -0.125,
        "train/value_loss": 0.75,
        "train/entropy_loss": -0.5,
        "train/explained_variance": 0.25,
    })

    def weights_sha256(self):
        return "a" * 64


class _ModelAdapter:
    def __init__(self):
        self._model = _Model()

    def weights_sha256(self):
        return self._model.weights_sha256()


def _info(*, done=False):
    return {
        "track_id": "straight-sprint-v1",
        "train_reset": {"seed": 1000, "rank": 0, "reset_index": 2, "draw_seed": 2000,
                        "track_id": "straight-sprint-v1", "configuration_sha256": "d" * 64},
        "reward_components": {"progress": 0.5, "time": -0.1},
        "completed_checkpoint_id": "checkpoint-0" if done else None,
        "research_outcome": {"kind": "active", "reason": "accepted_step"},
        "training_telemetry": {
            "schema_version": "f1rl_training_state_v1",
            "track_id": "straight-sprint-v1",
            "seed_identity": {"seed": 1000, "rank": 0, "reset_index": 2, "draw_seed": 2000,
                              "track_id": "straight-sprint-v1", "configuration_sha256": "d" * 64},
            "observation_schema": {"version": "f1rl_observation_features_v1",
                                   "feature_names": ["target_x", "target_z"]},
            "position": (1.0, 0.0, 2.0),
            "velocity": (0.0, 0.0, 3.0),
            "speed": 3.0,
            "grounded": True,
            "heading_rad": 0.25,
            "yaw_rate_radps": 0.1,
            "lateral_route_deviation_m": 0.4,
            "route_progress_m": 2.0,
            "route_highwater_m": 2.0,
            "next_checkpoint_index": 1,
            "completed_checkpoint_id": "checkpoint-0" if done else None,
            "finished": False,
            "action_contract_version": "steering-throttle-brake0-v1",
        },
    }


def test_recorder_writes_step_update_and_checkpoint_identity(tmp_path):
    path = tmp_path / "training.jsonl"
    recorder = TrainingTelemetryRecorder(
        path,
        run_identity={"run_id": "run-1", "source_commit": "b" * 40,
                      "policy_id": "policy-1", "split": "TRAIN",
                      "track_ids": ["straight-sprint-v1"],
                      "seed_identity": {"ppo": 0, "sampling": 0}},
        max_timesteps=4,
    )
    model = _ModelAdapter()
    recorder.on_training_start(model, total_timesteps=4, n_envs=1)
    recorder.record_checkpoint("untrained.zip", timestep=0,
                                archive_sha256="c" * 64,
                                policy_weights_sha256="a" * 64)
    locals_ = {
        "obs": np.array([[0.25, 0.75]], dtype=np.float32),
        "actions": np.array([[1.25, -0.25]], dtype=np.float32),
        "clipped_actions": np.array([[1.0, 0.0]], dtype=np.float32),
        "rewards": np.array([0.4], dtype=np.float32),
        "dones": np.array([True]),
        "infos": [_info(done=True)],
    }
    recorder.on_step(locals_, model._model)
    recorder.on_optimizer_step()
    recorder.on_update(model)
    recorder.close()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    step = next(row for row in events if row["event_type"] == "step")
    update = next(row for row in events if row["event_type"] == "ppo_update")
    assert step["global_timestep"] == 2
    assert step["episode_id"] == "run-1/env-0/episode-0"
    assert step["track_id"] == "straight-sprint-v1"
    assert step["seed_identity"]["train_reset"]["seed"] == 1000
    assert step["observation"] == [0.25, 0.75]
    assert step["raw_ppo_action"] == [1.25, -0.25]
    assert step["action_sent_to_environment"] == [1.0, 0.0]
    assert step["applied_control"] == {"steering": 1.0, "throttle": 0.0, "brake": 0.0}
    assert step["vehicle_position"] == [1.0, 0.0, 2.0]
    assert step["reward_components"] == {"progress": 0.5, "time": -0.1}
    assert step["terminated"] is True
    assert step["completed_checkpoint_id"] == "checkpoint-0"
    assert step["policy_weights_sha256"] == "a" * 64
    assert update["optimizer_steps"] == 1
    assert update["ppo_epoch_updates"] == 3
    assert update["metrics"]["train/value_loss"] == 0.75
    assert update["metrics"]["train/approx_kl"] is None
    assert update["policy_weights_sha256"] == "a" * 64
    assert recorder.manifest(complete=True)["step_rows"] == 1
    assert recorder.manifest(complete=True)["update_rows"] == 1


def test_recorder_fails_closed_on_nonfinite_observation(tmp_path):
    recorder = TrainingTelemetryRecorder(
        tmp_path / "training.jsonl",
        run_identity={"run_id": "run-2", "source_commit": "b" * 40,
                      "policy_id": "policy-1", "split": "TRAIN",
                      "track_ids": ["straight-sprint-v1"],
                      "seed_identity": {"ppo": 0}},
        max_timesteps=2,
    )
    model = _ModelAdapter()
    recorder.on_training_start(model, total_timesteps=2, n_envs=1)
    locals_ = {
        "obs": np.array([[np.nan, 0.0]], dtype=np.float32),
        "actions": np.zeros((1, 2), dtype=np.float32),
        "clipped_actions": np.zeros((1, 2), dtype=np.float32),
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.zeros(1, dtype=bool),
        "infos": [_info()],
    }
    with pytest.raises(ValueError, match="finite"):
        recorder.on_step(locals_, model._model)
    recorder.close()


def test_recorder_requires_reset_seed_identity_on_every_transition(tmp_path):
    with pytest.raises(ValueError, match="identified TRAIN"):
        TrainingTelemetryRecorder(
            tmp_path / "empty-seeds.jsonl",
            run_identity={"run_id": "run-seed", "source_commit": "b" * 40,
                          "policy_id": "policy-1", "split": "TRAIN",
                          "track_ids": ["straight-sprint-v1"], "seed_identity": {}},
            max_timesteps=2,
        )
    recorder = TrainingTelemetryRecorder(
        tmp_path / "missing-reset.jsonl",
        run_identity={"run_id": "run-seed", "source_commit": "b" * 40,
                      "policy_id": "policy-1", "split": "TRAIN",
                      "track_ids": ["straight-sprint-v1"], "seed_identity": {"ppo": 0}},
        max_timesteps=2,
    )
    recorder.on_training_start(_ModelAdapter(), total_timesteps=2, n_envs=1)
    info = _info()
    info.pop("train_reset")
    with pytest.raises(ValueError, match="reset seed identity"):
        recorder.on_step({
            "obs": np.zeros((1, 2), dtype=np.float32),
            "actions": np.zeros((1, 2), dtype=np.float32),
            "clipped_actions": np.zeros((1, 2), dtype=np.float32),
            "rewards": np.zeros(1, dtype=np.float32),
            "dones": np.zeros(1, dtype=bool), "infos": [info],
        }, _Model())
    recorder.close()


def test_recorder_rejects_tracks_outside_registered_train_set(tmp_path):
    recorder = TrainingTelemetryRecorder(
        tmp_path / "training.jsonl",
        run_identity={"run_id": "run-3", "source_commit": "b" * 40,
                      "policy_id": "policy-1", "split": "TRAIN",
                      "track_ids": ["straight-sprint-v1"], "seed_identity": {"ppo": 0}},
        max_timesteps=2,
    )
    model = _ModelAdapter()
    recorder.on_training_start(model, total_timesteps=2, n_envs=1)
    heldout = _info()
    heldout["track_id"] = "heldout-track-v1"
    heldout["train_reset"]["track_id"] = "heldout-track-v1"
    heldout["training_telemetry"]["track_id"] = "heldout-track-v1"
    heldout["training_telemetry"]["seed_identity"]["track_id"] = "heldout-track-v1"
    locals_ = {
        "obs": np.zeros((1, 2), dtype=np.float32),
        "actions": np.zeros((1, 2), dtype=np.float32),
        "clipped_actions": np.zeros((1, 2), dtype=np.float32),
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.zeros(1, dtype=bool),
        "infos": [heldout],
    }
    with pytest.raises(ValueError, match="TRAIN set"):
        recorder.on_step(locals_, model._model)
    recorder.close()


def test_recorder_fails_closed_on_oversized_update_row(tmp_path):
    recorder = TrainingTelemetryRecorder(
        tmp_path / "training.jsonl",
        run_identity={"run_id": "run-4", "source_commit": "b" * 40,
                      "policy_id": "policy-1", "split": "TRAIN",
                      "track_ids": ["straight-sprint-v1"], "seed_identity": {"ppo": 0}},
        max_timesteps=2,
    )
    model = _ModelAdapter()
    recorder.on_training_start(model, total_timesteps=2, n_envs=1)
    model._model.logger.name_to_value = {"train/" + "x" * 20_000: 1.0}
    with pytest.raises(ValueError, match="storage bound"):
        recorder.on_update(model)
    recorder.close()
