import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from f1rl.learning.training.telemetry import TrainingTelemetryRecorder
from f1rl.learning.train_reset import derive_train_reset_draw_seed


def _reset_contracts():
    return {"straight-sprint-v1": {
        "version": "wp7_train_reset_research_v1", "seed": 1000,
        "configuration_sha256": "d" * 64, "world_sha256": "e" * 64,
    }}


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
    reset = {"version": "wp7_train_reset_research_v1", "seed": 1000, "rank": 0,
             "reset_index": 0,
             "draw_seed": derive_train_reset_draw_seed(version="wp7_train_reset_research_v1",
                 seed=1000, rank=0, track_id="straight-sprint-v1", reset_index=0),
             "track_id": "straight-sprint-v1", "configuration_sha256": "d" * 64,
             "world_sha256": "e" * 64}
    return {
        "track_id": "straight-sprint-v1",
        "train_reset": reset,
        "reward_components": {"progress": 0.5, "time": -0.1},
        "completed_checkpoint_id": "checkpoint-0" if done else None,
        "research_outcome": {"kind": "active", "reason": "accepted_step"},
        "training_telemetry": {
            "schema_version": "f1rl_training_state_v1",
            "track_id": "straight-sprint-v1",
            "seed_identity": reset,
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
                      "seed_identity": {"ppo": 0, "sampling": 0, "training_reset": 1000},
                      "reset_contracts": _reset_contracts()},
        max_timesteps=4,
    )
    model = _ModelAdapter()
    recorder.on_training_start(model, total_timesteps=1, n_envs=1)
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
                      "seed_identity": {"ppo": 0, "training_reset": 1000},
                      "reset_contracts": _reset_contracts()},
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
                          "track_ids": ["straight-sprint-v1"], "seed_identity": {},
                          "reset_contracts": {}},
            max_timesteps=2,
        )
    recorder = TrainingTelemetryRecorder(
        tmp_path / "missing-reset.jsonl",
        run_identity={"run_id": "run-seed", "source_commit": "b" * 40,
                      "policy_id": "policy-1", "split": "TRAIN",
                      "track_ids": ["straight-sprint-v1"],
                      "seed_identity": {"ppo": 0, "training_reset": 1000},
                      "reset_contracts": _reset_contracts()},
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
                      "track_ids": ["straight-sprint-v1"],
                      "seed_identity": {"ppo": 0, "training_reset": 1000},
                      "reset_contracts": _reset_contracts()},
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
                      "track_ids": ["straight-sprint-v1"],
                      "seed_identity": {"ppo": 0, "training_reset": 1000},
                      "reset_contracts": _reset_contracts()},
        max_timesteps=2,
    )
    model = _ModelAdapter()
    recorder.on_training_start(model, total_timesteps=2, n_envs=1)
    model._model.logger.name_to_value = {"train/" + "x" * 20_000: 1.0}
    with pytest.raises(ValueError, match="storage bound"):
        recorder.on_update(model)
    recorder.close()


def test_manifest_streams_and_verifies_counts_checksum_bytes_and_incomplete_state(tmp_path, monkeypatch):
    path = tmp_path / "integrity.jsonl"
    recorder = TrainingTelemetryRecorder(
        path, run_identity={"run_id": "integrity", "source_commit": "b" * 40,
            "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
            "seed_identity": {"training_reset": 1000}, "reset_contracts": _reset_contracts()},
        max_timesteps=2)
    recorder.close()
    monkeypatch.setattr(Path, "read_bytes", lambda *_: (_ for _ in ()).throw(AssertionError("read_bytes")))
    manifest = recorder.manifest(complete=False)
    assert manifest["complete"] is False
    assert manifest["events"] == 1
    assert manifest["bytes"] == path.stat().st_size
    assert manifest["sha256"] == recorder._content_hash.hexdigest()


@pytest.mark.parametrize("corruption", [b"truncated", b"{}\n", b"{\"event_type\": NaN}\n"])
def test_manifest_rejects_jsonl_corruption(tmp_path, corruption):
    path = tmp_path / "corrupt.jsonl"
    recorder = TrainingTelemetryRecorder(
        path, run_identity={"run_id": "corrupt", "source_commit": "b" * 40,
            "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
            "seed_identity": {"training_reset": 1000}, "reset_contracts": _reset_contracts()},
        max_timesteps=2)
    recorder.close()
    path.write_bytes(corruption)
    with pytest.raises(ValueError, match="telemetry JSONL"):
        recorder.manifest(complete=False)


def test_invalid_reset_contract_fails_before_exclusive_log_creation(tmp_path):
    identity = {"run_id": "invalid", "source_commit": "b" * 40,
        "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
        "seed_identity": {"training_reset": 1001}, "reset_contracts": _reset_contracts()}
    path = tmp_path / "must-not-exist.jsonl"
    with pytest.raises(ValueError, match="declared training reset seed"):
        TrainingTelemetryRecorder(path, run_identity=identity, max_timesteps=2)
    assert not path.exists()


def test_recorder_freezes_declared_reset_contract_at_construction(tmp_path):
    identity = {"run_id": "frozen", "source_commit": "b" * 40,
        "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
        "seed_identity": {"training_reset": 1000}, "reset_contracts": _reset_contracts()}
    recorder = TrainingTelemetryRecorder(tmp_path / "frozen.jsonl", run_identity=identity,
                                         max_timesteps=2)
    identity["reset_contracts"]["straight-sprint-v1"]["seed"] = 1001
    identity["track_ids"].clear()
    recorder.on_training_start(_ModelAdapter(), total_timesteps=1, n_envs=1)
    recorder.on_step({
        "obs": np.zeros((1, 2), dtype=np.float32),
        "actions": np.zeros((1, 2), dtype=np.float32),
        "clipped_actions": np.zeros((1, 2), dtype=np.float32),
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.zeros(1, dtype=bool), "infos": [_info()],
    }, _Model())
    recorder.close()
    assert recorder.manifest(complete=True)["step_rows"] == 1


def test_storage_preflight_reserves_the_full_declared_byte_bound(tmp_path, monkeypatch):
    import f1rl.learning.training.telemetry as telemetry
    monkeypatch.setattr(telemetry.shutil, "disk_usage", lambda _path: SimpleNamespace(
        total=100_000_000, used=99_000_000, free=1_000_000))
    with pytest.raises(OSError, match="insufficient free space"):
        TrainingTelemetryRecorder(
            tmp_path / "no-space.jsonl",
            run_identity={"run_id": "capacity", "source_commit": "b" * 40,
                "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
                "seed_identity": {"training_reset": 1000}, "reset_contracts": _reset_contracts()},
            max_timesteps=100)
    assert not (tmp_path / "no-space.jsonl").exists()


def test_complete_manifest_rejects_partial_training_budget(tmp_path):
    recorder = TrainingTelemetryRecorder(
        tmp_path / "partial.jsonl",
        run_identity={"run_id": "partial", "source_commit": "b" * 40,
            "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
            "seed_identity": {"training_reset": 1000}, "reset_contracts": _reset_contracts()},
        max_timesteps=4)
    model = _ModelAdapter()
    recorder.on_training_start(model, total_timesteps=2, n_envs=1)
    recorder.on_step({
        "obs": np.zeros((1, 2), dtype=np.float32),
        "actions": np.zeros((1, 2), dtype=np.float32),
        "clipped_actions": np.zeros((1, 2), dtype=np.float32),
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.zeros(1, dtype=bool), "infos": [_info()],
    }, model._model)
    recorder.close()
    assert recorder.manifest(complete=False)["step_rows"] == 1
    with pytest.raises(ValueError, match="step count"):
        recorder.manifest(complete=True)


def test_storage_budget_over_hard_cap_is_rejected_before_file_creation(tmp_path):
    path = tmp_path / "over-cap.jsonl"
    with pytest.raises(ValueError, match="4 GiB capacity"):
        TrainingTelemetryRecorder(
            path, run_identity={"run_id": "over-cap", "source_commit": "b" * 40,
                "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
                "seed_identity": {"training_reset": 1000}, "reset_contracts": _reset_contracts()},
            max_timesteps=300_000)
    assert not path.exists()


def test_initial_oversized_run_record_is_cleaned_up_after_exclusive_create(tmp_path):
    path = tmp_path / "oversized-run.jsonl"
    with pytest.raises(ValueError, match="storage bound"):
        TrainingTelemetryRecorder(
            path, run_identity={"run_id": "r" * 20_000, "source_commit": "b" * 40,
                "policy_id": "policy", "split": "TRAIN", "track_ids": ["straight-sprint-v1"],
                "seed_identity": {"training_reset": 1000}, "reset_contracts": _reset_contracts()},
            max_timesteps=2)
    assert not path.exists()
