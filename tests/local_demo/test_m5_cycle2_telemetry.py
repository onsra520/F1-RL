import json

import numpy as np
import pytest

from f1rl.local_demo.m5_cycle2_telemetry import M5Cycle2Telemetry


class FakeModel:
    num_timesteps = 10

    def weights_sha256(self):
        return "a" * 64

    def optimizer_gradient_integrity(self):
        return {"finite": True, "nonzero_gradient_elements": 3}


def _locals(track="slalom-v1", raw_action=None, action_sent=None):
    if raw_action is None:
        raw_action = [0.2, 0.7, 0.0]
    if action_sent is None:
        action_sent = raw_action
    return {
        "infos": [{"track_id": track, "reward_components": {"route_progress": 1.0},
                   "race": {"next_checkpoint_index": 2},
                   "telemetry": {"obstacle_hit_ids": []}}],
        "actions": np.array([raw_action], dtype=np.float32),
        "clipped_actions": np.array([action_sent], dtype=np.float32),
        "obs": np.zeros((1, 16), dtype=np.float32),
        "rewards": np.array([1.0], dtype=np.float32),
        "dones": np.array([False]),
    }


def test_cycle2_telemetry_hashes_steps_optimizer_and_track_exposure(tmp_path):
    path = tmp_path / "training.jsonl"
    observer = M5Cycle2Telemetry(path, run_id="run-1", expected_steps=2)
    model = FakeModel()
    observer.on_training_start(model, total_timesteps=2, n_envs=1)
    observer.on_step(_locals(), model)
    observer.on_optimizer_step()
    observer.on_update(model)
    observer.on_step(_locals("ramp-v1"), model)
    observer.on_optimizer_step()
    observer.on_update(model)
    summary = observer.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert summary["step_count"] == 2
    assert summary["optimizer_steps"] == 2
    assert summary["update_count"] == 2
    assert summary["gradient_finite"] is True
    assert summary["track_exposure_steps"]["slalom-v1"] == 1
    assert summary["track_exposure_steps"]["ramp-v1"] == 1
    assert [row["event_type"] for row in rows] == [
        "chunk_start", "transition", "ppo_update", "transition", "ppo_update", "chunk_complete"]


def test_cycle2_telemetry_rejects_unregistered_track(tmp_path):
    observer = M5Cycle2Telemetry(tmp_path / "training.jsonl", run_id="run-2", expected_steps=1)
    observer.on_training_start(FakeModel(), total_timesteps=1, n_envs=1)
    with pytest.raises(ValueError, match="unregistered track"):
        observer.on_step(_locals("right-turn-v1"), FakeModel())
    observer._stream.close()


def test_cycle2_telemetry_preserves_raw_and_environment_clipped_actions(tmp_path):
    path = tmp_path / "training.jsonl"
    observer = M5Cycle2Telemetry(path, run_id="run-3", expected_steps=1)
    model = FakeModel()
    observer.on_training_start(model, total_timesteps=1, n_envs=1)
    observer.on_step(_locals(raw_action=[1.2, 1.4, -0.2], action_sent=[1.0, 1.0, 0.0]), model)
    observer.on_optimizer_step()
    observer.on_update(model)
    observer.close()
    transition = next(json.loads(line) for line in path.read_text().splitlines()
                      if '"transition"' in line)
    assert transition["raw_ppo_action"] == [1.2000000476837158, 1.399999976158142, -0.20000000298023224]
    assert transition["action_sent_to_environment"] == [1.0, 1.0, 0.0]
    assert transition["applied_action"] == {"steering": 1.0, "throttle": 1.0, "brake": 0.0}
