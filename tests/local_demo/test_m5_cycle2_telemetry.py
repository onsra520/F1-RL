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


def _locals(track="slalom-v1"):
    return {
        "infos": [{"track_id": track, "reward_components": {"route_progress": 1.0},
                   "race": {"next_checkpoint_index": 2},
                   "telemetry": {"obstacle_hit_ids": []}}],
        "actions": np.array([[0.2, 0.7, 0.0]], dtype=np.float32),
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
