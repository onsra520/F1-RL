from __future__ import annotations

from pathlib import Path

import numpy as np

from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.custom_tracks import convert_custom_track


class DeterministicThrottlePolicy:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, observation: np.ndarray) -> np.ndarray:
        assert np.isfinite(observation).all()
        self.calls += 1
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def test_track_builder_geometry_reaches_v4_finish_through_fake_policy(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "/assets/tracks/evaluation/" in path.as_posix().lower():
            raise PermissionError("HELD-OUT geometry is forbidden in this test")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    root = Path(__file__).parents[2]
    catalog = RuntimeCatalog(root)
    track = {
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "fake-policy-pipeline",
        "name": "Fake policy pipeline",
        "width": 10,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 90}],
        "props": [{"id": "checkpoint-1", "kind": "checkpoint", "t": 0.55, "offset": 0, "rotation": 0}],
    }
    converted = convert_custom_track(track, catalog)
    assert converted.ppo_compatible is True
    assert converted.factory.physics_config.ground_motion_version == "spatial_road_edge_partial_lifecycle_v4"
    assert converted.factory.race_config.version == "wp7_directed_lifecycle_crash_contract_v4"

    policy = DeterministicThrottlePolicy()
    env = converted.factory.create()
    try:
        observation, _ = env.reset(seed=20261108)
        assert env.observation_space.contains(observation)
        terminated = truncated = False
        info = {}
        for _ in range(1000):
            action = policy.predict(observation)
            assert env.action_space.contains(action)
            observation, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)
            if terminated or truncated:
                break

        assert policy.calls > 0
        assert terminated
        assert not truncated
        assert info["finished"] is True
        assert info["next_checkpoint_index"] == 1
        assert info.get("failure_reason") is None
    finally:
        env.close()
