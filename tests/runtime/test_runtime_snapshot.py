from __future__ import annotations

from pathlib import Path

import numpy as np

from f1rl.learning.reconstruction import RecoveryCandidate, build_training_scenarios
from f1rl.local_demo.runtime.adapter import RuntimeEnvironmentAdapter


def test_runtime_snapshot_tracks_a_train_environment_without_opening_heldout_geometry(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        normalized = path.as_posix().lower()
        if "/assets/tracks/evaluation/" in normalized:
            raise PermissionError("HELD-OUT geometry is forbidden in this test")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    candidate = RecoveryCandidate.from_yaml(
        Path(__file__).parents[2] / "configs/recovery/wp7-c2-five-track-paired-v2.yml"
    )
    scenarios = build_training_scenarios(candidate)
    assert tuple(item.track_id for item in scenarios.scenarios) == (
        "straight-sprint-v1",
        "slalom-v1",
        "right-turn-train-v2",
        "ramp-v1",
        "left-turn-train-v2",
    )

    for scenario in scenarios.scenarios:
        env = RuntimeEnvironmentAdapter(scenario.factory.create())
        try:
            observation, _ = env.reset(seed=20261108)
            before = env.runtime_snapshot()
            assert before["schema_version"] == "f1rl_runtime_snapshot_v1"
            assert before["tick"] == 0
            assert before["route_progress_m"] == 0.0
            assert np.isfinite(observation).all()

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_observation, _, terminated, truncated, _ = env.step(action)
            after = env.runtime_snapshot()
            assert after["tick"] == 1
            assert after["time_s"] == after["dt_s"]
            assert not terminated
            assert not truncated
            assert np.isfinite(next_observation).all()
        finally:
            env.close()
