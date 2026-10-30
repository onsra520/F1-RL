import math
from pathlib import Path

import numpy as np

from src.runtime.catalog import RuntimeCatalog

ROOT = Path(__file__).resolve().parents[2]


def test_gym_environment_exposes_read_only_authoritative_runtime_snapshot():
    env = RuntimeCatalog(ROOT).create_environment("straight-sprint-v1", seed=412)
    try:
        env.reset(seed=412)
        before = env.runtime_snapshot()
        assert before["schema_version"] == "f1rl_runtime_snapshot_v1"
        assert before["tick"] == 0
        assert before["time_s"] == 0.0
        assert before["dt_s"] > 0.0
        assert len(before["position"]) == 3
        assert len(before["velocity"]) == 3
        assert len(before["orientation_wxyz"]) == 4
        assert math.isclose(sum(value * value for value in before["orientation_wxyz"]), 1.0)
        assert isinstance(before["grounded"], bool)
        assert before["route_progress_m"] >= 0.0
        assert env.runtime_snapshot() == before

        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        after = env.runtime_snapshot()
        assert after["tick"] == 1
        assert after["time_s"] == before["dt_s"]
        assert all(math.isfinite(value) for value in after["position"] + after["velocity"])
    finally:
        env.close()
