import math

import pytest

from scripts.m5_cycle1_validate import summarize_actions


def test_action_summary_uses_applied_vehicle_controls_and_checkpoints():
    rows = [
        {"applied_vehicle_control": {"steering": -0.5, "throttle": 0.8, "brake": 0.0},
         "next_checkpoint_index": 1},
        {"applied_vehicle_control": {"steering": 0.5, "throttle": 0.4, "brake": 0.2},
         "next_checkpoint_index": 2},
    ]

    summary = summarize_actions(rows)
    assert math.isclose(summary["mean_steering"], 0.0)
    assert math.isclose(summary["mean_throttle"], 0.6)
    assert math.isclose(summary["mean_brake"], 0.1)
    assert summary["checkpoint_indices"] == [1, 2]


def test_action_summary_rejects_missing_telemetry():
    with pytest.raises(ValueError, match="cannot be empty"):
        summarize_actions([])
