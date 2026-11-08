import numpy as np
import pytest

from f1rl.local_demo.runtime.frames import build_runtime_frame


def snapshot():
    return {
        "schema_version": "f1rl_runtime_snapshot_v1",
        "tick": 4,
        "time_s": 0.08,
        "dt_s": 0.02,
        "position": (1.0, 0.5, 3.0),
        "velocity": (4.0, 0.0, 2.0),
        "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
        "heading_rad": 0.0,
        "speed_mps": 4.5,
        "grounded": True,
        "route_progress_m": 8.0,
        "route_progress_fraction": 0.2,
        "route_highwater_m": 9.0,
        "next_checkpoint_index": 1,
        "checkpoint_count": 3,
    }


def test_runtime_frame_captures_actual_state_action_and_events():
    frame = build_runtime_frame(
        snapshot(),
        np.array([0.5, 0.75, 0.1], dtype=np.float32),
        {
            "telemetry": {"obstacle_hit_ids": ("cone-1",)},
            "race": {
                "completed_checkpoint_ids": ("cp-1",),
                "directed_crossings": ("cp-1",),
            },
            "research_outcome": {"kind": "active", "valid_sample": True},
        },
        previous_velocity=(1.0, 0.0, 0.0),
        collision_count_before=0,
        completed_checkpoint_ids_before=(),
    )

    assert frame.schema_version == "f1rl_runtime_frame_v1"
    assert frame.tick == 4 and frame.time_s == 0.08
    assert frame.position == (1.0, 0.5, 3.0)
    assert frame.action.steering == 0.5
    assert frame.action.throttle == 0.75
    assert frame.action.brake == pytest.approx(0.1)
    assert frame.acceleration_mps2 == pytest.approx((150.0, 0.0, 100.0))
    assert frame.collision_count == 1 and frame.collision_ids == ["cone-1"]
    assert frame.completed_checkpoint_ids == ["cp-1"]


@pytest.mark.parametrize("action", [
    np.array([1.01, 0.5, 0.0], dtype=np.float32),
    np.array([0.0, 0.5, -0.01], dtype=np.float32),
    np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, np.nan, 0.0], dtype=np.float32),
])
def test_runtime_frame_rejects_actions_outside_the_registered_contract(action):
    with pytest.raises(ValueError, match="action"):
        build_runtime_frame(
            snapshot(), action, {}, previous_velocity=(0.0, 0.0, 0.0),
            collision_count_before=0, completed_checkpoint_ids_before=(),
        )
