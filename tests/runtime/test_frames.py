import numpy as np
import pytest

from src.runtime.frames import build_runtime_frame


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

def test_flight_event_kinds_come_from_authoritative_physics_transition():
    state = {**snapshot(), "grounded": False, "physical_event_types": ["FlightTransition"]}
    info = {"physics": {"transition": {"events": ({"kind": "takeoff", "time_s": .003},)}}}
    frame = build_runtime_frame(
        state, np.zeros(3, dtype=np.float32), info,
        previous_velocity=(0., 0., 0.), collision_count_before=0,
        completed_checkpoint_ids_before=(),
    )
    assert not frame.grounded
    assert frame.physical_event_types == ["takeoff"]

def test_landing_kind_from_authoritative_transition():
    state = {**snapshot(), "grounded": True, "physical_event_types": ["FlightTransition"]}
    info = {"physics": {"transition": {"events": ({"kind": "landing", "time_s": .002},)}}}
    frame = build_runtime_frame(
        state, np.zeros(3, dtype=np.float32), info,
        previous_velocity=(0., 0., 0.), collision_count_before=0,
        completed_checkpoint_ids_before=(),
    )
    assert frame.grounded
    assert frame.physical_event_types == ["landing"]

def test_collision_replay_rejects_invalid_episode_result_in_backend():
    from pydantic import ValidationError
    from src.runtime.frames import ReplayDocument

    frame = build_runtime_frame(
        snapshot(), np.zeros(3, dtype=np.float32), {},
        previous_velocity=(0.0, 0.0, 0.0),
        collision_count_before=0, completed_checkpoint_ids_before=(),
    )
    digest = "a" * 64
    document = {
        "schema": "f1rl_racing_sandbox_replay_v2",
        "label": "collision",
        "provenance": {
            "dataset_split": "DEVELOPMENT", "source_sha256": digest,
            "model_id": "auto-drive", "track_id": "custom",
            "track_sha256": digest, "seed": 412,
        },
        "result": {
            "status": "failed", "reason": "collision",
            "valid_sample": False, "successful_finish": False,
            "steps": 4, "duration_s": .08, "trace_sha256": digest,
        },
        "frames": [frame.model_dump()],
        "collision_terminal": {
            "schema_version": "f1rl_runtime_collision_failure_dev_v1",
            "obstacle_ids": ["cone"], "last_accepted_tick": 4,
            "witness_interval_s": [.08, .1],
            "diagnostic_penalty": -5.0,
            "penalty_kind": "runtime_diagnostic_not_gym_reward",
            "physics_transition_accepted": False,
            "contact_time_certified": False,
            "scientific_evidence": False,
        },
    }
    assert ReplayDocument.model_validate(document).result.reason == "collision"
    for field, invalid in (
        ("status", "finished"), ("reason", "off_route"),
        ("valid_sample", True), ("successful_finish", True),
    ):
        broken = {**document, "result": {**document["result"], field: invalid}}
        with pytest.raises(ValidationError, match="unscored collision requires"):
            ReplayDocument.model_validate(broken)
