from pathlib import Path

import pytest

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import convert_custom_track
from src.runtime.session import (
    MAX_ACTIVE_SESSIONS,
    ManualActionController,
    RuntimeFault,
    RuntimeSessionManager,
)

ROOT = Path(__file__).resolve().parents[2]


def test_manual_action_controller_returns_only_bounded_latest_action():
    controller = ManualActionController()
    action = controller.predict(None)
    assert action.tolist() == [0.0, 0.0, 0.0]
    controller.set_action(-1.0, 1.0, 0.25)
    action = controller.predict(None)
    assert action.tolist() == [-1.0, 1.0, 0.25]
    controller.stop()
    action = controller.predict(None)
    assert action.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        controller.set_action(1.1, 0.0, 0.0)


@pytest.mark.parametrize(("model_id", "track_id"), [
    ("A-seed0", "straight-sprint-v1"),
    ("B-seed1", "slalom-v1"),
])
def test_frozen_model_runs_real_train_environment_and_returns_replay(model_id, track_id):
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        session = manager.create(model_id=model_id, track_id=track_id, seed=412)
        session.start()
        assert session.wait(120.0)

        status = session.status_document()
        replay = session.replay()
        assert status["status"] in {"finished", "failed"}
        assert status["provenance"] == "TRAIN-development"
        assert replay["schema"] == "f1rl_racing_sandbox_replay_v2"
        assert replay["provenance"]["dataset_split"] == "TRAIN"
        assert replay["provenance"]["runtime_mode"] == "development"
        assert replay["provenance"]["model_id"] == model_id
        assert replay["provenance"]["track_id"] == track_id
        assert replay["provenance"]["seed"] == 412
        assert replay["result"]["status"] == status["status"]
        if status["status"] == "finished":
            assert replay["result"]["valid_sample"] is True
        else:
            assert replay["result"]["valid_sample"] is False
            assert replay["result"]["reason"]
        assert len(replay["frames"]) >= 2
        assert replay["frames"][0]["time_s"] == 0.0
        assert replay["frames"][-1]["tick"] == replay["result"]["steps"]
        assert replay["frames"][-1]["position"] != replay["frames"][0]["position"]
        if model_id == "B-seed1":
            assert status["steps"] == 85
            assert status["error"] == (
                "SuspensionResolutionError: track_boundary_exit: "
                "front_right wheel center left road support"
            )
            assert replay["result"]["valid_sample"] is False
    finally:
        manager.shutdown()


def test_fixed_seed_frozen_rollout_trace_is_repeatable():
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        traces = []
        for _ in range(2):
            session = manager.create(
                model_id="A-seed0", track_id="straight-sprint-v1", seed=8841,
            )
            session.start()
            assert session.wait(120.0)
            traces.append(session.replay()["result"]["trace_sha256"])
            manager.delete(session.session_id)
        assert traces[0] == traces[1]
    finally:
        manager.shutdown()


def test_session_manager_bounds_ready_sessions_and_rejects_repeated_start():
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        sessions = [
            manager.create(model_id="A-seed0", track_id="straight-sprint-v1", seed=seed)
            for seed in range(MAX_ACTIVE_SESSIONS)
        ]
        with pytest.raises(RuntimeFault, match="session limit"):
            manager.create(model_id="A-seed0", track_id="straight-sprint-v1", seed=10)
        assert all(session.status == "ready" for session in sessions)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(("name", "points", "width", "props", "checkpoint_count", "bridge_count"), [
    ("Straight", [(0, 0), (0, 100)], 12, [], 0, 0),
    ("U-shaped", [(0, 0), (0, 40), (15, 60), (40, 60), (55, 40), (55, 0)], 12, [], 0, 0),
    ("S-curve", [(0, 0), (15, 20), (-15, 40), (15, 60), (0, 90)], 12, [], 0, 0),
    ("Ramp", [(0, 0), (0, 140)], 12,
     [{"id": "ramp", "kind": "ramp", "t": 0.5, "offset": 0, "rotation": 0}], 0, 0),
    ("Elevated turn", [(0, 0), (0, 60), (40, 100), (80, 60), (80, 0)], 12,
     [{"id": "bridge", "kind": "bridge", "t": 0.5, "offset": 0, "rotation": 0}], 0, 0),
    ("Bridge crossing", [(-40, -40), (40, 40), (40, -40), (-40, 40), (-40, 90)], 8, [], 0, 1),
    ("Checkpoint route", [(0, 0), (0, 120)], 12,
     [{"id": f"checkpoint-{index}", "kind": "checkpoint", "t": position, "offset": 0, "rotation": 0}
      for index, position in enumerate((0.25, 0.5, 0.75))], 3, 0),
])
def test_auto_drive_finishes_authored_acceptance_tracks_with_python_physics(
    name, points, width, props, checkpoint_count, bridge_count,
):
    catalog = RuntimeCatalog(ROOT)
    document = {
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": f"auto-{name.lower().replace(' ', '-')}", "name": f"Auto Drive {name}",
        "width": width, "points": [{"x": x, "z": z} for x, z in points], "props": props,
    }
    converted = convert_custom_track(document, catalog)
    assert converted.auto_bridge_count == bridge_count
    assert len(converted.factory.track.checkpoints) == checkpoint_count
    if bridge_count:
        assert max(sample["y"] for sample in converted.road_samples) >= 5.0

    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None, track_id=converted.track_id, seed=412, mode="auto",
            custom_factory=converted.factory, custom_track_sha256=converted.track_sha256,
        )
        session.start()
        assert session.wait(240.0)
        status = session.status_document()
        replay = session.replay()

        assert status["status"] == "finished", status["error"]
        assert status["mode"] == "auto"
        assert replay["result"]["successful_finish"] is True
        assert replay["provenance"]["model_id"] == "auto-drive"
        assert replay["provenance"]["track_sha256"] == converted.track_sha256
        assert replay["frames"][-1]["route_progress_fraction"] == 1.0
        if checkpoint_count:
            assert replay["frames"][-1]["completed_checkpoint_ids"] == [
                f"checkpoint-{index:03d}" for index in range(checkpoint_count)
            ]
        if bridge_count:
            samples = converted.road_samples
            first, second = min(
                ((
                    ((left["x"] - right["x"]) ** 2 + (left["z"] - right["z"]) ** 2) ** 0.5,
                    left, right,
                ) for index, left in enumerate(samples) for right in samples[index + 1:]
                 if right["t"] - left["t"] > 0.2),
                key=lambda pair: pair[0],
            )[1:]
            assert ((first["x"] - second["x"]) ** 2 + (first["z"] - second["z"]) ** 2) ** 0.5 < 1.0
            lower, upper = sorted((first, second), key=lambda sample: sample["y"])
            assert upper["y"] - lower["y"] >= 5.0
            crossing_frames = [
                min(replay["frames"], key=lambda frame: abs(frame["route_progress_fraction"] - sample["t"]))
                for sample in (lower, upper)
            ]
            assert all(
                ((frame["position"][0] - sample["x"]) ** 2
                 + (frame["position"][2] - sample["z"]) ** 2) ** 0.5 < width / 2.0 + 1.0
                for frame, sample in zip(crossing_frames, (lower, upper), strict=True)
            )
            assert crossing_frames[1]["position"][1] - crossing_frames[0]["position"][1] >= 4.5
    finally:
        manager.shutdown()


def test_auto_drive_uses_the_latest_edited_custom_track():
    catalog = RuntimeCatalog(ROOT)
    original = {
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "editable-map", "name": "Editable Map",
        "width": 10, "points": [{"x": 0, "z": 0}, {"x": 0, "z": 100}], "props": [],
    }
    edited = {
        **original,
        "points": [{"x": 100, "z": 0}, {"x": 100, "z": 120}],
    }
    previous = convert_custom_track(original, catalog)
    current = convert_custom_track(edited, catalog)
    assert previous.track_sha256 != current.track_sha256

    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None, track_id=current.track_id, seed=412, mode="auto",
            custom_factory=current.factory, custom_track_sha256=current.track_sha256,
        )
        session.start()
        assert session.wait(240.0)
        replay = session.replay()

        assert replay["result"]["successful_finish"] is True
        assert replay["provenance"]["track_sha256"] == current.track_sha256
        assert replay["frames"][0]["position"][0] == pytest.approx(100.0)
    finally:
        manager.shutdown()

def test_auto_drive_status_exposes_actual_physics_source_not_a_claimed_v4_label():
    """Live clients must see the applied immutable Physics identity."""
    catalog = RuntimeCatalog(ROOT)
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None, track_id="straight-sprint-v1", seed=412, mode="auto"
        )
        status = session.status_document()
        assert status["physics"]["mode"] == session.track.physics_config.mode
        assert status["physics"]["ground_motion_version"] == (
            session.track.physics_config.ground_motion_version
        )
        assert status["physics"]["sha256"] == session.track.physics_config.sha256
        assert status["physics"]["ground_motion_version"] != "spatial_road_edge_partial_lifecycle_v4"
    finally:
        manager.shutdown()

def test_t4_reset_emits_versioned_new_epoch_before_tick_zero():
    """Real paced Gym session: restart status must mark the new frame epoch."""
    import time

    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        session = manager.create(
            model_id=None, track_id="straight-sprint-v1",
            seed=412, mode="manual",
        )
        session.start(paced=True)
        initial_statuses = []
        deadline = time.monotonic() + 12.0
        while session.status_document()["steps"] < 5 and time.monotonic() < deadline:
            event = session.next_event(timeout=0.3)
            if event is not None and event["type"] == "status":
                initial_statuses.append(event["data"])
        assert session.status_document()["steps"] >= 5
        assert {"status": "running", "reset": False} in initial_statuses

        session.reset()
        events_after_reset = []
        restart_seen = False
        fresh_frame_seen = False
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not fresh_frame_seen:
            event = session.next_event(timeout=0.25)
            if event is None:
                continue
            events_after_reset.append(event)
            if (event["type"] == "status" and
                    event["data"].get("status") == "running" and
                    event["data"].get("reset") is True):
                restart_seen = True
            if event["type"] == "frame" and event["data"]["tick"] == 0:
                assert restart_seen, "new tick zero arrived without reset epoch marker"
                fresh_frame_seen = True
        assert fresh_frame_seen, "physical session did not restart after reset"
        assert any(e["type"] == "status" and
                   e["data"].get("status") == "resetting" for e in events_after_reset)
        assert session.status_document()["status"] in {"running", "paused", "finished"}
    finally:
        manager.shutdown()
