"""Unscored Physics v4 contacts remain failures without runtime synthesis."""
from pathlib import Path

import pytest

from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
from f1rl.local_demo.runtime.development_collision_terminal import inspect_collision_error
from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.local_demo.runtime.session import RuntimeSessionManager
from f1rl.sandbox.ballistic_flight import FlightPhysicsError


def _make_track(catalog, kind="cone"):
    return convert_custom_track({
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "collision-terminal-test",
        "name": "Collision terminal", "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [{"id": kind, "kind": kind, "t": 0.5, "offset": 0, "rotation": 0}],
    }, catalog)


def _make_session(manager, track):
    return manager.create(
        model_id=None, track_id=track.track_id, seed=412, mode="auto",
        custom_factory=track.factory, custom_track_sha256=track.track_sha256,
        auto_drive_config=AutoDriveConfig(driving_style="centerline"),
    )


@pytest.mark.parametrize("kind", ["cone", "barrier"])
def test_collision_terminal_is_out_of_band_without_fabricated_step(kind):
    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    converted = _make_track(catalog, kind)
    manager = RuntimeSessionManager(catalog)
    try:
        session = _make_session(manager, converted)
        session.start()
        assert session.wait(180)
        replay = session.replay()
        frames = replay["frames"]
        assert "collision_terminal" not in replay
        if replay["result"]["status"] == "finished":
            assert kind == "barrier"
            assert replay["result"]["valid_sample"] is False
        else:
            assert replay["result"]["status"] == "failed"
            assert "GroundedBodyCollisionUnscored" in replay["result"]["reason"]
            assert replay["result"]["valid_sample"] is False
            assert session.status_document()["error"] == replay["result"]["reason"]
        assert replay["result"]["steps"] == frames[-1]["tick"]
        assert all(b["tick"] == a["tick"] + 1 for a, b in zip(frames, frames[1:]))
        events = []
        while (event := session.next_event(timeout=0.01)) is not None:
            events.append(event)
        collisions = [event for event in events if event["type"] == "collision"]
        recorded_ids = [frame["collision_ids"] for frame in frames if frame["collision_ids"]]
        assert [event["data"]["ids"] for event in collisions] == recorded_ids
        assert any(event["type"] == "terminal" and event["data"]["status"] == replay["result"]["status"]
                   for event in events)
    finally:
        manager.shutdown()


def test_failed_collision_reset_starts_a_fresh_episode():
    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    converted = _make_track(catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = _make_session(manager, converted)
        session.start()
        assert session.wait(180)
        first = session.replay()
        assert "GroundedBodyCollisionUnscored" in first["result"]["reason"]
        session.reset()
        assert session.wait(180)
        second = session.replay()
        assert "GroundedBodyCollisionUnscored" in second["result"]["reason"]
        assert second["frames"][0]["tick"] == 0
        assert second["frames"] == first["frames"]
        assert "collision_terminal" not in second
        assert second["result"]["trace_sha256"] == first["result"]["trace_sha256"]
    finally:
        manager.shutdown()


def test_registered_v4_contact_does_not_use_the_v3_runtime_terminal_adapter():
    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    converted = _make_track(catalog)
    environment = converted.factory.create()
    try:
        environment.reset(seed=412)
        snapshot = environment._simulation.snapshot
        obstacle_id = converted.factory.track.obstacles[0].id
        error = FlightPhysicsError("contact", failure_code="verified_obstacle_contact")
        error.witnessed_obstacle_ids = (obstacle_id,)
        evidence = inspect_collision_error(error, environment)
        assert evidence is None
        assert environment._simulation.snapshot is snapshot
        error.witnessed_obstacle_ids = ("invalid-id",)
        assert inspect_collision_error(error, environment) is None
        uncertain = FlightPhysicsError("uncertain", failure_code="uncertified_physics")
        uncertain.witnessed_obstacle_ids = (obstacle_id,)
        assert inspect_collision_error(uncertain, environment) is None
    finally:
        environment.close()
