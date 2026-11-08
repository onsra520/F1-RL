from pathlib import Path
import time

import pytest

from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.directed_gate_guidance import (
    RIGHT_TURN_WORLD_SHA256,
    directed_gate_guidance,
)
from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.local_demo.runtime.session import RuntimeSessionManager
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.track_assets.model import AuthoredNavigation


def _navigation():
    points = tuple(Vec3(x, 0, z) for x, z in ((0, 4), (0, 35), (0, 64), (0, 70), (20, 70), (64, 70)))
    return AuthoredNavigation(points, ("road-entry", "road-entry", "road-entry", "road-exit", "road-exit"), (1, 4))


def _select(track, sha, navigation, config, position=Vec3(0, 0.5, 4), orientation=Quaternion(1, 0, 0, 0)):
    return directed_gate_guidance(track, sha, navigation, config, position, orientation)


def test_registered_right_turn_uses_private_guidance_and_preserves_requested_config():
    original = _navigation()
    requested = AutoDriveConfig()

    guidance, effective, selected = _select("right-turn-v1", RIGHT_TURN_WORLD_SHA256, original, requested)

    assert selected
    assert guidance != original
    assert effective.target_max_speed_kmh == 25
    assert requested.target_max_speed_kmh == 45
    assert guidance.checkpoint_point_indices == (1, 6)


def test_unregistered_worlds_or_user_lane_preferences_keep_authored_guidance():
    original = _navigation()
    for track, digest, navigation, config in (
        ("other", RIGHT_TURN_WORLD_SHA256, original, AutoDriveConfig()),
        ("right-turn-v1", "0" * 64, original, AutoDriveConfig()),
        ("right-turn-v1", RIGHT_TURN_WORLD_SHA256, original, AutoDriveConfig(driving_style="centerline")),
        ("right-turn-v1", RIGHT_TURN_WORLD_SHA256, original, AutoDriveConfig(lane_offset_m=2)),
    ):
        guidance, effective, selected = _select(track, digest, navigation, config)
        assert not selected
        assert guidance is navigation
        assert effective is config


def test_guidance_never_raises_a_lower_requested_speed():
    _, effective, selected = _select(
        "right-turn-v1", RIGHT_TURN_WORLD_SHA256, _navigation(),
        AutoDriveConfig(target_max_speed_kmh=15),
    )

    assert selected
    assert effective.target_max_speed_kmh == 15


def test_changed_spawn_does_not_use_registered_guidance():
    navigation = _navigation()
    for position, orientation in (
        (Vec3(1, 0.5, 4), Quaternion(1, 0, 0, 0)),
        (Vec3(0, 0.5, 4), Quaternion(0.70710678, 0, 0.70710678, 0)),
    ):
        _, _, selected = _select(
            "right-turn-v1", RIGHT_TURN_WORLD_SHA256, navigation, AutoDriveConfig(), position, orientation,
        )
        assert not selected


def test_unregistered_track_does_not_reuse_the_right_turn_guidance_profile():
    manager = RuntimeSessionManager(RuntimeCatalog(Path(__file__).resolve().parents[2]))
    try:
        session = manager.create(
            model_id=None, track_id="right-turn-train-v2", seed=412, mode="auto",
            auto_drive_config=AutoDriveConfig(target_max_speed_kmh=15),
        )
        status = session.status_document()
        assert status["auto_drive_config"]["target_max_speed_kmh"] == 15
        assert status["effective_max_speed_kmh"] == pytest.approx(15)
        assert "auto_drive_guidance" not in status
    finally:
        manager.shutdown()


def test_guided_session_reset_starts_a_new_physics_epoch():
    manager = RuntimeSessionManager(RuntimeCatalog(Path(__file__).resolve().parents[2]))
    try:
        session = manager.create(model_id=None, track_id="right-turn-train-v2", seed=412, mode="auto")
        session.start(paced=True)
        deadline = time.monotonic() + 15
        while session.status_document()["steps"] < 5 and time.monotonic() < deadline:
            session.next_event(timeout=0.25)
        assert session.status_document()["steps"] >= 5
        session.reset()
        restart_seen = False
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            event = session.next_event(timeout=0.25)
            if event and event["type"] == "status" and event["data"].get("reset") is True:
                restart_seen = True
            if event and event["type"] == "frame" and event["data"]["tick"] == 0 and restart_seen:
                break
        else:
            pytest.fail("guided session did not produce a new tick zero")
        assert "auto_drive_guidance" not in session.status_document()
    finally:
        manager.shutdown()
