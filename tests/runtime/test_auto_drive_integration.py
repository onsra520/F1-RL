"""Actual Python/Gym proof that navigation configuration changes the vehicle trajectory."""
from pathlib import Path

import pytest

from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.local_demo.runtime.session import RuntimeSessionManager

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["cone", "barrier"])
def test_auto_accelerates_brakes_and_recovers_speed_around_obstacle(kind):
    from f1rl.local_demo.runtime.custom_tracks import convert_custom_track

    catalog = RuntimeCatalog(ROOT)
    track = convert_custom_track({
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "speed-cycle", "name": "Speed cycle", "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 200}],
        "props": [{"id": kind, "kind": kind, "t": .6, "offset": 0, "rotation": 0}],
    }, catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(model_id=None, track_id=track.track_id, seed=412, mode="auto",
                                 custom_factory=track.factory, custom_track_sha256=track.track_sha256)
        session.start()
        assert session.wait(120)
        replay = session.replay()
        assert replay["result"]["successful_finish"], replay["result"]
        assert not replay["result"]["valid_sample"]
        assert "collision_terminal" not in replay
        frames = replay["frames"]
        before = [f for f in frames if 0 <= f["position"][2] < 70]
        approach = [f for f in frames if 70 <= f["position"][2] < 120]
        after = [f for f in frames if 140 <= f["position"][2] < 190]
        assert max(f["speed_mps"] for f in before) > 8
        assert min(f["speed_mps"] for f in approach) < 5
        assert any(f["action"]["brake"] > 0 for f in approach)
        assert any(current["action"]["brake"] > 0 and current["speed_mps"] < previous["speed_mps"]
                   for previous, current in zip(frames, frames[1:])
                   if 70 <= current["position"][2] < 120)
        assert max(f["speed_mps"] for f in after) > 8
        assert all(f["collision_count"] == 0 for f in frames)
    finally:
        manager.shutdown()


def test_right_hand_auto_finishes_around_centered_barrier_without_route_violation():
    from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
    from f1rl.sandbox.body_collision import obb_separation_lower_bound
    from f1rl.sandbox.geometry.rotation import Quaternion
    from f1rl.sandbox.geometry.vector import Vec3

    catalog = RuntimeCatalog(ROOT)
    draft = {
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "barrier-route", "name": "Barrier route", "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [{"id": "barrier", "kind": "barrier", "t": .25, "offset": 0, "rotation": 0}],
    }
    track = convert_custom_track(draft, catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(model_id=None, track_id=track.track_id, seed=412, mode="auto",
                                 custom_factory=track.factory, custom_track_sha256=track.track_sha256)
        session.start()
        assert session.wait(120)
        replay = session.replay()
        assert replay["result"]["successful_finish"], replay["result"]
        assert replay["result"]["valid_sample"] is False
        assert max(abs(frame["position"][0]) for frame in replay["frames"]) < 4.0
        assert all(frame["collision_count"] == 0 for frame in replay["frames"])
        assert "collision_terminal" not in replay
        body = track.factory.physics_config.body_geometry
        obstacle = track.factory.track.obstacles[0]
        assert all(obb_separation_lower_bound(
            body.box_at(Vec3(*frame["position"]), Quaternion(*frame["orientation_wxyz"])), obstacle.volume
        ) > 0 for frame in replay["frames"])
    finally:
        manager.shutdown()


def test_right_hand_auto_drive_finishes_slalom_without_collision():
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        session = manager.create(model_id=None, track_id="slalom-v1", seed=412, mode="auto")
        session.start()
        assert session.wait(120), "Slalom regression must terminate"
        replay = session.replay()
        assert replay["result"]["successful_finish"], replay["result"]
        assert len(replay["frames"][-1]["completed_checkpoint_ids"]) == 3
        assert all(frame["collision_count"] == 0 for frame in replay["frames"])
        assert any(frame["action"]["brake"] > 0 for frame in replay["frames"])
    finally:
        manager.shutdown()


@pytest.mark.parametrize(("profile", "speed", "seed"), [("normal", 45, 0), ("racing", 90, 1), ("hyper", 240, 412)])
def test_right_turn_guidance_is_not_reused_for_a_different_registered_track(profile, speed, seed):
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        config = AutoDriveConfig(driving_style="right", speed_profile=profile, target_max_speed_kmh=speed)
        session = manager.create(model_id=None, track_id="right-turn-train-v2", seed=seed, mode="auto", auto_drive_config=config)
        status = session.status_document()
        assert "auto_drive_guidance" not in status
        assert "guidance_max_speed_kmh" not in status
        assert status["auto_drive_config"]["target_max_speed_kmh"] == speed
    finally:
        manager.shutdown()


def test_right_hand_auto_drive_clears_a_two_cone_cluster():
    from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
    from f1rl.sandbox.body_collision import obb_separation_lower_bound
    from f1rl.sandbox.geometry.rotation import Quaternion
    from f1rl.sandbox.geometry.vector import Vec3

    document = {
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "sandbox-two-cones", "name": "Two cones",
        "width": 12, "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [
            {"id": "cone-a", "kind": "cone", "t": 0.5, "offset": -3, "rotation": 0},
            {"id": "cone-b", "kind": "cone", "t": 0.5, "offset": 1, "rotation": 0},
        ],
    }
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(document, catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(model_id=None, track_id=converted.track_id, seed=412, mode="auto",
                                 custom_factory=converted.factory, custom_track_sha256=converted.track_sha256)
        session.start()
        assert session.wait(120), "Two-cone regression must terminate"
        replay = session.replay()
        assert replay["result"]["successful_finish"], replay["result"]
        body = converted.factory.physics_config.body_geometry
        assert all(obb_separation_lower_bound(
            body.box_at(Vec3(*frame["position"]), Quaternion(*frame["orientation_wxyz"])), obstacle.volume
        ) > 0 for frame in replay["frames"] for obstacle in converted.factory.track.obstacles)
    finally:
        manager.shutdown()


def test_auto_drive_slows_for_a_cone_close_to_spawn():
    from f1rl.local_demo.runtime.custom_tracks import convert_custom_track

    document = {
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "near-cone", "name": "Near cone",
        "width": 12, "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [{"id": "cone-a", "kind": "cone", "t": 0.07, "offset": 0, "rotation": 0}],
    }
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(document, catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(model_id=None, track_id=converted.track_id, seed=412, mode="auto",
                                 custom_factory=converted.factory, custom_track_sha256=converted.track_sha256)
        session.start()
        assert session.wait(120), "Near-cone regression must terminate"
        replay = session.replay()
        assert replay["result"]["successful_finish"], replay["result"]
        assert all(frame["collision_count"] == 0 for frame in replay["frames"])
        assert max(frame["speed_mps"] for frame in replay["frames"]) > 3.0
    finally:
        manager.shutdown()


def test_right_vs_centerline_changes_real_physics_trajectory_and_replay_snapshot():
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        results = {}
        for style in ("right", "centerline"):
            config = AutoDriveConfig(driving_style=style, target_max_speed_kmh=40.0)
            session = manager.create(model_id=None, track_id="straight-sprint-v1", seed=412,
                                     mode="auto", auto_drive_config=config)
            session.start()
            assert session.wait(60.0)
            status, replay = session.status_document(), session.replay()
            assert status["auto_drive_config"]["driving_style"] == style
            assert replay["auto_drive_config"]["driving_style"] == style
            assert replay["provenance"]["dataset_split"] == "TRAIN"
            assert replay["provenance"]["model_id"] == "auto-drive"
            assert all(len(row["orientation_wxyz"]) == 4 for row in replay["frames"])
            assert len(replay["frames"]) > 25
            results[style] = replay["frames"][25]
        assert abs(results["right"]["position"][0] - results["centerline"]["position"][0]) > 0.02
        assert results["right"]["action"]["steering"] != results["centerline"]["action"]["steering"]
    finally:
        manager.shutdown()

def test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve():
    """Regression: at seed 412 the former auto offset produced -0.105 m at tick 86."""
    import json

    from f1rl.local_demo.runtime.auto_drive import AutoDriveController
    from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
    from f1rl.sandbox.geometry.vector import Vec3

    fixture = ROOT / "tests/runtime/fixtures/t05-original-obstacle-track.json"
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(json.loads(fixture.read_text(encoding="utf-8")), catalog)
    config = AutoDriveConfig(driving_style="right")
    manager = RuntimeSessionManager(catalog)
    env = converted.factory.create()
    try:
        session = manager.create(model_id=None, track_id=converted.track_id, seed=412, mode="auto",
                                 custom_factory=converted.factory,
                                 custom_track_sha256=converted.track_sha256, auto_drive_config=config)
        session.start()
        assert session.wait(120), "Original T05 regression must terminate"
        replay = session.replay()
        assert replay["result"]["successful_finish"], replay["result"]
        env.reset(seed=412)
        controller = AutoDriveController(
            env, converted.factory.navigation, converted.factory.vehicle_parameters,
            min(surface.width for surface in converted.factory.track.surfaces),
            config, obstacles=converted.factory.track.obstacles,
            body_geometry=converted.factory.physics_config.body_geometry,
        )
        lateral = []
        for frame in replay["frames"]:
            position = Vec3(*frame["position"])
            distance = controller.route.project(position, next_checkpoint_index=0).progress_meters
            origin = controller._point_at(distance)
            tx, tz = controller._tangent_at(distance)
            lateral.append((position.x - origin.x) * tz - (position.z - origin.z) * tx)
        assert min(lateral) >= -1e-4, (min(lateral), lateral.index(min(lateral)))
    finally:
        env.close()
        manager.shutdown()
