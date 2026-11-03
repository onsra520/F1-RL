"""Actual Python/Gym proof that navigation configuration changes the vehicle trajectory."""
from pathlib import Path

from src.runtime.catalog import RuntimeCatalog
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.session import RuntimeSessionManager

ROOT = Path(__file__).resolve().parents[2]


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
            assert replay["provenance"]["dataset_split"] == "DEVELOPMENT"
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

    from src.runtime.auto_drive import AutoDriveController
    from src.runtime.custom_tracks import convert_custom_track
    from src.simulation.geometry.vector import Vec3

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
