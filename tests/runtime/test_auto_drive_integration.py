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
