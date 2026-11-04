from pathlib import Path

import pytest

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import convert_custom_track
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.session import RuntimeSessionManager


@pytest.mark.parametrize("kind", ["cone", "barrier"])
def test_verified_contact_ends_with_collision_telemetry(kind):
    root = Path(__file__).resolve().parents[2]
    catalog = RuntimeCatalog(root)
    track = {
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "collision-terminal-test",
        "name": "Collision terminal", "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [{"id": kind, "kind": kind, "t": 0.5, "offset": 0, "rotation": 0}],
    }
    converted = convert_custom_track(track, catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(model_id=None, track_id=converted.track_id, seed=412, mode="auto",
                                 custom_factory=converted.factory,
                                 custom_track_sha256=converted.track_sha256,
                                 auto_drive_config=AutoDriveConfig(driving_style="centerline"))
        session.start()
        assert session.wait(180)
        replay = session.replay()
        assert replay["result"]["reason"] == "collision"
        assert replay["result"]["successful_finish"] is False
        assert replay["frames"][-1]["collision_ids"] == [converted.factory.track.obstacles[0].id]
        assert replay["frames"][-1]["collision_count"] == 1
    finally:
        manager.shutdown()
