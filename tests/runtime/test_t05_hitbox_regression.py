from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import convert_custom_track
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.session import RuntimeSessionManager
from src.simulation.body_collision import obb_separation_lower_bound
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = Path(__file__).parent / "fixtures" / "t05-original-obstacle-track.json"
ORIGINAL_HASH = "3359bae959535baa40d337ae7677540ab64f022e739acd148fa2f9b65369ec37"


@pytest.mark.parametrize("preset,target", [("normal", 45.0), ("racing", 90.0)])
def test_t05_original_cone_hitbox_racing_line_finishes_without_contact(preset, target):
    """Reproduces user-reported step-263 failure; hitbox and obstacle unchanged."""
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(json.loads(ORIGINAL.read_text(encoding="utf-8")), catalog)
    assert converted.track_sha256 == ORIGINAL_HASH
    assert len(converted.factory.track.obstacles) == 1
    body = converted.factory.physics_config.body_geometry
    assert body is not None
    assert body.width_m == 1.8 and body.length_m == 4.0
    cone = converted.factory.track.obstacles[0]
    assert cone.volume.half_extents == Vec3(0.65, 0.6, 0.65)

    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None, track_id=converted.track_id, seed=412, mode="auto",
            custom_factory=converted.factory, custom_track_sha256=converted.track_sha256,
            auto_drive_config=AutoDriveConfig(
                driving_style="racing", speed_profile=preset, target_max_speed_kmh=target,
            ),
        )
        session.start()
        assert session.wait(120.0), "T05 must terminate rather than hang"
        result = session.replay()
        assert session.status_document()["status"] == "finished"
        assert result["result"]["successful_finish"], result["result"]
        assert len(result["frames"]) > 263  # The original run failed at step 263.
        separations = [
            obb_separation_lower_bound(
                body.box_at(Vec3(*frame["position"]), Quaternion(*frame["orientation_wxyz"])),
                cone.volume,
            )
            for frame in result["frames"]
        ]
        # SAT projection gap: positive means hitboxes are provably separated.
        assert min(separations) > 0.2
    finally:
        manager.shutdown()
