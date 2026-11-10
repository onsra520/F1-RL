import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
from f1rl.local_demo.runtime.session import RuntimeSessionManager

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/runtime/fixtures/m1-development-geometry-v1.json"


def _fixtures():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "f1rl_m1_development_geometry_v1"
    assert data["version"] == 1
    assert data["scope"] == "TRAIN_DEVELOPMENT_ONLY"
    return data["fixtures"]


@pytest.mark.resource_sensitive
@pytest.mark.xdist_group("physics_timeout")
@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda fixture: fixture["id"])
def test_versioned_m1_development_geometry_and_reference_finish(fixture):
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(fixture["track"], catalog)
    expected = set(fixture["features"])
    track = converted.factory.track
    heights = [sample["y"] for sample in converted.road_samples]
    vectors = [(second["x"] - first["x"], second["z"] - first["z"])
               for first, second in pairwise(fixture["track"]["points"])]
    turn_signs = [a[0] * b[1] - a[1] * b[0]
                  for a, b in pairwise(vectors) if abs(a[0] * b[1] - a[1] * b[0]) > 1e-6]

    assert converted.development_only is True
    assert len(track.checkpoints) == sum(
        prop["kind"] == "checkpoint" for prop in fixture["track"]["props"]
    )
    assert len(track.finish_lines) == 1
    assert len(track.obstacles) == sum(prop["kind"] in ("cone", "barrier")
                                       for prop in fixture["track"]["props"])
    assert track.surfaces
    assert all(surface.width == fixture["track"]["width"] for surface in track.surfaces)
    assert np.isfinite(np.asarray([[point.x, point.y, point.z]
                                   for point in converted.factory.navigation.points])).all()
    if "intersection" in expected:
        assert converted.requires_layered_road_support
        assert converted.required_physics_profile == "experimental_v5_layered"
        assert max(heights) - min(heights) >= 2.0
    if "ramp" in expected:
        assert max(heights) - min(heights) >= 2.5
        assert any(abs(surface.normal.y - 1.0) > 1e-4 for surface in track.surfaces)
    if "cone" in expected:
        assert any(prop["kind"] == "cone" for prop in fixture["track"]["props"])
    if "barrier" in expected:
        assert any(prop["kind"] == "barrier" for prop in fixture["track"]["props"])
    if "left_turn" in expected and "right_turn" in expected and "s_turn" in expected:
        assert any(turn > 0 for turn in turn_signs)
        assert any(turn < 0 for turn in turn_signs)
        assert any(a * b < 0 for a, b in pairwise(turn_signs))

    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None,
            track_id=converted.track_id,
            seed=412,
            mode="auto",
            custom_factory=converted.factory,
            custom_track_sha256=converted.track_sha256,
            physics_profile=(converted.required_physics_profile
                             or ("experimental_v5_layered" if "ramp" in expected else None)),
        )
        session.start()
        assert session.wait(240), f"reference controller timed out on {fixture['id']}"
        status = session.status_document()
        assert status["status"] == "finished", status.get("episode_result")
        replay = session.replay()
        assert replay["result"]["successful_finish"], replay["result"]
        assert len(replay["frames"][-1]["completed_checkpoint_ids"]) == len(track.checkpoints)
        assert all(frame["collision_count"] == 0 for frame in replay["frames"])
        assert all(not frame["collision_ids"] for frame in replay["frames"])
        assert replay["provenance"]["dataset_split"] == "DEVELOPMENT"
    finally:
        manager.shutdown()
