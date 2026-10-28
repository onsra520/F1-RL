"""Reconstructed C2.1: authored navigation is versioned track content, not world geometry."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.track_assets import load_track_asset, load_track_manifest

ROOT = Path("assets/tracks")
MANIFEST = ROOT / "manifest.json"
WORLD_HASHES = {
    "straight-sprint-v1": "685f273f62bdc35399d73be5db922198022ad2b549623924dbeb8171d4a4088b",
    "slalom-v1": "6071829f2eb790ada04c5e59b22efb205e3a20d59fbe670e68fb78ed3eb304e0",
    "right-turn-v1": "d79133bc9a0599f9b056c7eb78658446631f564f038bc4afa91f3325fd4f9fd2",
    "ramp-v1": "775f795aa2763f64916e68e6a64fc763021e875c384050a974c747b6b4907978",
    "left-turn-v1": "4a613e8c851a9940395f81c54857c9b94f903cb893e29401b0457bebe8a10adf",
    "offset-slalom-v1": "32d903e81880ea290c840fd210ca2c43579004135acf36d58c49bd73d6c91346",
}


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "track.json"
    path.write_text(json.dumps(data, allow_nan=True), encoding="utf-8")
    return path


def _sample() -> dict:
    return json.loads((ROOT / "training" / "straight-sprint-v1.json").read_text(encoding="utf-8"))


def test_all_canonical_assets_are_navigation_schema_2_and_preserve_world_hashes() -> None:
    manifest = load_track_manifest(MANIFEST)
    assert len(manifest.tracks) == 6
    assert {e.track_id: e.world_sha256 for e in manifest.tracks} == WORLD_HASHES
    for entry in manifest.tracks:
        asset = load_track_asset(ROOT / entry.path)
        assert asset.schema_version == 2
        assert asset.navigation is not None
        assert len(asset.navigation.points) >= 2
        assert len(asset.navigation.surface_ids) == len(asset.navigation.points) - 1
        assert len(asset.navigation.checkpoint_point_indices) == len(asset.world.checkpoints)
        assert asset.world_sha256 == WORLD_HASHES[entry.track_id]
        assert asset.content_sha256 == entry.content_sha256
        assert asset.navigation.points[0] == asset.spawn.position
        assert asset.navigation.points[-1] == asset.world.finish_lines[0].volume.center


def test_legacy_schema_1_remains_loadable_without_navigation(tmp_path: Path) -> None:
    data = _sample()
    data["schema_version"] = 1
    data.pop("navigation", None)
    legacy = load_track_asset(_write(tmp_path, data))
    assert legacy.schema_version == 1
    assert legacy.navigation is None
    assert legacy.world_sha256 == WORLD_HASHES["straight-sprint-v1"]


@pytest.mark.parametrize("mutation", [
    "missing_navigation", "unknown_navigation_key", "adjacent_equal_points",
    "wrong_segment_count", "unknown_surface", "checkpoint_wrong_order",
    "off_surface_point", "off_checkpoint_point", "bad_start", "bad_finish",
    "nonfinite_waypoint",
])
def test_v2_rejects_invalid_navigation_contract(tmp_path: Path, mutation: str) -> None:
    data = _sample()
    nav = data["navigation"]
    if mutation == "missing_navigation":
        del data["navigation"]
    elif mutation == "unknown_navigation_key":
        nav["silent_extra"] = True
    elif mutation == "adjacent_equal_points":
        nav["points"][1] = nav["points"][0]
    elif mutation == "wrong_segment_count":
        nav["surface_ids"].pop()
    elif mutation == "unknown_surface":
        nav["surface_ids"][0] = "not-a-road"
    elif mutation == "checkpoint_wrong_order":
        nav["checkpoint_point_indices"] = list(reversed(nav["checkpoint_point_indices"]))
    elif mutation == "off_surface_point":
        nav["points"][1] = [999, 0, 35]
    elif mutation == "off_checkpoint_point":
        nav["points"][nav["checkpoint_point_indices"][0]] = [0, 0, 50]
    elif mutation == "bad_start":
        nav["points"][0] = [1, 0, 4]
    elif mutation == "bad_finish":
        nav["points"][-1] = [0, 0, 111]
    elif mutation == "nonfinite_waypoint":
        nav["points"][1] = [float("nan"), 0, 35]
    with pytest.raises(ValueError):
        load_track_asset(_write(tmp_path, data))


def test_navigation_content_hash_changes_without_changing_world_identity(tmp_path: Path) -> None:
    data = _sample()
    original = load_track_asset(_write(tmp_path, data))
    altered = copy.deepcopy(data)
    altered["navigation"]["points"].insert(1, [0, 0, 15])
    altered["navigation"]["surface_ids"].insert(0, "road-main")
    altered["navigation"]["checkpoint_point_indices"] = [
        i + 1 for i in altered["navigation"]["checkpoint_point_indices"]
    ]
    changed = load_track_asset(_write(tmp_path, altered))
    assert original.world_sha256 == changed.world_sha256
    assert original.content_sha256 != changed.content_sha256
