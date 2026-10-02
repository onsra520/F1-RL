import json
import math
from pathlib import Path

from f1rl.learning.episode.route import RaceRoute
from f1rl.learning.experiments import ExperimentDefinition, resolve_track_split
from f1rl.track_assets import load_track_asset, load_track_manifest


TRACK_ROOT = Path("assets/tracks")
PROFILE = Path("configs/experiments/ppo-generalization-v1.yml")
EXPECTED = {
    "straight-sprint-v1": "training/straight-sprint-v1.json",
    "slalom-v1": "training/slalom-v1.json",
    "right-turn-v1": "training/right-turn-v1.json",
    "ramp-v1": "training/ramp-v1.json",
    "left-turn-v1": "evaluation/left-turn-v1.json",
    "offset-slalom-v1": "evaluation/offset-slalom-v1.json",
}
EXPECTED_SURFACE_COUNTS = {
    "straight-sprint-v1": 1,
    "slalom-v1": 1,
    "right-turn-v1": 2,
    "ramp-v1": 3,
    "left-turn-v1": 2,
    "offset-slalom-v1": 1,
}


def test_candidate_manifest_contains_exact_six_assets_without_split_semantics():
    raw = json.loads((TRACK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(raw) == {"schema_version", "tracks"}
    assert raw["schema_version"] == 1
    assert {entry["track_id"]: entry["path"] for entry in raw["tracks"]} == EXPECTED
    for entry in raw["tracks"]:
        assert set(entry) == {"track_id", "path", "world_sha256", "content_sha256"}
        assert not ({"split", "role", "training", "evaluation", "held_out"} & set(entry))


def test_all_candidate_assets_load_match_manifest_and_are_race_ready():
    manifest = load_track_manifest(TRACK_ROOT / "manifest.json")
    assert len(manifest.tracks) == 6
    world_hashes = set()
    content_hashes = set()

    for entry in manifest.tracks:
        asset = load_track_asset(TRACK_ROOT / entry.path)
        assert asset.track_id == Path(entry.path).stem == entry.track_id
        assert asset.world_sha256 == entry.world_sha256
        assert asset.content_sha256 == entry.content_sha256
        assert len(asset.world.surfaces) == EXPECTED_SURFACE_COUNTS[entry.track_id]
        assert len(asset.world.finish_lines) == 1
        RaceRoute.from_track(asset.world)
        world_hashes.add(asset.world_sha256)
        content_hashes.add(asset.content_sha256)

    assert len(world_hashes) == 6
    assert len(content_hashes) == 6


def test_canonical_experiment_resolves_approved_four_plus_two_split():
    definition = ExperimentDefinition.from_yaml(PROFILE)
    split = resolve_track_split(definition.tracks)

    assert tuple(entry.track_id for entry in split.training) == (
        "straight-sprint-v1",
        "slalom-v1",
        "right-turn-v1",
        "ramp-v1",
    )
    assert tuple(entry.track_id for entry in split.heldout) == (
        "left-turn-v1",
        "offset-slalom-v1",
    )
    selected = (*split.training, *split.heldout)
    assert len({entry.track_id for entry in selected}) == 6
    assert len({entry.world_sha256 for entry in selected}) == 6


def test_candidate_geometry_establishes_wp1b_scale_envelope():
    manifest = load_track_manifest(TRACK_ROOT / "manifest.json")
    widths = []
    total_surface_lengths = {}
    maximum_surface_origin_y = {}

    for entry in manifest.tracks:
        asset = load_track_asset(TRACK_ROOT / entry.path)
        assert all(math.isfinite(surface.width) and surface.width > 0 for surface in asset.world.surfaces)
        assert all(math.isfinite(surface.length) and surface.length > 0 for surface in asset.world.surfaces)
        widths.extend(surface.width for surface in asset.world.surfaces)
        total_surface_lengths[entry.track_id] = sum(surface.length for surface in asset.world.surfaces)
        maximum_surface_origin_y[entry.track_id] = max(surface.origin.y for surface in asset.world.surfaces)

    assert min(widths) == 12.0
    assert max(widths) == 16.0
    assert all(120.0 <= length <= 160.0 for length in total_surface_lengths.values())
    assert maximum_surface_origin_y["ramp-v1"] > 8.0
