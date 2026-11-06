import json
import math
from pathlib import Path

from f1rl.learning.episode.route import RaceRoute
from f1rl.learning.experiments import ExperimentDefinition, resolve_track_split
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.state import TrackState
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


def test_canonical_assets_pin_expected_track_mechanics_and_unchanged_hashes():
    manifest = load_track_manifest(TRACK_ROOT / "manifest.json")
    entries = {entry.track_id: entry for entry in manifest.tracks}
    assets = {
        track_id: load_track_asset(TRACK_ROOT / entry.path)
        for track_id, entry in entries.items()
    }

    slalom = assets["slalom-v1"].world
    assert len(slalom.boost_pads) == 1
    boost = slalom.boost_pads[0]
    assert boost.id == "boost-0"
    assert boost.volume.center == Vec3(0.0, 1.0, 36.0)
    assert boost.volume.half_extents == Vec3(7.0, 1.0, 4.0)
    assert boost.volume.orientation == Quaternion.identity()
    assert boost.acceleration_bonus == 3.0
    assert slalom.slow_zones == ()

    offset = assets["offset-slalom-v1"].world
    assert len(offset.boost_pads) == 1
    boost = offset.boost_pads[0]
    assert boost.id == "boost-0"
    assert boost.volume.center == Vec3(0.0, 1.0, 36.0)
    assert boost.volume.half_extents == Vec3(8.0, 1.0, 4.0)
    assert boost.volume.orientation == Quaternion.identity()
    assert boost.acceleration_bonus == 3.0
    assert offset.slow_zones == ()

    for track_id in ("right-turn-v1", "left-turn-v1"):
        world = assets[track_id].world
        assert world.boost_pads == ()
        assert len(world.slow_zones) == 1
        slow = world.slow_zones[0]
        assert slow.id == "slow-0"
        assert slow.volume.center == Vec3(0.0, 1.0, 50.0)
        assert slow.volume.half_extents == Vec3(6.0, 1.0, 8.0)
        assert slow.volume.orientation == Quaternion.identity()
        assert slow.drag_multiplier == 2.0

    for track_id in ("straight-sprint-v1", "ramp-v1"):
        world = assets[track_id].world
        assert world.boost_pads == ()
        assert world.slow_zones == ()

    assert entries["straight-sprint-v1"].world_sha256 == (
        "685f273f62bdc35399d73be5db922198022ad2b549623924dbeb8171d4a4088b"
    )
    assert entries["straight-sprint-v1"].content_sha256 == (
        "29aecc1482be48e0d4279b8067b47d5bd44ae0b376edc2de48354c01bf1b8bed"
    )
    assert entries["ramp-v1"].world_sha256 == (
        "775f795aa2763f64916e68e6a64fc763021e875c384050a974c747b6b4907978"
    )
    assert entries["ramp-v1"].content_sha256 == (
        "67a8280888e8010a2727fd113436bb237952693b8e29cebc2c3c1b8944eb7c70"
    )


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


def test_canonical_modifier_volumes_are_grounded_reachable_and_separated():
    manifest = load_track_manifest(TRACK_ROOT / "manifest.json")
    assets = {
        entry.track_id: load_track_asset(TRACK_ROOT / entry.path)
        for entry in manifest.tracks
    }
    state = TrackState(frozenset())

    for track_id in ("slalom-v1", "offset-slalom-v1"):
        world = assets[track_id].world
        boost = world.boost_pads[0]
        vehicle_point = Vec3(boost.volume.center.x, 0.5, boost.volume.center.z)
        query = world.query_point(vehicle_point, 0.75, state)
        assert query.contact is not None
        assert query.boost_pads == (boost,)
        boost_max_z = boost.volume.center.z + boost.volume.half_extents.z
        first_obstacle_min_z = min(
            obstacle.volume.center.z - obstacle.volume.half_extents.z
            for obstacle in world.obstacles
        )
        finish_min_z = min(
            finish.volume.center.z - finish.volume.half_extents.z
            for finish in world.finish_lines
        )
        assert boost_max_z < first_obstacle_min_z
        assert boost_max_z < finish_min_z
        RaceRoute.from_track(world)

    for track_id in ("right-turn-v1", "left-turn-v1"):
        world = assets[track_id].world
        slow = world.slow_zones[0]
        vehicle_point = Vec3(slow.volume.center.x, 0.5, slow.volume.center.z)
        query = world.query_point(vehicle_point, 0.75, state)
        assert query.contact is not None
        assert query.slow_zones == (slow,)
        slow_min_z = slow.volume.center.z - slow.volume.half_extents.z
        slow_max_z = slow.volume.center.z + slow.volume.half_extents.z
        first_checkpoint_max_z = (
            world.checkpoints[0].volume.center.z
            + world.checkpoints[0].volume.half_extents.z
        )
        second_checkpoint_min_z = (
            world.checkpoints[1].volume.center.z
            - world.checkpoints[1].volume.half_extents.z
        )
        entry = next(surface for surface in world.surfaces if surface.id == "road-entry")
        entry_end_z = entry.origin.z + entry.length / 2.0
        assert first_checkpoint_max_z < slow_min_z
        assert slow_max_z < second_checkpoint_min_z
        assert slow_max_z < entry_end_z
        assert all(
            not slow.volume.contains(finish.volume.center)
            for finish in world.finish_lines
        )
        RaceRoute.from_track(world)


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
