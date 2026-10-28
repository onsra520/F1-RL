from dataclasses import FrozenInstanceError

import pytest

from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.track.model import TrackWorld
from src.simulation.track.patch import PlanarSurfacePatch
from src.tracks import SpawnPose, TrackAsset
from src.tracks.canonical import _build_track_asset


EMPTY_WORLD_HASH = "934b882bec775fa4fbf7c398296f7683e7fed50177ec572a7d6395652dbb8788"
EMPTY_CONTENT_HASH = "5aa8477c4fca084953f9719376d28675e81983f9d427b08c59d56769049a192c"


def _spawn(x=0.0, y=0.0, z=0.0, q=None):
    return SpawnPose(Vec3(float(x), float(y), float(z)), q or Quaternion.identity())


def _surface(surface_id: str, *, width=12.0):
    return PlanarSurfacePatch(
        surface_id,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        width,
        20.0,
    )


def test_empty_world_hash_regression_vector():
    asset = _build_track_asset(
        schema_version=1,
        track_id="empty-a",
        spawn=_spawn(),
        world=TrackWorld(),
    )
    assert asset.world_sha256 == EMPTY_WORLD_HASH
    assert asset.content_sha256 == EMPTY_CONTENT_HASH


def test_track_id_does_not_change_semantic_hashes():
    first = _build_track_asset(schema_version=1, track_id="one", spawn=_spawn(), world=TrackWorld())
    second = _build_track_asset(schema_version=1, track_id="two", spawn=_spawn(), world=TrackWorld())
    assert (first.world_sha256, first.content_sha256) == (second.world_sha256, second.content_sha256)


def test_spawn_changes_only_content_hash():
    first = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(), world=TrackWorld())
    second = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(z=5), world=TrackWorld())
    assert first.world_sha256 == second.world_sha256
    assert first.content_sha256 != second.content_sha256


def test_geometry_change_changes_both_hashes():
    first = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(), world=TrackWorld((_surface("a"),)))
    second = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(), world=TrackWorld((_surface("a", width=13.0),)))
    assert first.world_sha256 != second.world_sha256
    assert first.content_sha256 != second.content_sha256


def test_entity_tuple_input_order_is_non_semantic():
    first = TrackWorld((_surface("a"), _surface("b")))
    second = TrackWorld((_surface("b"), _surface("a")))
    a = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(), world=first)
    b = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(), world=second)
    assert a.world_sha256 == b.world_sha256
    assert a.content_sha256 == b.content_sha256


def test_continuous_int_and_float_spellings_are_semantically_equal():
    a = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(), world=TrackWorld((_surface("a", width=12),)))
    b = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(), world=TrackWorld((_surface("a", width=12.0),)))
    assert a.world_sha256 == b.world_sha256


def test_negative_zero_is_canonicalized():
    a = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(x=-0.0), world=TrackWorld())
    b = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(x=0.0), world=TrackWorld())
    assert a.content_sha256 == b.content_sha256


def test_equivalent_quaternion_signs_are_canonicalized_even_with_leading_zeroes():
    q = Quaternion(0.0, 0.0, 1.0, 0.0)
    neg = Quaternion(-0.0, -0.0, -1.0, -0.0)
    a = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(q=q), world=TrackWorld())
    b = _build_track_asset(schema_version=1, track_id="same", spawn=_spawn(q=neg), world=TrackWorld())
    assert a.content_sha256 == b.content_sha256


def test_dtos_are_frozen_and_validate_owned_fields():
    spawn = _spawn()
    asset = _build_track_asset(schema_version=1, track_id="valid", spawn=spawn, world=TrackWorld())
    with pytest.raises(FrozenInstanceError):
        spawn.position = Vec3(1.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        asset.track_id = "other"
    with pytest.raises(ValueError):
        TrackAsset(2, "valid", spawn, TrackWorld(), EMPTY_WORLD_HASH, EMPTY_CONTENT_HASH)
    with pytest.raises(ValueError):
        TrackAsset(1, " ", spawn, TrackWorld(), EMPTY_WORLD_HASH, EMPTY_CONTENT_HASH)
    with pytest.raises(ValueError):
        TrackAsset(1, "valid", spawn, TrackWorld(), "ABC", EMPTY_CONTENT_HASH)


def test_hash_domain_strings_are_pinned_by_regression_vectors():
    # The expected constants above were calculated from exact literal canonical preimages:
    # {"hash_schema":"f1rl-track-world-v1","world":{...empty categories...}}
    # {"hash_schema":"f1rl-track-content-v1","spawn":{...identity...},"world":{...empty categories...}}
    assert len(EMPTY_WORLD_HASH) == 64
    assert len(EMPTY_CONTENT_HASH) == 64
