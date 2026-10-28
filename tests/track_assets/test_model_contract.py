import pytest

from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.track.model import TrackWorld
from src.track_assets import SpawnPose, TrackAsset, TrackManifest


ZERO_HASH = "0" * 64


def _spawn() -> SpawnPose:
    return SpawnPose(Vec3(0.0, 0.0, 0.0), Quaternion.identity())


def test_track_asset_schema_version_requires_strict_integer_one():
    with pytest.raises(ValueError, match="schema_version"):
        TrackAsset(1.0, "track", _spawn(), TrackWorld(), ZERO_HASH, ZERO_HASH)


def test_track_manifest_schema_version_requires_strict_integer_one():
    with pytest.raises(ValueError, match="schema_version"):
        TrackManifest(1.0, ())
