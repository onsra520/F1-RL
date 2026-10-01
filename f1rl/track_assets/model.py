from __future__ import annotations

from dataclasses import dataclass
import re

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _validate_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")


def _validate_schema_version(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("schema_version must be integer 1")


@dataclass(frozen=True, slots=True)
class SpawnPose:
    position: Vec3
    orientation: Quaternion

    def __post_init__(self) -> None:
        if not isinstance(self.position, Vec3):
            raise ValueError("position must be Vec3")
        if not isinstance(self.orientation, Quaternion):
            raise ValueError("orientation must be Quaternion")


@dataclass(frozen=True, slots=True)
class TrackAsset:
    schema_version: int
    track_id: str
    spawn: SpawnPose
    world: TrackWorld
    world_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.spawn, SpawnPose):
            raise ValueError("spawn must be SpawnPose")
        if not isinstance(self.world, TrackWorld):
            raise ValueError("world must be TrackWorld")
        _validate_sha256(self.world_sha256, "world_sha256")
        _validate_sha256(self.content_sha256, "content_sha256")


@dataclass(frozen=True, slots=True)
class TrackManifestEntry:
    track_id: str
    path: str
    world_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("path must be a non-empty string")
        _validate_sha256(self.world_sha256, "world_sha256")
        _validate_sha256(self.content_sha256, "content_sha256")


@dataclass(frozen=True, slots=True)
class TrackManifest:
    schema_version: int
    tracks: tuple[TrackManifestEntry, ...]

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if not isinstance(self.tracks, tuple):
            raise ValueError("tracks must be a tuple")
        if not all(isinstance(entry, TrackManifestEntry) for entry in self.tracks):
            raise ValueError("tracks must contain TrackManifestEntry values")
