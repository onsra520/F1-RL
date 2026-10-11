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


def _validate_schema_version(value: object, *, allow_navigation: bool = False) -> None:
    valid = (1, 2) if allow_navigation else (1,)
    if isinstance(value, bool) or not isinstance(value, int) or value not in valid:
        raise ValueError("schema_version must be an allowed integer")


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
class AuthoredNavigation:
    points: tuple[Vec3, ...]
    surface_ids: tuple[str, ...]
    checkpoint_point_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple) or len(self.points) < 2 or not all(isinstance(p, Vec3) for p in self.points):
            raise ValueError("navigation.points must be at least two Vec3 values")
        if any((b-a).magnitude <= 1e-8 for a,b in zip(self.points,self.points[1:])):
            raise ValueError("navigation has adjacent duplicate points")
        if not isinstance(self.surface_ids, tuple) or len(self.surface_ids) != len(self.points)-1 or not all(isinstance(s,str) and s.strip() for s in self.surface_ids):
            raise ValueError("navigation.surface_ids must contain exactly one surface id per segment")
        indices=self.checkpoint_point_indices
        if not isinstance(indices,tuple) or any(isinstance(i,bool) or not isinstance(i,int) for i in indices):
            raise ValueError("navigation.checkpoint_point_indices must be integers")
        if any(i<=0 or i>=len(self.points)-1 for i in indices) or tuple(sorted(set(indices))) != indices:
            raise ValueError("navigation.checkpoint_point_indices must be strictly increasing interior indices")


@dataclass(frozen=True, slots=True)
class TrackAsset:
    schema_version: int
    track_id: str
    spawn: SpawnPose
    world: TrackWorld
    world_sha256: str
    content_sha256: str
    navigation: AuthoredNavigation | None = None

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, allow_navigation=True)
        if self.schema_version == 1 and self.navigation is not None:
            raise ValueError("schema version 1 cannot have navigation")
        if self.schema_version == 2 and not isinstance(self.navigation, AuthoredNavigation):
            raise ValueError("schema version 2 requires authored navigation")
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
