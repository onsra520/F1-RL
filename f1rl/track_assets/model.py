from __future__ import annotations

from dataclasses import dataclass
import re

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must be integer 1")
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.spawn, SpawnPose):
            raise ValueError("spawn must be SpawnPose")
        if not isinstance(self.world, TrackWorld):
            raise ValueError("world must be TrackWorld")
        if not isinstance(self.world_sha256, str) or _SHA256_RE.fullmatch(self.world_sha256) is None:
            raise ValueError("world_sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(self.content_sha256, str) or _SHA256_RE.fullmatch(self.content_sha256) is None:
            raise ValueError("content_sha256 must be 64 lowercase hexadecimal characters")
