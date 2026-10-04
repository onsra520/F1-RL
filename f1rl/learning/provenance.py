from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
import re

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_FILE_ROLES = (
    "experiment_definition",
    "training_runtime_config",
    "track_manifest",
)
_PACKAGE_NAMES = (
    "numpy",
    "gymnasium",
    "stable-baselines3",
    "torch",
    "PyYAML",
)
_EXPERIMENT_STATUSES = {"draft", "frozen"}
_TRAINING_STATUSES = {"pilot_owned", "frozen"}


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _relative_posix_path(value: object, name: str) -> str:
    raw = _nonblank(value, name)
    if "\\" in raw:
        raise ValueError(f"{name} must use POSIX '/' separators")
    if PureWindowsPath(raw).drive:
        raise ValueError(f"{name} must be repository-relative")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise ValueError(f"{name} must be repository-relative")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be normalized repository-relative POSIX text")
    return pure.as_posix()


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in _FILE_ROLES:
            raise ValueError(f"role must be one of {_FILE_ROLES!r}")
        object.__setattr__(self, "path", _relative_posix_path(self.path, "path"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))


@dataclass(frozen=True, slots=True)
class GitProvenance:
    commit_sha: str
    dirty: bool

    def __post_init__(self) -> None:
        if not isinstance(self.commit_sha, str) or _GIT_SHA_RE.fullmatch(self.commit_sha) is None:
            raise ValueError("commit_sha must be 40 lowercase hexadecimal characters")
        if type(self.dirty) is not bool:
            raise ValueError("dirty must be bool")


@dataclass(frozen=True, slots=True)
class PackageVersion:
    name: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonblank(self.name, "package name"))
        object.__setattr__(self, "version", _nonblank(self.version, "package version"))


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    python_version: str
    python_implementation: str
    system: str
    release: str
    machine: str
    packages: tuple[PackageVersion, ...]

    def __post_init__(self) -> None:
        for name in (
            "python_version",
            "python_implementation",
            "system",
            "release",
            "machine",
        ):
            object.__setattr__(self, name, _nonblank(getattr(self, name), name))
        if not isinstance(self.packages, tuple):
            raise ValueError("packages must be a tuple")
        if not all(isinstance(item, PackageVersion) for item in self.packages):
            raise ValueError("packages must contain PackageVersion values")
        names = tuple(item.name for item in self.packages)
        if names != _PACKAGE_NAMES:
            raise ValueError(f"packages must use exact v1 order {_PACKAGE_NAMES!r}")


@dataclass(frozen=True, slots=True)
class TrackProvenance:
    track_id: str
    world_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonblank(self.track_id, "track_id"))
        object.__setattr__(
            self,
            "world_sha256",
            _sha256(self.world_sha256, "world_sha256"),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )


@dataclass(frozen=True, slots=True)
class RunProvenance:
    schema_version: int
    experiment_id: str
    protocol_version: int
    experiment_status: str
    training_protocol_status: str
    git: GitProvenance
    files: tuple[FileFingerprint, ...]
    runtime: RuntimeProvenance
    tracks: tuple[TrackProvenance, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version != 1:
            raise ValueError("schema_version must be integer 1")
        object.__setattr__(
            self,
            "experiment_id",
            _nonblank(self.experiment_id, "experiment_id"),
        )
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version <= 0
        ):
            raise ValueError("protocol_version must be a positive integer")
        if self.experiment_status not in _EXPERIMENT_STATUSES:
            raise ValueError("experiment_status must be draft or frozen")
        if self.training_protocol_status not in _TRAINING_STATUSES:
            raise ValueError("training_protocol_status must be pilot_owned or frozen")
        if not isinstance(self.git, GitProvenance):
            raise ValueError("git must be GitProvenance")
        if not isinstance(self.files, tuple) or not all(
            isinstance(item, FileFingerprint) for item in self.files
        ):
            raise ValueError("files must be a tuple of FileFingerprint values")
        roles = tuple(item.role for item in self.files)
        if roles != _FILE_ROLES:
            raise ValueError(f"files must use exact v1 role order {_FILE_ROLES!r}")
        paths = tuple(item.path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("file fingerprint paths must be unique")
        if not isinstance(self.runtime, RuntimeProvenance):
            raise ValueError("runtime must be RuntimeProvenance")
        if not isinstance(self.tracks, tuple) or not self.tracks:
            raise ValueError("tracks must be a non-empty tuple")
        if not all(isinstance(item, TrackProvenance) for item in self.tracks):
            raise ValueError("tracks must contain TrackProvenance values")
        track_ids = tuple(item.track_id for item in self.tracks)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique")


def _provenance_document(provenance: RunProvenance) -> dict[str, object]:
    if not isinstance(provenance, RunProvenance):
        raise ValueError("provenance must be RunProvenance")
    return {
        "schema_version": provenance.schema_version,
        "experiment": {
            "id": provenance.experiment_id,
            "protocol_version": provenance.protocol_version,
            "status": provenance.experiment_status,
            "training_protocol_status": provenance.training_protocol_status,
        },
        "git": {
            "commit_sha": provenance.git.commit_sha,
            "dirty": provenance.git.dirty,
        },
        "files": [
            {
                "role": item.role,
                "path": item.path,
                "sha256": item.sha256,
            }
            for item in provenance.files
        ],
        "runtime": {
            "python_version": provenance.runtime.python_version,
            "python_implementation": provenance.runtime.python_implementation,
            "system": provenance.runtime.system,
            "release": provenance.runtime.release,
            "machine": provenance.runtime.machine,
            "packages": [
                {"name": item.name, "version": item.version}
                for item in provenance.runtime.packages
            ],
        },
        "tracks": [
            {
                "track_id": item.track_id,
                "world_sha256": item.world_sha256,
                "content_sha256": item.content_sha256,
            }
            for item in provenance.tracks
        ],
    }
