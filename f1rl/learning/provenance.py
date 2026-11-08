from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import subprocess

from .experiments import ExperimentDefinition
from ..track_assets import load_track_manifest

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


def _fingerprint_file(
    *,
    repository_root: Path,
    source: Path,
    role: str,
) -> FileFingerprint:
    if not isinstance(repository_root, Path):
        raise ValueError("repository_root must be Path")
    if not repository_root.exists() or not repository_root.is_dir():
        raise ValueError("repository_root must be an existing directory")
    if not isinstance(source, Path):
        raise ValueError("source must be Path")
    if not source.exists() or not source.is_file():
        raise ValueError("source must be an existing file")

    root = repository_root.resolve()
    resolved_source = source.resolve()
    try:
        relative = resolved_source.relative_to(root)
    except ValueError as error:
        raise ValueError("source must resolve inside repository_root") from error

    digest = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    return FileFingerprint(role, relative.as_posix(), digest)


def _fingerprint_canonical_files(
    *,
    repository_root: Path,
    experiment_path: Path,
    training_runtime_config: Path,
    track_manifest: Path,
) -> tuple[FileFingerprint, ...]:
    return (
        _fingerprint_file(
            repository_root=repository_root,
            source=experiment_path,
            role="experiment_definition",
        ),
        _fingerprint_file(
            repository_root=repository_root,
            source=training_runtime_config,
            role="training_runtime_config",
        ),
        _fingerprint_file(
            repository_root=repository_root,
            source=track_manifest,
            role="track_manifest",
        ),
    )


def _run_git(repository_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Git command could not be executed") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git command failed: {detail or 'unknown error'}")
    return result.stdout


def _capture_git_provenance(repository_root: Path) -> GitProvenance:
    if not isinstance(repository_root, Path):
        raise ValueError("repository_root must be Path")
    if not repository_root.exists() or not repository_root.is_dir():
        raise ValueError("repository_root must be an existing directory")

    root = repository_root.resolve()
    commit_sha = _run_git(root, "rev-parse", "HEAD").strip()
    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    try:
        return GitProvenance(commit_sha, bool(status.strip()))
    except ValueError as error:
        raise RuntimeError("Git HEAD/status output is invalid") from error


def _capture_runtime_provenance(
    version_lookup: Callable[[str], str] = metadata.version,
) -> RuntimeProvenance:
    if not callable(version_lookup):
        raise ValueError("version_lookup must be callable")

    packages: list[PackageVersion] = []
    for name in _PACKAGE_NAMES:
        try:
            version = version_lookup(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required distribution version is unavailable: {name}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"failed to resolve required distribution version: {name}"
            ) from error
        try:
            packages.append(PackageVersion(name, version))
        except ValueError as error:
            raise RuntimeError(
                f"invalid required distribution version: {name}"
            ) from error

    return RuntimeProvenance(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        packages=tuple(packages),
    )


def _path_argument(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be str or Path")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be blank")
    return Path(value)


def _path_inside_repository(
    *,
    repository_root: Path,
    path: Path,
    name: str,
    require_file: bool = True,
) -> Path:
    if require_file and (not path.exists() or not path.is_file()):
        raise ValueError(f"{name} must identify an existing file")
    resolved = path.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"{name} must resolve inside repository_root") from error
    return resolved


def capture_run_provenance(
    *,
    repository_root: str | Path,
    experiment_path: str | Path,
    definition: ExperimentDefinition,
    selected_track_ids: tuple[str, ...],
) -> RunProvenance:
    root_input = _path_argument(repository_root, "repository_root")
    if not root_input.exists() or not root_input.is_dir():
        raise ValueError("repository_root must identify an existing directory")
    root = root_input.resolve()

    experiment_input = _path_argument(experiment_path, "experiment_path")
    experiment = _path_inside_repository(
        repository_root=root,
        path=experiment_input,
        name="experiment_path",
    )

    if not isinstance(definition, ExperimentDefinition):
        raise ValueError("definition must be ExperimentDefinition")
    if (
        not isinstance(selected_track_ids, tuple)
        or not selected_track_ids
        or not all(
            isinstance(track_id, str) and track_id.strip()
            for track_id in selected_track_ids
        )
        or len(selected_track_ids) != len(set(selected_track_ids))
    ):
        raise ValueError(
            "selected_track_ids must be a non-empty tuple of unique non-blank strings"
        )

    reloaded = ExperimentDefinition.from_yaml(experiment)
    if reloaded != definition:
        raise ValueError(
            "supplied definition does not match the experiment_path contents"
        )

    runtime_config = _path_inside_repository(
        repository_root=root,
        path=definition.training.runtime_config,
        name="training runtime config",
    )
    manifest_path = _path_inside_repository(
        repository_root=root,
        path=definition.tracks.manifest,
        name="track manifest",
    )
    files = _fingerprint_canonical_files(
        repository_root=root,
        experiment_path=experiment,
        training_runtime_config=runtime_config,
        track_manifest=manifest_path,
    )

    try:
        manifest = load_track_manifest(
            manifest_path, validate_asset_ids=selected_track_ids
        )
    except ValueError as error:
        if str(error) != "requested asset IDs are missing from manifest":
            raise
        declared = {entry.track_id for entry in load_track_manifest(
            manifest_path, validate_asset_ids=()
        ).tracks}
        for missing_id in selected_track_ids:
            if missing_id not in declared:
                raise ValueError(
                    f"selected track ID is absent from manifest: {missing_id}"
                ) from error
        raise
    by_id = {entry.track_id: entry for entry in manifest.tracks}
    tracks: list[TrackProvenance] = []
    for track_id in selected_track_ids:
        try:
            entry = by_id[track_id]
        except KeyError as error:
            raise ValueError(
                f"selected track ID is absent from manifest: {track_id}"
            ) from error
        tracks.append(
            TrackProvenance(
                track_id=entry.track_id,
                world_sha256=entry.world_sha256,
                content_sha256=entry.content_sha256,
            )
        )

    return RunProvenance(
        schema_version=1,
        experiment_id=definition.metadata.id,
        protocol_version=definition.metadata.protocol_version,
        experiment_status=definition.metadata.status,
        training_protocol_status=definition.training.protocol_status,
        git=_capture_git_provenance(root),
        files=files,
        runtime=_capture_runtime_provenance(),
        tracks=tuple(tracks),
    )


def write_run_provenance(
    path: str | Path,
    provenance: RunProvenance,
) -> None:
    destination = _path_argument(path, "path")
    if not isinstance(provenance, RunProvenance):
        raise ValueError("provenance must be RunProvenance")
    parent = destination.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError("provenance destination parent must be an existing directory")
    if destination.exists():
        if destination.is_dir():
            raise ValueError("provenance destination must be a file path")
        raise FileExistsError(f"provenance destination already exists: {destination}")

    payload = (
        json.dumps(
            _provenance_document(provenance),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    created = False
    try:
        with destination.open("xb") as handle:
            created = True
            handle.write(payload)
    except Exception:
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise
