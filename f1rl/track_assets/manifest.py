from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .loader import (
    _array,
    _exact_keys,
    _mapping,
    _read_strict_json,
    _strict_int,
    _text,
    load_track_asset,
)
from .model import TrackManifest, TrackManifestEntry, _validate_sha256


def _portable_path(value: Any, name: str) -> str:
    raw = _text(value, name)
    if "\\" in raw:
        raise ValueError(f"{name} must use portable POSIX '/' separators")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or PureWindowsPath(raw).drive:
        raise ValueError(f"{name} must be a relative portable path")
    normalized = posixpath.normpath(raw)
    if normalized in ("", ".") or normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"{name} must remain inside the manifest directory")
    return normalized


def _entry(value: Any, index: int) -> TrackManifestEntry:
    name = f"tracks[{index}]"
    data = _mapping(value, name)
    _exact_keys(data, {"track_id", "path", "world_sha256", "content_sha256"}, name)
    track_id = _text(data["track_id"], f"{name}.track_id")
    path = _portable_path(data["path"], f"{name}.path")
    world_sha256 = data["world_sha256"]
    content_sha256 = data["content_sha256"]
    _validate_sha256(world_sha256, f"{name}.world_sha256")
    _validate_sha256(content_sha256, f"{name}.content_sha256")
    return TrackManifestEntry(track_id, path, world_sha256, content_sha256)


def load_track_manifest(path: str | Path) -> TrackManifest:
    source = Path(path)
    root_data = _mapping(_read_strict_json(source), "manifest")
    _exact_keys(root_data, {"schema_version", "tracks"}, "manifest")
    schema_version = _strict_int(root_data["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be integer 1")

    entries = tuple(_entry(value, index) for index, value in enumerate(_array(root_data["tracks"], "tracks")))
    track_ids = tuple(entry.track_id for entry in entries)
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("manifest track_id values must be unique")
    paths = tuple(entry.path for entry in entries)
    if len(paths) != len(set(paths)):
        raise ValueError("manifest path values must be unique after normalization")

    manifest_root = source.resolve().parent
    for entry in entries:
        target = (manifest_root / Path(entry.path)).resolve(strict=True)
        try:
            target.relative_to(manifest_root)
        except ValueError as exc:
            raise ValueError(f"manifest path resolves outside manifest directory: {entry.path}") from exc
        if not target.is_file():
            raise ValueError(f"manifest path must refer to a file: {entry.path}")
        asset = load_track_asset(target)
        if asset.track_id != entry.track_id:
            raise ValueError(f"manifest track_id mismatch for {entry.path}")
        if asset.world_sha256 != entry.world_sha256:
            raise ValueError(f"manifest world_sha256 mismatch for {entry.path}")
        if asset.content_sha256 != entry.content_sha256:
            raise ValueError(f"manifest content_sha256 mismatch for {entry.path}")

    return TrackManifest(schema_version, entries)
