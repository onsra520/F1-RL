import json
import os
from pathlib import Path

import pytest

from f1rl.track_assets import load_track_asset, load_track_manifest


def _track(track_id: str = "track-a") -> dict:
    return {
        "schema_version": 1,
        "track_id": track_id,
        "spawn": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]},
        "surfaces": [],
        "obstacles": [],
        "boost_pads": [],
        "slow_zones": [],
        "checkpoints": [],
        "finish_lines": [],
        "score_pickups": [],
    }


def _write_track(root: Path, rel: str = "training/track-a.json", track_id: str = "track-a"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_track(track_id)), encoding="utf-8")
    return path


def _entry(root: Path, rel: str = "training/track-a.json", track_id: str = "track-a"):
    asset = load_track_asset(root / rel)
    return {
        "track_id": track_id,
        "path": rel,
        "world_sha256": asset.world_sha256,
        "content_sha256": asset.content_sha256,
    }


def _write_manifest(root: Path, entries, name="manifest.json"):
    path = root / name
    path.write_text(json.dumps({"schema_version": 1, "tracks": entries}), encoding="utf-8")
    return path


def _valid_manifest(tmp_path: Path):
    _write_track(tmp_path)
    return _write_manifest(tmp_path, [_entry(tmp_path)])


def test_valid_manifest_resolves_relative_to_manifest_not_cwd(tmp_path, monkeypatch):
    manifest = _valid_manifest(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    loaded = load_track_manifest(manifest)
    assert loaded.schema_version == 1
    assert loaded.tracks[0].track_id == "track-a"
    assert loaded.tracks[0].path == "training/track-a.json"


def test_manifest_loading_does_not_modify_manifest_or_track_bytes(tmp_path):
    manifest = _valid_manifest(tmp_path)
    track = tmp_path / "training/track-a.json"
    before_manifest = manifest.read_bytes()
    before_track = track.read_bytes()
    load_track_manifest(manifest)
    assert manifest.read_bytes() == before_manifest
    assert track.read_bytes() == before_track


@pytest.mark.parametrize("schema", [True, 0, 2, "1"])
def test_manifest_schema_is_strict_integer_one(tmp_path, schema):
    _write_track(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": schema, "tracks": [_entry(tmp_path)]}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_track_manifest(manifest)


def test_manifest_unknown_missing_and_duplicate_keys_are_rejected(tmp_path):
    _write_track(tmp_path)
    entry = _entry(tmp_path)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"schema_version": 1, "tracks": [entry], "extra": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        load_track_manifest(unknown)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="tracks"):
        load_track_manifest(missing)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"tracks":[{"track_id":"a","track_id":"b","path":"x","world_sha256":"' + "0"*64 + '","content_sha256":"' + "0"*64 + '"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*track_id"):
        load_track_manifest(duplicate)


def test_duplicate_track_ids_and_normalized_paths_are_rejected(tmp_path):
    _write_track(tmp_path, "training/a.json", "a")
    _write_track(tmp_path, "training/b.json", "b")
    a = _entry(tmp_path, "training/a.json", "a")
    b = _entry(tmp_path, "training/b.json", "b")

    duplicate_id = dict(b, track_id="a")
    with pytest.raises(ValueError, match="track_id.*unique|unique.*track_id"):
        load_track_manifest(_write_manifest(tmp_path, [a, duplicate_id], "dup-id.json"))

    duplicate_path = dict(b, path="training/x/../a.json")
    with pytest.raises(ValueError, match="path.*unique|unique.*path"):
        load_track_manifest(_write_manifest(tmp_path, [a, duplicate_path], "dup-path.json"))


@pytest.mark.parametrize("bad_hash", ["A"*64, "0"*63, "g"*64, 7, None])
def test_manifest_hashes_are_lowercase_64_hex(tmp_path, bad_hash):
    _write_track(tmp_path)
    entry = _entry(tmp_path)
    entry["world_sha256"] = bad_hash
    with pytest.raises(ValueError, match="world_sha256"):
        load_track_manifest(_write_manifest(tmp_path, [entry]))


@pytest.mark.parametrize(
    "bad_path",
    ["", "training\\track-a.json", "/tmp/track.json", "C:/track.json", "C:\\track.json", "../../escape.json"],
)
def test_nonportable_or_escaping_paths_are_rejected(tmp_path, bad_path):
    _write_track(tmp_path)
    entry = _entry(tmp_path)
    entry["path"] = bad_path
    with pytest.raises(ValueError, match="path"):
        load_track_manifest(_write_manifest(tmp_path, [entry]))


def test_manifest_target_must_be_file(tmp_path):
    directory = tmp_path / "training/track-a.json"
    directory.mkdir(parents=True)
    entry = {
        "track_id": "track-a",
        "path": "training/track-a.json",
        "world_sha256": "0"*64,
        "content_sha256": "0"*64,
    }
    with pytest.raises(ValueError, match="file"):
        load_track_manifest(_write_manifest(tmp_path, [entry]))


def test_symlink_escape_is_rejected_when_supported(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    outside_track = _write_track(outside, "track.json", "track-a")
    link = tmp_path / "escape.json"
    try:
        os.symlink(outside_track, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    asset = load_track_asset(outside_track)
    entry = {
        "track_id": "track-a",
        "path": "escape.json",
        "world_sha256": asset.world_sha256,
        "content_sha256": asset.content_sha256,
    }
    with pytest.raises(ValueError, match="path|outside"):
        load_track_manifest(_write_manifest(tmp_path, [entry]))


def test_asset_id_mismatch_is_rejected(tmp_path):
    _write_track(tmp_path, track_id="actual")
    asset = load_track_asset(tmp_path / "training/track-a.json")
    entry = {
        "track_id": "declared",
        "path": "training/track-a.json",
        "world_sha256": asset.world_sha256,
        "content_sha256": asset.content_sha256,
    }
    with pytest.raises(ValueError, match="track_id"):
        load_track_manifest(_write_manifest(tmp_path, [entry]))


def test_world_and_content_hash_tampering_are_detected_independently(tmp_path):
    _write_track(tmp_path)
    valid = _entry(tmp_path)
    bad_world = dict(valid, world_sha256="0"*64)
    with pytest.raises(ValueError, match="world_sha256"):
        load_track_manifest(_write_manifest(tmp_path, [bad_world], "bad-world.json"))

    bad_content = dict(valid, content_sha256="0"*64)
    with pytest.raises(ValueError, match="content_sha256"):
        load_track_manifest(_write_manifest(tmp_path, [bad_content], "bad-content.json"))
