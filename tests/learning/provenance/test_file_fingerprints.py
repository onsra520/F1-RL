import hashlib
from pathlib import Path

import pytest

from src.learning.provenance import (
    _fingerprint_canonical_files,
    _fingerprint_file,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


def test_fingerprint_uses_exact_binary_bytes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    source = root / "config.yml"
    payload = b"alpha:\r\n  beta: 1\r\n"
    source.write_bytes(payload)

    fingerprint = _fingerprint_file(
        repository_root=root,
        source=source,
        role="training_runtime_config",
    )

    assert fingerprint.sha256 == hashlib.sha256(payload).hexdigest()
    assert fingerprint.path == "config.yml"


def test_whitespace_and_newline_bytes_change_digest(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    first = root / "first.yml"
    second = root / "second.yml"
    third = root / "third.yml"
    first.write_bytes(b"a: 1\n")
    second.write_bytes(b"a: 1 \n")
    third.write_bytes(b"a: 1\r\n")

    hashes = {
        _fingerprint_file(
            repository_root=root,
            source=source,
            role="experiment_definition",
        ).sha256
        for source in (first, second, third)
    }

    assert len(hashes) == 3


def test_empty_file_has_standard_sha256(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    source = root / "empty.yml"
    source.write_bytes(b"")

    fingerprint = _fingerprint_file(
        repository_root=root,
        source=source,
        role="experiment_definition",
    )

    assert fingerprint.sha256 == hashlib.sha256(b"").hexdigest()


def test_same_bytes_reproduce_same_digest(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    one = root / "one.yml"
    two = root / "two.yml"
    one.write_bytes(b"same")
    two.write_bytes(b"same")

    assert _fingerprint_file(
        repository_root=root,
        source=one,
        role="experiment_definition",
    ).sha256 == _fingerprint_file(
        repository_root=root,
        source=two,
        role="experiment_definition",
    ).sha256


def test_stored_path_is_repo_relative_posix(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    source = root / "configs" / "experiments" / "experiment.yml"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x")

    fingerprint = _fingerprint_file(
        repository_root=root,
        source=source,
        role="experiment_definition",
    )

    assert fingerprint.path == "configs/experiments/experiment.yml"
    assert str(root) not in fingerprint.path
    assert "\\" not in fingerprint.path


@pytest.mark.parametrize(
    "bad_root_kind",
    ["missing", "file"],
)
def test_repository_root_must_be_existing_directory(
    tmp_path: Path,
    bad_root_kind: str,
) -> None:
    root = tmp_path / "bad"
    if bad_root_kind == "file":
        root.write_text("not-dir", encoding="utf-8")
    source = tmp_path / "source.yml"
    source.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="repository_root"):
        _fingerprint_file(
            repository_root=root,
            source=source,
            role="experiment_definition",
        )


def test_source_must_be_existing_file(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="source"):
        _fingerprint_file(
            repository_root=root,
            source=root / "missing.yml",
            role="experiment_definition",
        )

    directory = root / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="source"):
        _fingerprint_file(
            repository_root=root,
            source=directory,
            role="experiment_definition",
        )


def test_source_outside_repository_is_rejected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    outside = tmp_path / "outside.yml"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="inside repository_root"):
        _fingerprint_file(
            repository_root=root,
            source=outside,
            role="experiment_definition",
        )


def test_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    outside = tmp_path / "outside.yml"
    outside.write_text("x", encoding="utf-8")
    link = root / "link.yml"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="inside repository_root"):
        _fingerprint_file(
            repository_root=root,
            source=link,
            role="experiment_definition",
        )


def test_canonical_file_helper_pins_exact_role_order(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    experiment = root / "configs" / "experiments" / "experiment.yml"
    runtime = root / "config.yml"
    manifest = root / "assets" / "tracks" / "manifest.json"
    experiment.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    experiment.write_text("experiment", encoding="utf-8")
    runtime.write_text("runtime", encoding="utf-8")
    manifest.write_text("manifest", encoding="utf-8")

    fingerprints = _fingerprint_canonical_files(
        repository_root=root,
        experiment_path=experiment,
        training_runtime_config=runtime,
        track_manifest=manifest,
    )

    assert tuple(item.role for item in fingerprints) == (
        "experiment_definition",
        "training_runtime_config",
        "track_manifest",
    )
    assert tuple(item.path for item in fingerprints) == (
        "configs/experiments/experiment.yml",
        "config.yml",
        "assets/tracks/manifest.json",
    )
