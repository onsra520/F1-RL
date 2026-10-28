import builtins
import json
from pathlib import Path

import pytest

from src.rl.provenance import (
    FileFingerprint,
    GitProvenance,
    PackageVersion,
    RunProvenance,
    RuntimeProvenance,
    TrackProvenance,
    _provenance_document,
    write_run_provenance,
)


def _provenance() -> RunProvenance:
    return RunProvenance(
        schema_version=1,
        experiment_id="ppo-generalization-v1",
        protocol_version=1,
        experiment_status="draft",
        training_protocol_status="pilot_owned",
        git=GitProvenance("1" * 40, False),
        files=(
            FileFingerprint(
                "experiment_definition",
                "configs/experiments/ppo-generalization-v1.yml",
                "a" * 64,
            ),
            FileFingerprint(
                "training_runtime_config",
                "config.yml",
                "b" * 64,
            ),
            FileFingerprint(
                "track_manifest",
                "assets/tracks/manifest.json",
                "c" * 64,
            ),
        ),
        runtime=RuntimeProvenance(
            "3.12.10",
            "CPython",
            "Windows",
            "11",
            "AMD64",
            (
                PackageVersion("numpy", "2.1.0"),
                PackageVersion("gymnasium", "1.3.0"),
                PackageVersion("stable-baselines3", "2.9.0"),
                PackageVersion("torch", "2.8.0"),
                PackageVersion("PyYAML", "6.0.2"),
            ),
        ),
        tracks=(
            TrackProvenance(
                "straight-sprint-v1",
                "d" * 64,
                "e" * 64,
            ),
        ),
    )


def test_writer_emits_exact_deterministic_utf8_json_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "provenance.json"
    provenance = _provenance()

    write_run_provenance(destination, provenance)

    expected = (
        json.dumps(
            _provenance_document(provenance),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert destination.read_bytes() == expected
    assert json.loads(destination.read_text(encoding="utf-8")) == _provenance_document(
        provenance
    )


def test_same_snapshot_serializes_byte_identically_to_different_paths(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    provenance = _provenance()

    write_run_provenance(first, provenance)
    write_run_provenance(second, provenance)

    assert first.read_bytes() == second.read_bytes()


def test_writer_rejects_missing_parent_and_directory_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent"):
        write_run_provenance(tmp_path / "missing" / "provenance.json", _provenance())

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="destination"):
        write_run_provenance(directory, _provenance())


def test_writer_requires_run_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance"):
        write_run_provenance(
            tmp_path / "provenance.json",
            object(),  # type: ignore[arg-type]
        )


def test_writer_never_overwrites_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "provenance.json"
    sentinel = b"keep-me"
    destination.write_bytes(sentinel)

    with pytest.raises(FileExistsError):
        write_run_provenance(destination, _provenance())

    assert destination.read_bytes() == sentinel


def test_writer_output_contains_no_absolute_project_path(tmp_path: Path) -> None:
    destination = tmp_path / "provenance.json"

    write_run_provenance(destination, _provenance())

    text = destination.read_text(encoding="utf-8")
    assert "D:\\" not in text
    assert "/home/" not in text


def test_failed_write_removes_partial_new_file_and_reraises_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "provenance.json"
    error = OSError("write failed")
    original_open = Path.open

    class FailingWriter:
        def __init__(self, raw: object) -> None:
            self.raw = raw

        def __enter__(self) -> "FailingWriter":
            return self

        def __exit__(self, *args: object) -> None:
            close = getattr(self.raw, "close")
            close()

        def write(self, data: bytes) -> int:
            raw_write = getattr(self.raw, "write")
            raw_write(data[:10])
            raise error

    def failing_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path == destination and mode == "xb":
            raw = builtins.open(path, "xb")
            return FailingWriter(raw)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError) as caught:
        write_run_provenance(destination, _provenance())

    assert caught.value is error
    assert not destination.exists()
