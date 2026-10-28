from dataclasses import FrozenInstanceError

import pytest

from src.rl.provenance import (
    FileFingerprint,
    GitProvenance,
    PackageVersion,
    RunProvenance,
    RuntimeProvenance,
    TrackProvenance,
)


def _runtime() -> RuntimeProvenance:
    return RuntimeProvenance(
        python_version="3.12.10",
        python_implementation="CPython",
        system="Windows",
        release="11",
        machine="AMD64",
        packages=(
            PackageVersion("numpy", "2.1.0"),
            PackageVersion("gymnasium", "1.3.0"),
            PackageVersion("stable-baselines3", "2.9.0"),
            PackageVersion("torch", "2.8.0"),
            PackageVersion("PyYAML", "6.0.2"),
        ),
    )


def _files() -> tuple[FileFingerprint, ...]:
    return (
        FileFingerprint(
            "experiment_definition",
            "configs/experiments/ppo-generalization-v1.yml",
            "a" * 64,
        ),
        FileFingerprint("training_runtime_config", "config.yml", "b" * 64),
        FileFingerprint("track_manifest", "assets/tracks/manifest.json", "c" * 64),
    )


def _run(**overrides: object) -> RunProvenance:
    values: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "ppo-generalization-v1",
        "protocol_version": 1,
        "experiment_status": "draft",
        "training_protocol_status": "pilot_owned",
        "git": GitProvenance("d" * 40, False),
        "files": _files(),
        "runtime": _runtime(),
        "tracks": (
            TrackProvenance(
                "straight-sprint-v1",
                "e" * 64,
                "f" * 64,
            ),
        ),
    }
    values.update(overrides)
    return RunProvenance(**values)  # type: ignore[arg-type]


def test_provenance_contracts_are_frozen_and_slotted() -> None:
    value = _run()
    assert hasattr(RunProvenance, "__slots__")
    assert hasattr(FileFingerprint, "__slots__")
    assert hasattr(GitProvenance, "__slots__")
    assert hasattr(PackageVersion, "__slots__")
    assert hasattr(RuntimeProvenance, "__slots__")
    assert hasattr(TrackProvenance, "__slots__")
    with pytest.raises(FrozenInstanceError):
        value.protocol_version = 2  # type: ignore[misc]


@pytest.mark.parametrize("schema_version", [0, 2, True, "1"])
def test_schema_version_is_exact_integer_one(schema_version: object) -> None:
    with pytest.raises(ValueError):
        _run(schema_version=schema_version)


@pytest.mark.parametrize("protocol_version", [0, -1, True, 1.5])
def test_protocol_version_must_be_positive_integer(protocol_version: object) -> None:
    with pytest.raises(ValueError):
        _run(protocol_version=protocol_version)


@pytest.mark.parametrize("experiment_id", ["", "   ", None, 1])
def test_experiment_id_must_be_nonblank_string(experiment_id: object) -> None:
    with pytest.raises(ValueError):
        _run(experiment_id=experiment_id)


@pytest.mark.parametrize("status", ["", "pilot", None])
def test_experiment_status_is_pinned(status: object) -> None:
    with pytest.raises(ValueError):
        _run(experiment_status=status)


@pytest.mark.parametrize("status", ["", "draft", None])
def test_training_protocol_status_is_pinned(status: object) -> None:
    with pytest.raises(ValueError):
        _run(training_protocol_status=status)


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("A" * 64, "sha"),
        ("a" * 63, "sha"),
        ("g" * 64, "sha"),
        ("D" * 40, "git"),
        ("d" * 39, "git"),
        ("z" * 40, "git"),
    ],
)
def test_hashes_require_lowercase_hex(value: str, kind: str) -> None:
    with pytest.raises(ValueError):
        if kind == "sha":
            FileFingerprint("experiment_definition", "config.yml", value)
        else:
            GitProvenance(value, False)


@pytest.mark.parametrize("dirty", [0, 1, None, "false"])
def test_git_dirty_requires_real_bool(dirty: object) -> None:
    with pytest.raises(ValueError):
        GitProvenance("d" * 40, dirty)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "role",
    ["", "experiment", "runtime", None],
)
def test_file_role_is_closed_v1_set(role: object) -> None:
    with pytest.raises(ValueError):
        FileFingerprint(role, "config.yml", "a" * 64)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        "",
        "   ",
        "/absolute/config.yml",
        "C:/repo/config.yml",
        "C:\\repo\\config.yml",
        "../config.yml",
        "configs/../config.yml",
        "configs\\config.yml",
    ],
)
def test_file_path_must_be_normalized_repo_relative_posix(path: str) -> None:
    with pytest.raises(ValueError):
        FileFingerprint("experiment_definition", path, "a" * 64)


def test_run_requires_exact_file_roles_order_and_unique_paths() -> None:
    files = _files()
    with pytest.raises(ValueError):
        _run(files=(files[1], files[0], files[2]))
    with pytest.raises(ValueError):
        _run(files=(files[0], files[1]))
    with pytest.raises(ValueError):
        _run(
            files=(
                files[0],
                FileFingerprint(
                    "training_runtime_config",
                    files[0].path,
                    "b" * 64,
                ),
                files[2],
            )
        )


def test_runtime_requires_exact_v1_package_order() -> None:
    packages = _runtime().packages
    with pytest.raises(ValueError):
        RuntimeProvenance(
            "3.12.10",
            "CPython",
            "Windows",
            "11",
            "AMD64",
            (packages[1], packages[0], *packages[2:]),
        )
    with pytest.raises(ValueError):
        RuntimeProvenance(
            "3.12.10",
            "CPython",
            "Windows",
            "11",
            "AMD64",
            packages[:-1],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_version", ""),
        ("python_implementation", " "),
        ("system", ""),
        ("release", ""),
        ("machine", ""),
    ],
)
def test_runtime_platform_fields_are_nonblank(field: str, value: object) -> None:
    kwargs = {
        "python_version": "3.12.10",
        "python_implementation": "CPython",
        "system": "Windows",
        "release": "11",
        "machine": "AMD64",
        "packages": _runtime().packages,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        RuntimeProvenance(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "version"),
    [("", "1"), ("numpy", ""), (" ", "1"), ("numpy", " ")],
)
def test_package_name_and_version_are_nonblank(name: str, version: str) -> None:
    with pytest.raises(ValueError):
        PackageVersion(name, version)


def test_tracks_are_nonempty_unique_and_validated() -> None:
    first = TrackProvenance("track-a", "a" * 64, "b" * 64)
    second = TrackProvenance("track-b", "c" * 64, "d" * 64)
    assert _run(tracks=(first, second)).tracks == (first, second)
    with pytest.raises(ValueError):
        _run(tracks=())
    with pytest.raises(ValueError):
        _run(
            tracks=(
                first,
                TrackProvenance("track-a", "c" * 64, "d" * 64),
            )
        )
    with pytest.raises(ValueError):
        TrackProvenance("", "a" * 64, "b" * 64)
