from pathlib import Path

from src.learning.provenance import (
    FileFingerprint,
    GitProvenance,
    PackageVersion,
    RunProvenance,
    RuntimeProvenance,
    TrackProvenance,
    _provenance_document,
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


def test_document_has_exact_stable_shape_and_order() -> None:
    document = _provenance_document(_provenance())

    assert tuple(document) == (
        "schema_version",
        "experiment",
        "git",
        "files",
        "runtime",
        "tracks",
    )
    assert document["schema_version"] == 1
    assert tuple(document["experiment"]) == (
        "id",
        "protocol_version",
        "status",
        "training_protocol_status",
    )
    assert tuple(document["git"]) == ("commit_sha", "dirty")

    files = document["files"]
    assert isinstance(files, list)
    assert [tuple(item) for item in files] == [
        ("role", "path", "sha256"),
        ("role", "path", "sha256"),
        ("role", "path", "sha256"),
    ]
    assert [item["role"] for item in files] == [
        "experiment_definition",
        "training_runtime_config",
        "track_manifest",
    ]

    runtime = document["runtime"]
    assert tuple(runtime) == (
        "python_version",
        "python_implementation",
        "system",
        "release",
        "machine",
        "packages",
    )
    assert [tuple(item) for item in runtime["packages"]] == [
        ("name", "version"),
    ] * 5

    tracks = document["tracks"]
    assert [tuple(item) for item in tracks] == [
        ("track_id", "world_sha256", "content_sha256"),
    ]


def test_document_contains_plain_json_values_only_and_no_machine_identity() -> None:
    document = _provenance_document(_provenance())

    def walk(value: object) -> None:
        assert not isinstance(value, Path)
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert isinstance(value, (str, int, bool)) or value is None

    walk(document)
    text = repr(document).lower()
    for forbidden in (
        "timestamp",
        "run_id",
        "pid",
        "hostname",
        "username",
        "device",
        "c:\\",
        "/home/",
    ):
        assert forbidden not in text
