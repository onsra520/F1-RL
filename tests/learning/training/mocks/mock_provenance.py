from src.rl.provenance import (
    FileFingerprint,
    GitProvenance,
    PackageVersion,
    RunProvenance,
    RuntimeProvenance,
    TrackProvenance,
)


def make_provenance(track_ids: tuple[str, ...]) -> RunProvenance:
    return RunProvenance(
        schema_version=1,
        experiment_id="test-experiment",
        protocol_version=1,
        experiment_status="draft",
        training_protocol_status="pilot_owned",
        git=GitProvenance("1" * 40, False),
        files=(
            FileFingerprint(
                "experiment_definition",
                "configs/experiments/test.yml",
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
                PackageVersion("numpy", "1"),
                PackageVersion("gymnasium", "1"),
                PackageVersion("stable-baselines3", "1"),
                PackageVersion("torch", "1"),
                PackageVersion("PyYAML", "1"),
            ),
        ),
        tracks=tuple(
            TrackProvenance(track_id, "d" * 64, "e" * 64)
            for track_id in track_ids
        ),
    )
