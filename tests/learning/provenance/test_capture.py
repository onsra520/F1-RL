from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess

import pytest

from f1rl.learning.experiments import ExperimentDefinition
from f1rl.learning.provenance import capture_run_provenance
from f1rl.track_assets import load_track_manifest


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "configs" / "experiments" / "ppo-generalization-v1.yml"


def _definition() -> ExperimentDefinition:
    return ExperimentDefinition.from_yaml(EXPERIMENT)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def test_capture_rejects_invalid_static_arguments(tmp_path: Path) -> None:
    definition = _definition()
    valid_ids = definition.tracks.training_ids

    with pytest.raises(ValueError, match="repository_root"):
        capture_run_provenance(
            repository_root="",
            experiment_path=EXPERIMENT,
            definition=definition,
            selected_track_ids=valid_ids,
        )
    with pytest.raises(ValueError, match="experiment_path"):
        capture_run_provenance(
            repository_root=ROOT,
            experiment_path="",
            definition=definition,
            selected_track_ids=valid_ids,
        )
    with pytest.raises(ValueError, match="definition"):
        capture_run_provenance(
            repository_root=ROOT,
            experiment_path=EXPERIMENT,
            definition=object(),  # type: ignore[arg-type]
            selected_track_ids=valid_ids,
        )
    for ids in ((), ("",), ("straight-sprint-v1", "straight-sprint-v1")):
        with pytest.raises(ValueError, match="selected_track_ids"):
            capture_run_provenance(
                repository_root=ROOT,
                experiment_path=EXPERIMENT,
                definition=definition,
                selected_track_ids=ids,
            )


def test_capture_reloads_and_rejects_definition_file_mismatch() -> None:
    definition = _definition()
    changed = replace(
        definition,
        metadata=replace(definition.metadata, id="different-id"),
    )

    with pytest.raises(ValueError, match="does not match"):
        capture_run_provenance(
            repository_root=ROOT,
            experiment_path=EXPERIMENT,
            definition=changed,
            selected_track_ids=definition.tracks.training_ids,
        )


def test_capture_rejects_experiment_file_outside_repository(tmp_path: Path) -> None:
    outside = tmp_path / "experiment.yml"
    outside.write_bytes(EXPERIMENT.read_bytes())

    with pytest.raises(ValueError, match="inside repository_root"):
        capture_run_provenance(
            repository_root=ROOT,
            experiment_path=outside,
            definition=ExperimentDefinition.from_yaml(outside),
            selected_track_ids=("straight-sprint-v1",),
        )


def test_capture_fingerprints_exact_canonical_source_bytes() -> None:
    definition = _definition()
    value = capture_run_provenance(
        repository_root=ROOT,
        experiment_path=EXPERIMENT,
        definition=definition,
        selected_track_ids=definition.tracks.training_ids,
    )

    expected = (
        (
            "experiment_definition",
            "configs/experiments/ppo-generalization-v1.yml",
            hashlib.sha256(EXPERIMENT.read_bytes()).hexdigest(),
        ),
        (
            "training_runtime_config",
            "config.yml",
            hashlib.sha256(definition.training.runtime_config.read_bytes()).hexdigest(),
        ),
        (
            "track_manifest",
            "assets/tracks/manifest.json",
            hashlib.sha256(definition.tracks.manifest.read_bytes()).hexdigest(),
        ),
    )
    assert tuple((item.role, item.path, item.sha256) for item in value.files) == expected


def test_capture_records_current_experiment_lifecycle_honestly() -> None:
    definition = _definition()
    value = capture_run_provenance(
        repository_root=ROOT,
        experiment_path=EXPERIMENT,
        definition=definition,
        selected_track_ids=definition.tracks.training_ids,
    )

    assert value.experiment_id == "ppo-generalization-v1"
    assert value.protocol_version == 1
    assert value.experiment_status == "draft"
    assert value.training_protocol_status == "pilot_owned"


def test_capture_resolves_selected_tracks_in_caller_order_from_verified_manifest() -> None:
    definition = _definition()
    requested = tuple(reversed(definition.tracks.training_ids))
    manifest = load_track_manifest(definition.tracks.manifest)
    by_id = {entry.track_id: entry for entry in manifest.tracks}

    value = capture_run_provenance(
        repository_root=ROOT,
        experiment_path=EXPERIMENT,
        definition=definition,
        selected_track_ids=requested,
    )

    assert tuple(item.track_id for item in value.tracks) == requested
    assert tuple(
        (item.world_sha256, item.content_sha256) for item in value.tracks
    ) == tuple(
        (by_id[track_id].world_sha256, by_id[track_id].content_sha256)
        for track_id in requested
    )


def test_capture_rejects_missing_selected_track_id() -> None:
    definition = _definition()
    with pytest.raises(ValueError, match="missing-track"):
        capture_run_provenance(
            repository_root=ROOT,
            experiment_path=EXPERIMENT,
            definition=definition,
            selected_track_ids=("missing-track",),
        )


def test_capture_is_equal_for_unchanged_inputs() -> None:
    definition = _definition()
    kwargs = {
        "repository_root": ROOT,
        "experiment_path": EXPERIMENT,
        "definition": definition,
        "selected_track_ids": definition.tracks.training_ids,
    }

    first = capture_run_provenance(**kwargs)
    second = capture_run_provenance(**kwargs)

    assert first == second
    assert first.git.commit_sha == _git("rev-parse", "HEAD").strip()
    expected_dirty = bool(
        _git("status", "--porcelain=v1", "--untracked-files=all").strip()
    )
    assert first.git.dirty is expected_dirty
