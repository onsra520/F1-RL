import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

from f1rl.learning.experiments import ExperimentDefinition
from f1rl.learning.provenance import capture_run_provenance, write_run_provenance
from f1rl.track_assets import load_track_manifest


ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "configs" / "experiments" / "ppo-generalization-v1.yml"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def test_canonical_training_provenance_binds_real_project_identity(
    tmp_path: Path,
) -> None:
    definition = ExperimentDefinition.from_yaml(EXPERIMENT)
    selected = definition.tracks.training_ids

    provenance = capture_run_provenance(
        repository_root=ROOT,
        experiment_path=EXPERIMENT,
        definition=definition,
        selected_track_ids=selected,
    )

    assert provenance.schema_version == 1
    assert provenance.experiment_id == "ppo-generalization-v1"
    assert provenance.protocol_version == 1
    assert provenance.experiment_status == "draft"
    assert provenance.training_protocol_status == "pilot_owned"

    assert tuple((item.role, item.path) for item in provenance.files) == (
        (
            "experiment_definition",
            "configs/experiments/ppo-generalization-v1.yml",
        ),
        ("training_runtime_config", "config.yml"),
        ("track_manifest", "assets/tracks/manifest.json"),
    )
    assert tuple(item.sha256 for item in provenance.files) == (
        hashlib.sha256(EXPERIMENT.read_bytes()).hexdigest(),
        hashlib.sha256(definition.training.runtime_config.read_bytes()).hexdigest(),
        hashlib.sha256(definition.tracks.manifest.read_bytes()).hexdigest(),
    )

    assert tuple(item.track_id for item in provenance.tracks) == (
        "straight-sprint-v1",
        "slalom-v1",
        "right-turn-v1",
        "ramp-v1",
    )
    manifest = load_track_manifest(definition.tracks.manifest)
    by_id = {entry.track_id: entry for entry in manifest.tracks}
    assert tuple(
        (item.world_sha256, item.content_sha256)
        for item in provenance.tracks
    ) == tuple(
        (
            by_id[track_id].world_sha256,
            by_id[track_id].content_sha256,
        )
        for track_id in selected
    )

    package_names = (
        "numpy",
        "gymnasium",
        "stable-baselines3",
        "torch",
        "PyYAML",
    )
    assert tuple(item.name for item in provenance.runtime.packages) == package_names
    assert tuple(item.version for item in provenance.runtime.packages) == tuple(
        metadata.version(name) for name in package_names
    )

    assert provenance.git.commit_sha == _git("rev-parse", "HEAD").strip()
    expected_dirty = bool(
        _git("status", "--porcelain=v1", "--untracked-files=all").strip()
    )
    assert provenance.git.dirty is expected_dirty

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_run_provenance(first, provenance)
    write_run_provenance(second, provenance)

    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    serialized = first.read_text(encoding="utf-8")
    assert str(ROOT.resolve()) not in serialized
    assert "\\" not in serialized
