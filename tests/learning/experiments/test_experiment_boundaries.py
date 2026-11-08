from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from f1rl.learning.experiments import (
    EpisodeExperimentConfig,
    EvaluationExperimentConfig,
    ExperimentDefinition,
    ExperimentMetadata,
    ExperimentSemantics,
    ObservationExperimentConfig,
    RangefinderExperimentSpec,
    RewardExperimentConfig,
    SeedProtocol,
    SimulationExperimentConfig,
    TrackExperimentConfig,
    TrainingExperimentConfig,
    VehicleExperimentConfig,
)


def _frozen(tmp_path: Path) -> ExperimentDefinition:
    runtime_config = tmp_path / "training.yml"
    runtime_config.write_text("training: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return ExperimentDefinition(
        metadata=ExperimentMetadata("fixture", 1, "frozen"),
        semantics=ExperimentSemantics(
            "ppo",
            "MlpPolicy",
            "steering-throttle-brake0-v1",
            "rangefinders-local-navigation-v1",
            "target-progress-collision-time-v1",
            "single-lap-ordered-checkpoints-v1",
        ),
        vehicle=VehicleExperimentConfig(2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81),
        simulation=SimulationExperimentConfig(0.1, 0.5),
        observation=ObservationExperimentConfig(
            20.0,
            100.0,
            (RangefinderExperimentSpec("front", (0.0, 0.0, 1.0), 30.0),),
        ),
        reward=RewardExperimentConfig(1.0, 5.0, 0.1),
        episode=EpisodeExperimentConfig(100),
        training=TrainingExperimentConfig(runtime_config, "frozen"),
        seeds=SeedProtocol((101, 202, 303), (1101, 1202, 1303)),
        tracks=TrackExperimentConfig(manifest, ("train-a",), ("heldout-a",)),
        evaluation=EvaluationExperimentConfig(True),
    )


@pytest.mark.parametrize(
    "seeds",
    [
        SeedProtocol((101, 101, 303), (1101, 1202, 1303)),
        SeedProtocol((101, 202, 303), (1101, 1101, 1303)),
    ],
)
def test_frozen_replicates_require_unique_ppo_and_sampling_seeds(
    tmp_path: Path,
    seeds: SeedProtocol,
) -> None:
    definition = replace(_frozen(tmp_path), seeds=seeds)
    with pytest.raises(ValueError):
        definition.require_frozen()


def test_base_learning_package_does_not_reexport_experiment_api() -> None:
    import f1rl.learning as learning

    assert not hasattr(learning, "ExperimentDefinition")
    assert not hasattr(learning, "ExperimentMetadata")


def test_base_learning_import_does_not_require_pyyaml() -> None:
    code = r'''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "yaml" or name.startswith("yaml."):
        raise ImportError("blocked yaml")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import f1rl.learning
assert not hasattr(f1rl.learning, "ExperimentDefinition")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
