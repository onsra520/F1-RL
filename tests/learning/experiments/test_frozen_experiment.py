from dataclasses import replace
from pathlib import Path

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


def _complete(tmp_path: Path) -> ExperimentDefinition:
    runtime_config = tmp_path / "config.yml"
    runtime_config.write_text("training: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return ExperimentDefinition(
        metadata=ExperimentMetadata("fixture-experiment", 1, "frozen"),
        semantics=ExperimentSemantics(
            algorithm="ppo",
            policy="MlpPolicy",
            action="steering-throttle-brake0-v1",
            observation="rangefinders-local-navigation-v1",
            reward="target-progress-collision-time-v1",
            race="single-lap-ordered-checkpoints-v1",
        ),
        # Fixture-only mechanics values; these are not research defaults.
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
        tracks=TrackExperimentConfig(manifest),
        evaluation=EvaluationExperimentConfig(True),
    )


def test_complete_frozen_experiment_passes_readiness_gate(tmp_path: Path) -> None:
    assert _complete(tmp_path).require_frozen() is None


def test_draft_status_is_not_frozen_ready(tmp_path: Path) -> None:
    definition = _complete(tmp_path)
    definition = replace(definition, metadata=replace(definition.metadata, status="draft"))
    with pytest.raises(ValueError):
        definition.require_frozen()


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("vehicle", "wheelbase"),
        ("vehicle", "ride_height"),
        ("vehicle", "max_steering_angle"),
        ("vehicle", "max_acceleration"),
        ("vehicle", "max_braking"),
        ("vehicle", "rolling_drag"),
        ("vehicle", "air_drag"),
        ("vehicle", "max_ground_speed"),
        ("vehicle", "gravity"),
        ("simulation", "fixed_dt"),
        ("simulation", "max_surface_distance"),
        ("observation", "velocity_scale"),
        ("observation", "target_distance_scale"),
        ("reward", "target_progress_weight_per_meter"),
        ("reward", "collision_penalty"),
        ("reward", "time_penalty_per_second"),
        ("episode", "max_steps"),
    ],
)
def test_frozen_readiness_rejects_any_unresolved_scalar(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    definition = _complete(tmp_path)
    section_value = replace(getattr(definition, section), **{field: None})
    definition = replace(definition, **{section: section_value})
    with pytest.raises(ValueError):
        definition.require_frozen()


def test_frozen_readiness_requires_at_least_one_rangefinder(tmp_path: Path) -> None:
    definition = _complete(tmp_path)
    definition = replace(
        definition,
        observation=replace(definition.observation, rangefinders=()),
    )
    with pytest.raises(ValueError):
        definition.require_frozen()


def test_frozen_readiness_requires_frozen_training_profile(tmp_path: Path) -> None:
    definition = _complete(tmp_path)
    definition = replace(
        definition,
        training=replace(definition.training, protocol_status="pilot_owned"),
    )
    with pytest.raises(ValueError):
        definition.require_frozen()


@pytest.mark.parametrize(
    "seeds",
    [
        SeedProtocol((), ()),
        SeedProtocol((101, 202, 303), (1101, 1202)),
        SeedProtocol((101, 202), (1101, 1202)),
    ],
)
def test_frozen_readiness_requires_three_or_more_paired_seed_replicates(
    tmp_path: Path,
    seeds: SeedProtocol,
) -> None:
    definition = replace(_complete(tmp_path), seeds=seeds)
    with pytest.raises(ValueError):
        definition.require_frozen()


def test_frozen_readiness_requires_existing_reference_files(tmp_path: Path) -> None:
    definition = _complete(tmp_path)
    definition.training.runtime_config.unlink()
    with pytest.raises(ValueError):
        definition.require_frozen()

    definition = _complete(tmp_path)
    definition.tracks.manifest.unlink()
    with pytest.raises(ValueError):
        definition.require_frozen()


def test_frozen_readiness_requires_deterministic_policy_evaluation(tmp_path: Path) -> None:
    definition = replace(
        _complete(tmp_path),
        evaluation=EvaluationExperimentConfig(False),
    )
    with pytest.raises(ValueError):
        definition.require_frozen()
