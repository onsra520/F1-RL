from dataclasses import FrozenInstanceError
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


def _semantics() -> ExperimentSemantics:
    return ExperimentSemantics(
        algorithm="ppo",
        policy="MlpPolicy",
        action="steering-throttle-brake0-v1",
        observation="rangefinders-local-navigation-v1",
        reward="target-progress-collision-time-v1",
        race="single-lap-ordered-checkpoints-v1",
    )


def _definition() -> ExperimentDefinition:
    return ExperimentDefinition(
        metadata=ExperimentMetadata("ppo-generalization-v1", 1, "draft"),
        semantics=_semantics(),
        vehicle=VehicleExperimentConfig(),
        simulation=SimulationExperimentConfig(),
        observation=ObservationExperimentConfig(),
        reward=RewardExperimentConfig(),
        episode=EpisodeExperimentConfig(),
        training=TrainingExperimentConfig(Path("config.yml"), "pilot_owned"),
        seeds=SeedProtocol((), ()),
        tracks=TrackExperimentConfig(Path("assets/tracks/manifest.json")),
        evaluation=EvaluationExperimentConfig(True),
    )


def test_experiment_contracts_are_immutable() -> None:
    definition = _definition()
    with pytest.raises(FrozenInstanceError):
        definition.metadata = ExperimentMetadata("other", 1, "draft")
    with pytest.raises(FrozenInstanceError):
        definition.metadata.status = "frozen"


@pytest.mark.parametrize(
    "args",
    [
        ("", 1, "draft"),
        ("   ", 1, "draft"),
        (123, 1, "draft"),
        ("exp", 0, "draft"),
        ("exp", -1, "draft"),
        ("exp", True, "draft"),
        ("exp", 1.5, "draft"),
        ("exp", 1, "other"),
    ],
)
def test_metadata_rejects_invalid_identity_version_or_status(args: tuple[object, object, object]) -> None:
    with pytest.raises(ValueError):
        ExperimentMetadata(*args)


def test_v1_semantic_identifiers_are_exact() -> None:
    assert _semantics() == ExperimentSemantics(
        algorithm="ppo",
        policy="MlpPolicy",
        action="steering-throttle-brake0-v1",
        observation="rangefinders-local-navigation-v1",
        reward="target-progress-collision-time-v1",
        race="single-lap-ordered-checkpoints-v1",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("algorithm", "sac"),
        ("policy", "CnnPolicy"),
        ("action", "other"),
        ("observation", "other"),
        ("reward", "other"),
        ("race", "other"),
    ],
)
def test_semantics_reject_unsupported_identifiers(field: str, value: str) -> None:
    kwargs = {
        "algorithm": "ppo",
        "policy": "MlpPolicy",
        "action": "steering-throttle-brake0-v1",
        "observation": "rangefinders-local-navigation-v1",
        "reward": "target-progress-collision-time-v1",
        "race": "single-lap-ordered-checkpoints-v1",
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        ExperimentSemantics(**kwargs)


def test_draft_environment_sections_allow_explicit_unresolved_values() -> None:
    definition = _definition()
    assert definition.vehicle.wheelbase is None
    assert definition.simulation.fixed_dt is None
    assert definition.observation.velocity_scale is None
    assert definition.observation.rangefinders == ()
    assert definition.reward.target_progress_weight_per_meter is None
    assert definition.episode.max_steps is None


def test_rangefinder_spec_uses_real_sensor_validation() -> None:
    valid = RangefinderExperimentSpec("front", (0.0, 0.0, 1.0), 30.0)
    assert valid.id == "front"

    with pytest.raises(ValueError):
        RangefinderExperimentSpec("front", (0.0, 0.0, 2.0), 30.0)
    with pytest.raises(ValueError):
        RangefinderExperimentSpec("front", (0.0, 0.0, 1.0), 0.0)
    with pytest.raises(ValueError):
        RangefinderExperimentSpec("front", (True, 0.0, 0.0), 30.0)


def test_observation_rejects_duplicate_rangefinder_ids() -> None:
    sensor = RangefinderExperimentSpec("front", (0.0, 0.0, 1.0), 30.0)
    with pytest.raises(ValueError):
        ObservationExperimentConfig(rangefinders=(sensor, sensor))


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_seed_protocol_rejects_invalid_seed_values(seed: object) -> None:
    with pytest.raises(ValueError):
        SeedProtocol((seed,), ())


def test_seed_protocol_accepts_empty_draft_and_uint32_boundaries() -> None:
    assert SeedProtocol((), ()) == SeedProtocol((), ())
    assert SeedProtocol((0, 2**32 - 1), (1, 2)) == SeedProtocol(
        (0, 2**32 - 1),
        (1, 2),
    )


def test_definition_rejects_wrong_section_types() -> None:
    definition = _definition()
    values = {
        "metadata": definition.metadata,
        "semantics": definition.semantics,
        "vehicle": definition.vehicle,
        "simulation": definition.simulation,
        "observation": definition.observation,
        "reward": definition.reward,
        "episode": definition.episode,
        "training": definition.training,
        "seeds": definition.seeds,
        "tracks": definition.tracks,
        "evaluation": definition.evaluation,
    }
    values["vehicle"] = object()
    with pytest.raises(ValueError):
        ExperimentDefinition(**values)
