from pathlib import Path

import pytest

from src.learning.experiments import (
    EpisodeExperimentConfig,
    ExperimentDefinition,
    ExperimentSemantics,
    RangefinderExperimentSpec,
    RewardExperimentConfig,
    SimulationExperimentConfig,
    VehicleExperimentConfig,
)


PROFILE = Path("configs/experiments/ppo-generalization-v1.yml")


def test_checked_in_generalization_profile_is_explicit_draft() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)

    assert definition.metadata.id == "ppo-generalization-v1"
    assert definition.metadata.protocol_version == 1
    assert definition.metadata.status == "draft"
    assert definition.semantics == ExperimentSemantics(
        algorithm="ppo",
        policy="MlpPolicy",
        action="steering-throttle-brake0-v1",
        observation="rangefinders-local-navigation-v1",
        reward="target-progress-collision-time-v1",
        race="single-lap-ordered-checkpoints-v1",
    )
    assert definition.training.runtime_config == Path("config.yml").resolve()
    assert definition.training.protocol_status == "pilot_owned"
    assert definition.tracks.manifest == Path("assets/tracks/manifest.json").resolve()
    assert definition.tracks.training_ids == (
        "straight-sprint-v1",
        "slalom-v1",
        "right-turn-v1",
        "ramp-v1",
    )
    assert definition.tracks.heldout_ids == (
        "left-turn-v1",
        "offset-slalom-v1",
    )
    assert definition.evaluation.deterministic_policy is True


def test_checked_in_profile_pins_wp1b_environment_protocol() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)

    assert definition.vehicle == VehicleExperimentConfig(
        3.0,
        0.5,
        0.5,
        6.0,
        10.0,
        0.4,
        0.1,
        15.0,
        9.81,
    )
    assert definition.simulation == SimulationExperimentConfig(0.1, 0.75)
    assert definition.observation.velocity_scale == 15.0
    assert definition.observation.target_distance_scale == 80.0
    assert definition.observation.rangefinders == (
        RangefinderExperimentSpec("left-30deg", (-0.5, 0.0, 0.8660254037844386), 30.0),
        RangefinderExperimentSpec(
            "left-15deg",
            (-0.25881904510252074, 0.0, 0.9659258262890683),
            30.0,
        ),
        RangefinderExperimentSpec(
            "left-7p5deg",
            (-0.13052619222005157, 0.0, 0.9914448613738104),
            30.0,
        ),
        RangefinderExperimentSpec("front", (0.0, 0.0, 1.0), 30.0),
        RangefinderExperimentSpec(
            "right-7p5deg",
            (0.13052619222005157, 0.0, 0.9914448613738104),
            30.0,
        ),
        RangefinderExperimentSpec(
            "right-15deg",
            (0.25881904510252074, 0.0, 0.9659258262890683),
            30.0,
        ),
        RangefinderExperimentSpec("right-30deg", (0.5, 0.0, 0.8660254037844386), 30.0),
    )
    assert definition.reward == RewardExperimentConfig(1.0, 5.0, 0.1)
    assert definition.episode == EpisodeExperimentConfig(500)
    assert definition.metadata.status == "draft"
    assert definition.training.protocol_status == "pilot_owned"
    assert definition.seeds.ppo == ()
    assert definition.seeds.sampling == ()


def test_checked_in_environment_is_resolved_but_whole_experiment_is_still_draft() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)

    resolved_scalars = (
        definition.vehicle.wheelbase,
        definition.vehicle.ride_height,
        definition.vehicle.max_steering_angle,
        definition.vehicle.max_acceleration,
        definition.vehicle.max_braking,
        definition.vehicle.rolling_drag,
        definition.vehicle.air_drag,
        definition.vehicle.max_ground_speed,
        definition.vehicle.gravity,
        definition.simulation.fixed_dt,
        definition.simulation.max_surface_distance,
        definition.observation.velocity_scale,
        definition.observation.target_distance_scale,
        definition.reward.target_progress_weight_per_meter,
        definition.reward.collision_penalty,
        definition.reward.time_penalty_per_second,
        definition.episode.max_steps,
    )
    assert all(value is not None for value in resolved_scalars)
    assert len(definition.observation.rangefinders) == 7
    assert definition.metadata.status == "draft"
    assert definition.training.protocol_status == "pilot_owned"
    assert definition.seeds.ppo == ()
    assert definition.seeds.sampling == ()
    with pytest.raises(ValueError, match="experiment status must be frozen"):
        definition.require_frozen()
