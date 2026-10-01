from pathlib import Path

import pytest

from f1rl.learning.experiments import ExperimentDefinition, ExperimentSemantics


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
    assert definition.evaluation.deterministic_policy is True


def test_checked_in_profile_keeps_research_values_unresolved() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)

    assert all(
        value is None
        for value in (
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
    )
    assert definition.observation.rangefinders == ()
    assert definition.seeds.ppo == ()
    assert definition.seeds.sampling == ()


def test_checked_in_draft_cannot_be_mistaken_for_frozen_protocol() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)
    with pytest.raises(ValueError):
        definition.require_frozen()
