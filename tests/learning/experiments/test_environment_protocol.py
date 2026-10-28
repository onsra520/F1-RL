from pathlib import Path

from src.learning.episode import EpisodeConfig
from src.learning.experiments import ExperimentDefinition
from src.learning.observation import ObservationConfig
from src.learning.rewards import (
    CollisionPenalty,
    RewardComposer,
    TargetProgressReward,
    TimePenalty,
)
from src.runtime import SimulationConfig
from src.sandbox.geometry.vector import Vec3
from src.sandbox.perception import RangefinderSpec, SensorSuite
from src.sandbox.vehicle import VehicleParameters


PROFILE = Path("configs/experiments/ppo-generalization-v1.yml")


def test_wp1b_protocol_instantiates_real_environment_contracts() -> None:
    definition = ExperimentDefinition.from_yaml(PROFILE)

    vehicle = VehicleParameters(
        definition.vehicle.wheelbase,
        definition.vehicle.ride_height,
        definition.vehicle.max_steering_angle,
        definition.vehicle.max_acceleration,
        definition.vehicle.max_braking,
        definition.vehicle.rolling_drag,
        definition.vehicle.air_drag,
        definition.vehicle.max_ground_speed,
        definition.vehicle.gravity,
    )
    simulation = SimulationConfig(
        definition.simulation.fixed_dt,
        definition.simulation.max_surface_distance,
    )
    rangefinders = tuple(
        RangefinderSpec(
            spec.id,
            Vec3(*spec.local_direction),
            spec.max_distance,
        )
        for spec in definition.observation.rangefinders
    )
    sensor_suite = SensorSuite(rangefinders)
    observation = ObservationConfig(
        sensor_suite,
        definition.observation.velocity_scale,
        definition.observation.target_distance_scale,
    )
    rewards = RewardComposer(
        (
            TargetProgressReward(definition.reward.target_progress_weight_per_meter),
            CollisionPenalty(definition.reward.collision_penalty),
            TimePenalty(definition.reward.time_penalty_per_second),
        )
    )
    episode = EpisodeConfig(definition.episode.max_steps)

    assert vehicle == VehicleParameters(3.0, 0.5, 0.5, 6.0, 10.0, 0.4, 0.1, 15.0, 9.81)
    assert simulation == SimulationConfig(0.1, 0.75)
    assert tuple(spec.id for spec in observation.sensor_suite.rangefinders) == (
        "left-30deg",
        "left-15deg",
        "left-7p5deg",
        "front",
        "right-7p5deg",
        "right-15deg",
        "right-30deg",
    )
    assert tuple(spec.max_distance for spec in observation.sensor_suite.rangefinders) == (30.0,) * 7
    assert observation.velocity_scale == 15.0
    assert observation.target_distance_scale == 80.0
    assert tuple(term.name for term in rewards.terms) == ("target_progress", "collision", "time")
    assert rewards.terms == (
        TargetProgressReward(1.0),
        CollisionPenalty(5.0),
        TimePenalty(0.1),
    )
    assert episode == EpisodeConfig(500)
