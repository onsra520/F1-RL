from pathlib import Path

import numpy as np

from src.rl.episode import EpisodeConfig
from src.rl.experiments import ExperimentDefinition
from src.rl.gym_environment import F1RLEnv
from src.rl.observation import ObservationConfig, ObservationEncoder
from src.rl.rewards import (
    CollisionPenalty,
    RewardComposer,
    TargetProgressReward,
    TimePenalty,
)
from src.core.simulation import SimulationConfig
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.state import InteractionState
from src.simulation.perception import RangefinderSpec, SensorSuite
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.state import TrackState
from src.simulation.vehicle import VehicleParameters, VehicleState
from src.simulation.world import WorldState
from src.tracks import load_track_asset


PROFILE = Path("configs/experiments/ppo-generalization-v1.yml")
TRACK_ROOT = Path("assets/tracks")
def _protocol():
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
    sensor_suite = SensorSuite(
        tuple(
            RangefinderSpec(spec.id, Vec3(*spec.local_direction), spec.max_distance)
            for spec in definition.observation.rangefinders
        )
    )
    observation = ObservationEncoder(
        ObservationConfig(
            sensor_suite,
            definition.observation.velocity_scale,
            definition.observation.target_distance_scale,
        )
    )
    rewards = RewardComposer(
        (
            TargetProgressReward(definition.reward.target_progress_weight_per_meter),
            CollisionPenalty(definition.reward.collision_penalty),
            TimePenalty(definition.reward.time_penalty_per_second),
        )
    )
    episode = EpisodeConfig(definition.episode.max_steps)
    return definition, vehicle, simulation, observation, rewards, episode


def _canonical_env(
    relative_path: str,
    *,
    position: Vec3,
    velocity: Vec3,
    grounded: bool,
) -> F1RLEnv:
    definition, vehicle, simulation, observation, rewards, episode = _protocol()
    asset = load_track_asset(TRACK_ROOT / relative_path)
    initial = WorldState(
        VehicleState(
            position,
            velocity,
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            grounded,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    return F1RLEnv(
        asset.world,
        initial,
        vehicle,
        simulation,
        observation,
        rewards,
        episode,
    )


def _ids(items) -> tuple[str, ...]:
    return tuple(item.id for item in items)


def test_telemetry_is_info_only_and_preserves_canonical_policy_contract() -> None:
    env = _canonical_env(
        "training/slalom-v1.json",
        position=Vec3(0.0, 0.5, 36.0),
        velocity=Vec3(0.0, 0.0, 5.0),
        grounded=True,
    )

    observation, reset_info = env.reset()

    assert observation.shape == (16,)
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(
        env.action_space.low,
        np.array([-1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        env.action_space.high,
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert set(reset_info) == {
        "tick",
        "elapsed_time",
        "next_checkpoint_index",
        "finished",
        "completed_checkpoint_id",
        "finished_now",
        "game_score",
        "reward_components",
        "telemetry",
    }
    assert len(reset_info["telemetry"]) == 7
    assert not any(
        name in env._observation_encoder.feature_names
        for name in (
            "vehicle_position",
            "obstacle_hit_ids",
            "applied_boost_pad_ids",
            "applied_slow_zone_ids",
            "track_id",
        )
    )

    next_observation, reward, _, _, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    assert next_observation.shape == (16,)
    assert next_observation.dtype == np.float32
    assert reward == sum(info["reward_components"].values())
    assert len(info["telemetry"]) == 7


def test_grounded_canonical_boost_and_slow_are_reported_as_applied() -> None:
    boost_env = _canonical_env(
        "training/slalom-v1.json",
        position=Vec3(0.0, 0.5, 36.0),
        velocity=Vec3(0.0, 0.0, 5.0),
        grounded=True,
    )
    _, boost_reset = boost_env.reset()
    assert boost_reset["telemetry"]["applied_boost_pad_ids"] == ()
    boost_query = boost_env._simulation.snapshot.query
    assert boost_query.contact is not None
    assert _ids(boost_query.boost_pads) == ("boost-0",)
    _, _, _, _, boost_info = boost_env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert boost_info["telemetry"]["applied_boost_pad_ids"] == ("boost-0",)
    assert boost_info["telemetry"]["applied_slow_zone_ids"] == ()

    slow_env = _canonical_env(
        "training/right-turn-v1.json",
        position=Vec3(0.0, 0.5, 50.0),
        velocity=Vec3(0.0, 0.0, 10.0),
        grounded=True,
    )
    slow_env.reset()
    slow_query = slow_env._simulation.snapshot.query
    assert slow_query.contact is not None
    assert _ids(slow_query.slow_zones) == ("slow-0",)

    _, _, _, _, slow_info = slow_env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert slow_info["telemetry"]["applied_slow_zone_ids"] == ("slow-0",)
    assert slow_info["telemetry"]["applied_boost_pad_ids"] == ()


def test_airborne_spatial_overlap_does_not_count_modifier_as_applied() -> None:
    env = _canonical_env(
        "training/slalom-v1.json",
        position=Vec3(0.0, 1.0, 36.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        grounded=False,
    )
    env.reset()
    query_before = env._simulation.snapshot.query
    assert query_before.contact is None
    assert _ids(query_before.boost_pads) == ("boost-0",)

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert info["telemetry"]["applied_boost_pad_ids"] == ()
    assert info["telemetry"]["applied_slow_zone_ids"] == ()


def test_modifier_entry_is_delayed_until_next_tick() -> None:
    env = _canonical_env(
        "training/slalom-v1.json",
        position=Vec3(0.0, 0.5, 31.5),
        velocity=Vec3(0.0, 0.0, 10.0),
        grounded=True,
    )
    env.reset()
    query_before = env._simulation.snapshot.query
    assert _ids(query_before.boost_pads) == ()

    _, _, _, _, entry_info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    query_after = env._simulation.snapshot.query

    assert _ids(query_after.boost_pads) == ("boost-0",)
    assert entry_info["telemetry"]["applied_boost_pad_ids"] == ()

    _, _, _, _, next_info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert next_info["telemetry"]["applied_boost_pad_ids"] == ("boost-0",)
def test_modifier_exit_still_counts_for_tick_that_started_inside() -> None:
    env = _canonical_env(
        "training/slalom-v1.json",
        position=Vec3(0.0, 0.5, 39.5),
        velocity=Vec3(0.0, 0.0, 10.0),
        grounded=True,
    )
    env.reset()
    query_before = env._simulation.snapshot.query
    assert _ids(query_before.boost_pads) == ("boost-0",)

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    query_after = env._simulation.snapshot.query

    assert _ids(query_after.boost_pads) == ()
    assert info["telemetry"]["applied_boost_pad_ids"] == ("boost-0",)
