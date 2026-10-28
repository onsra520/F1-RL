from pathlib import Path

import numpy as np
import pytest

from src.rl.algorithms.ppo import PPOConfig, PPOModel
from src.rl.episode import EpisodeConfig
from src.evaluation import EvaluationPolicy, evaluate_episode
from src.rl.experiments import ExperimentDefinition
from src.rl.observation import ObservationConfig, ObservationEncoder
from src.rl.rewards import (
    CollisionPenalty,
    RewardComposer,
    TargetProgressReward,
    TimePenalty,
)
from src.training import F1RLEnvFactory
from src.core import SimulationConfig
from src.simulation.entities.finish_line import FinishLine
from src.simulation.entities.obstacle import Obstacle
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.state import InteractionState
from src.simulation.perception import RangefinderSpec, SensorSuite
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.model import TrackWorld
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.track.state import TrackState
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.vehicle import VehicleParameters, VehicleState
from src.simulation.world import WorldState
from src.tracks import build_initial_world_state, load_track_asset

from tests.learning.evaluation.mocks.policies import ConstantPolicy


PROFILE = Path("configs/experiments/ppo-generalization-v1.yml")
TRACK_ROOT = Path("assets/tracks")


def _protocol() -> tuple[
    ExperimentDefinition,
    VehicleParameters,
    SimulationConfig,
    ObservationEncoder,
    RewardComposer,
    EpisodeConfig,
]:
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
    suite = SensorSuite(
        tuple(
            RangefinderSpec(
                spec.id,
                Vec3(*spec.local_direction),
                spec.max_distance,
            )
            for spec in definition.observation.rangefinders
        )
    )
    observation = ObservationEncoder(
        ObservationConfig(
            suite,
            definition.observation.velocity_scale,
            definition.observation.target_distance_scale,
        )
    )
    rewards = RewardComposer(
        (
            TargetProgressReward(
                definition.reward.target_progress_weight_per_meter
            ),
            CollisionPenalty(definition.reward.collision_penalty),
            TimePenalty(definition.reward.time_penalty_per_second),
        )
    )
    return (
        definition,
        vehicle,
        simulation,
        observation,
        rewards,
        EpisodeConfig(definition.episode.max_steps),
    )


def _canonical_factory(
    relative_path: str,
    *,
    initial_state: WorldState | None = None,
    episode_config: EpisodeConfig | None = None,
) -> F1RLEnvFactory:
    (
        definition,
        vehicle,
        simulation,
        observation,
        rewards,
        default_episode,
    ) = _protocol()
    asset = load_track_asset(TRACK_ROOT / relative_path)
    if initial_state is None:
        initial_state = build_initial_world_state(
            asset,
            max_surface_distance=definition.simulation.max_surface_distance,
        )
    return F1RLEnvFactory(
        track=asset.world,
        initial_state=initial_state,
        vehicle_parameters=vehicle,
        simulation_config=simulation,
        observation_encoder=observation,
        reward_composer=rewards,
        episode_config=episode_config or default_episode,
    )


def test_straight_training_track_evaluation_is_reproducible() -> None:
    factory = _canonical_factory("training/straight-sprint-v1.json")
    action = np.array([0.0, 1.0], dtype=np.float32)

    first = evaluate_episode(
        track_id="straight-sprint-v1",
        factory=factory,
        policy=ConstantPolicy(action),
    )
    second = evaluate_episode(
        track_id="straight-sprint-v1",
        factory=factory,
        policy=ConstantPolicy(action),
    )

    assert first == second
    assert first.success is True
    assert first.truncated is False
    assert first.steps == 95
    assert first.episode_time == pytest.approx(9.5)
    assert first.completion_time == pytest.approx(9.5)
    assert first.final_normalized_route_progress == 1.0
    assert np.isfinite(first.episode_return)
    assert np.isfinite(first.path_length)
    assert first.path_length > 0.0
    assert first.collision_count == 0


def test_canonical_training_boost_telemetry_becomes_evaluator_exposure() -> None:
    asset = load_track_asset(TRACK_ROOT / "training/slalom-v1.json")
    initial = WorldState(
        VehicleState(
            Vec3(0.0, 0.5, 36.0),
            Vec3(0.0, 0.0, 5.0),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    factory = _canonical_factory(
        "training/slalom-v1.json",
        initial_state=initial,
        episode_config=EpisodeConfig(1),
    )

    result = evaluate_episode(
        track_id=asset.track_id,
        factory=factory,
        policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
    )

    assert result.steps == 1
    assert result.truncated is True
    assert result.boost_applied_steps == 1
    assert result.slow_zone_applied_steps == 0


def _collision_factory() -> F1RLEnvFactory:
    _, vehicle, simulation, observation, rewards, _ = _protocol()
    road = PlanarSurfacePatch(
        "road",
        Vec3(0.0, 0.0, 50.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        10.0,
        100.0,
    )
    volume = OrientedBoxVolume(
        Vec3(0.0, 0.5, 0.06),
        Vec3(1.0, 1.0, 0.02),
        Quaternion.identity(),
    )
    track = TrackWorld(
        surfaces=(road,),
        obstacles=(
            Obstacle("obstacle-b", volume),
            Obstacle("obstacle-a", volume),
        ),
        finish_lines=(
            FinishLine(
                "finish",
                OrientedBoxVolume(
                    Vec3(0.0, 0.5, 90.0),
                    Vec3(5.0, 2.0, 1.0),
                    Quaternion.identity(),
                ),
            ),
        ),
    )
    initial = WorldState(
        VehicleState(
            Vec3(0.0, 0.5, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    return F1RLEnvFactory(
        track=track,
        initial_state=initial,
        vehicle_parameters=vehicle,
        simulation_config=simulation,
        observation_encoder=observation,
        reward_composer=rewards,
        episode_config=EpisodeConfig(1),
    )


def test_real_multi_obstacle_events_become_event_count_not_collision_tick_count() -> None:
    result = evaluate_episode(
        track_id="collision-fixture",
        factory=_collision_factory(),
        policy=ConstantPolicy(np.array([0.0, 1.0], dtype=np.float32)),
    )

    assert result.steps == 1
    assert result.truncated is True
    assert result.collision_count == 2
    assert result.path_length == pytest.approx(0.0)
    assert result.max_speed == pytest.approx(0.0)


def test_ppo_model_satisfies_evaluation_policy_without_evaluator_sb3_import() -> None:
    factory = _canonical_factory(
        "training/straight-sprint-v1.json",
        episode_config=EpisodeConfig(1),
    )
    training_env = factory.create()
    try:
        model = PPOModel.create(
            training_env,
            PPOConfig(
                n_steps=2,
                batch_size=2,
                n_epochs=1,
                seed=0,
                device="cpu",
            ),
        )
        assert isinstance(model, EvaluationPolicy)

        source = Path("src/evaluation/evaluator.py").read_text(
            encoding="utf-8"
        ).lower()
        assert "stable_baselines3" not in source
        assert "import torch" not in source
        assert "from torch" not in source

        result = evaluate_episode(
            track_id="straight-sprint-v1",
            factory=factory,
            policy=model,
        )

        assert result.steps == 1
        assert result.truncated is True
        assert np.isfinite(result.episode_return)
    finally:
        training_env.close()
