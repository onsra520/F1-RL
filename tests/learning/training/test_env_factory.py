from dataclasses import replace

import numpy as np
import pytest

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.gym_environment import F1RLEnv
from f1rl.learning.training.env_factory import F1RLEnvFactory
from f1rl.runtime.simulation import SimulationConfig
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState


def _factory() -> F1RLEnvFactory:
    track = TrackWorld(
        finish_lines=(FinishLine(
            "finish",
            OrientedBoxVolume(Vec3(100.0, 0.0, 0.0), Vec3(0.5, 5.0, 5.0), Quaternion.identity()),
        ),)
    )
    state = WorldState(
        vehicle=VehicleState(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            False,
        ),
        track=TrackState(frozenset()),
        interactions=InteractionState(frozenset(), frozenset()),
        score=ScoreState(),
    )
    parameters = VehicleParameters(
        wheelbase=2.5,
        ride_height=0.3,
        max_steering_angle=0.6,
        max_acceleration=1.0,
        max_braking=0.0,
        rolling_drag=0.0,
        air_drag=0.0,
        max_ground_speed=50.0,
        gravity=9.81,
    )
    encoder = ObservationEncoder(
        ObservationConfig(SensorSuite(()), 20.0, 100.0)
    )
    return F1RLEnvFactory(
        track=track,
        initial_state=state,
        vehicle_parameters=parameters,
        simulation_config=SimulationConfig(0.1, 0.5),
        observation_encoder=encoder,
        reward_composer=RewardComposer(()),
        episode_config=EpisodeConfig(100),
    )


def test_factory_create_returns_distinct_equivalent_envs() -> None:
    factory = _factory()
    first = factory.create()
    second = factory.create()

    assert isinstance(first, F1RLEnv)
    assert isinstance(second, F1RLEnv)
    assert first is not second
    first_observation, first_info = first.reset(seed=0)
    second_observation, second_info = second.reset(seed=0)
    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info


def test_factory_envs_own_independent_runtime_state() -> None:
    first = _factory().create()
    second = _factory().create()
    first.reset()
    second.reset()

    first.step(np.array([0.0, 1.0], dtype=np.float32))

    assert first._simulation.snapshot.tick == 1
    assert second._simulation.snapshot.tick == 0
    assert second._simulation.snapshot.world == _factory().initial_state


def test_research_admission_marker_reaches_created_environment() -> None:
    environment = replace(_factory(), research_admission_required=True).create()

    assert environment.requires_scientific_admission is True


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("track", object()),
        ("initial_state", object()),
        ("vehicle_parameters", object()),
        ("simulation_config", object()),
        ("observation_encoder", object()),
        ("reward_composer", object()),
        ("episode_config", object()),
        ("research_admission_required", "yes"),
    ],
)
def test_factory_rejects_wrong_constructor_field_types(
    field: str,
    bad_value: object,
) -> None:
    factory = _factory()
    values = {
        "track": factory.track,
        "initial_state": factory.initial_state,
        "vehicle_parameters": factory.vehicle_parameters,
        "simulation_config": factory.simulation_config,
        "observation_encoder": factory.observation_encoder,
        "reward_composer": factory.reward_composer,
        "episode_config": factory.episode_config,
    }
    values[field] = bad_value
    with pytest.raises(ValueError):
        F1RLEnvFactory(**values)  # type: ignore[arg-type]
