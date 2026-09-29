from dataclasses import FrozenInstanceError, replace

import gymnasium as gym
import numpy as np
import pytest

from f1rl.learning import ObservationConfig, ObservationEncoder, RewardComposer, TimePenalty
from f1rl.learning.training import env_factory
from f1rl.learning.training.env_factory import F1RLEnvFactory
from f1rl.runtime.simulation import SimulationConfig
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.perception.rangefinder import RangefinderSpec
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from tests.learning.training.test_env_factory import _factory


def test_scenario_set_accepts_separately_instantiated_value_equivalent_contracts() -> None:
    def make_factory() -> F1RLEnvFactory:
        factory = _factory()
        encoder = ObservationEncoder(ObservationConfig(
            SensorSuite((RangefinderSpec("front", Vec3(0.0, 0.0, 1.0), 30.0),)),
            20.0,
            100.0,
        ))
        return replace(factory, observation_encoder=encoder)

    first = make_factory()
    second = make_factory()

    scenarios = env_factory.TrainingScenarioSet((
        env_factory.TrainingScenario("north", first),
        env_factory.TrainingScenario("south", second),
    ))

    assert scenarios.scenarios[0].track_id == "north"
    assert scenarios.scenarios[1].track_id == "south"
    assert scenarios.action_space == first.create().action_space
    assert scenarios.observation_space == first.create().observation_space


@pytest.mark.parametrize("track_id", ["", "   ", 3, None])
def test_training_scenario_rejects_invalid_track_id(track_id: object) -> None:
    with pytest.raises(ValueError):
        env_factory.TrainingScenario(track_id, _factory())  # type: ignore[arg-type]


def test_training_scenario_rejects_non_factory() -> None:
    with pytest.raises(ValueError):
        env_factory.TrainingScenario("north", object())  # type: ignore[arg-type]


@pytest.mark.parametrize("scenarios", [[], (), "scenario"])
def test_scenario_set_rejects_non_tuple_scenarios(scenarios: object) -> None:
    with pytest.raises(ValueError):
        env_factory.TrainingScenarioSet(scenarios)  # type: ignore[arg-type]


def test_scenario_set_rejects_empty_tuple() -> None:
    with pytest.raises(ValueError):
        env_factory.TrainingScenarioSet(())


def test_scenario_set_rejects_duplicate_track_ids() -> None:
    with pytest.raises(ValueError) as error:
        env_factory.TrainingScenarioSet((
            env_factory.TrainingScenario("same", _factory()),
            env_factory.TrainingScenario("same", _factory()),
        ))


def test_scenario_and_scenario_set_are_frozen_and_preserve_order() -> None:
    first = env_factory.TrainingScenario("first", _factory())
    second = env_factory.TrainingScenario("second", _factory())
    scenarios = env_factory.TrainingScenarioSet((first, second))

    with pytest.raises(FrozenInstanceError):
        first.track_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        scenarios.scenarios = (second, first)  # type: ignore[misc]
    assert scenarios.scenarios == (first, second)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("vehicle_parameters", lambda value: replace(value, max_acceleration=2.0)),
        ("simulation_config", lambda value: SimulationConfig(0.2, value.max_surface_distance)),
        (
            "observation_encoder",
            lambda value: ObservationEncoder(
                ObservationConfig(
                    value.config.sensor_suite,
                    value.config.velocity_scale + 1.0,
                    value.config.target_distance_scale,
                )
            ),
        ),
        ("reward_composer", lambda value: RewardComposer((TimePenalty(0.25),))),
        ("episode_config", lambda value: replace(value, max_steps=value.max_steps + 1)),
    ],
)
def test_scenario_set_rejects_semantically_different_shared_contract(
    field: str,
    changed,
) -> None:
    first = _factory()
    second = replace(first, **{field: changed(getattr(first, field))})

    with pytest.raises(ValueError) as error:
        env_factory.TrainingScenarioSet((
            env_factory.TrainingScenario("first", first),
            env_factory.TrainingScenario("second", second),
        ))


def test_scenario_set_rejects_sensor_direction_change_with_compatible_spaces() -> None:
    base = _factory()

    def encoder(direction: Vec3) -> ObservationEncoder:
        return ObservationEncoder(ObservationConfig(
            SensorSuite((RangefinderSpec("front", direction, 30.0),)),
            20.0,
            100.0,
        ))

    first = replace(base, observation_encoder=encoder(Vec3(0.0, 0.0, 1.0)))
    second = replace(base, observation_encoder=encoder(Vec3(1.0, 0.0, 0.0)))
    first_env = first.create()
    second_env = second.create()
    assert first_env.observation_space == second_env.observation_space

    with pytest.raises(ValueError):
        env_factory.TrainingScenarioSet((
            env_factory.TrainingScenario("first", first),
            env_factory.TrainingScenario("second", second),
        ))


@pytest.mark.parametrize(
    ("space_name", "alternate"),
    [
        ("observation_space", gym.spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32)),
        ("observation_space", gym.spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float64)),
        ("observation_space", gym.spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32)),
        ("action_space", gym.spaces.Box(-2.0, 1.0, shape=(2,), dtype=np.float32)),
        ("action_space", gym.spaces.Discrete(2)),
    ],
)
def test_scenario_set_rejects_incompatible_gym_spaces(
    monkeypatch: pytest.MonkeyPatch,
    space_name: str,
    alternate: gym.Space,
) -> None:
    first = _factory()
    second = _factory()
    created = 0
    closed: list[int] = []

    class SpaceEnv:
        def __init__(self, index: int) -> None:
            self.observation_space = gym.spaces.Box(
                np.array([-1.0] * 6 + [0.0] * 3, dtype=np.float32),
                np.ones(9, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Box(
                np.array([-1.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            setattr(self, space_name, alternate if index == 1 else getattr(self, space_name))
            self.index = index

        def close(self) -> None:
            closed.append(self.index)

    def create(_factory: F1RLEnvFactory) -> SpaceEnv:
        nonlocal created
        index = created
        created += 1
        return SpaceEnv(index)

    monkeypatch.setattr(F1RLEnvFactory, "create", create)
    with pytest.raises(ValueError) as error:
        env_factory.TrainingScenarioSet((
            env_factory.TrainingScenario("first", first),
            env_factory.TrainingScenario("second", second),
        ))
    assert closed == [0, 1], str(error.value)
