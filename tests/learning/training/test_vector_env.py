import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from src.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from src.learning.gym_environment import F1RLEnv
from src.learning.training.config import VectorEnvConfig
from src.learning.training import env_factory, vector_env
from src.learning.training.env_factory import F1RLEnvFactory
from src.learning.training.vector_env import _RankedEnvFactory, make_vector_env
from src.runtime.simulation import SimulationConfig
from src.sandbox.entities.finish_line import FinishLine
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.state import InteractionState
from src.sandbox.perception.sensor_suite import SensorSuite
from src.sandbox.rules.scoring import ScoreState
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.state import TrackState
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleParameters, VehicleState
from src.sandbox.world import WorldState


TELEMETRY_KEYS = {
    "vehicle_position",
    "vehicle_velocity",
    "vehicle_speed",
    "grounded",
    "obstacle_hit_ids",
    "applied_boost_pad_ids",
    "applied_slow_zone_ids",
}


def _factory(*, finish_z: float = 100.0, max_steps: int = 100) -> F1RLEnvFactory:
    road = PlanarSurfacePatch(
        "road", Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0), 10.0, 200.0,
    )
    finish = FinishLine(
        "finish",
        OrientedBoxVolume(
            Vec3(0.0, 0.3, finish_z),
            Vec3(5.0, 5.0, 0.25),
            Quaternion.identity(),
        ),
    )
    track = TrackWorld(surfaces=(road,), finish_lines=(finish,))
    state = WorldState(
        VehicleState(
            Vec3(0.0, 0.3, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    parameters = VehicleParameters(2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81)
    encoder = ObservationEncoder(ObservationConfig(SensorSuite(()), 20.0, 100.0))
    return F1RLEnvFactory(
        track, state, parameters, SimulationConfig(0.1, 0.5),
        encoder, RewardComposer(()), EpisodeConfig(max_steps),
    )


@pytest.mark.parametrize(
    ("factory", "config"),
    [(object(), VectorEnvConfig()), (_factory(), object())],
)
def test_make_vector_env_rejects_wrong_input_types(factory: object, config: object) -> None:
    with pytest.raises(ValueError):
        make_vector_env(factory, config)  # type: ignore[arg-type]


def test_ranked_factory_preserves_rank_and_returns_fresh_envs() -> None:
    factory = _factory()
    ranked = _RankedEnvFactory(factory, 3)
    assert ranked.rank == 3
    assert ranked() is not ranked()


def test_dummy_vector_env_has_distinct_slots_and_expected_type() -> None:
    vector = make_vector_env(_factory(), VectorEnvConfig(n_envs=4))
    try:
        assert isinstance(vector, DummyVecEnv)
        assert vector.num_envs == 4
        assert len({id(env) for env in vector.envs}) == 4
    finally:
        vector.close()


def test_dummy_reset_and_step_have_expected_batched_shapes() -> None:
    vector = make_vector_env(_factory(), VectorEnvConfig(n_envs=4))
    try:
        observations = vector.reset()
        assert observations.shape == (4, 9)
        assert observations.dtype == np.float32
        assert len(vector.reset_infos) == 4
        assert all(set(info["telemetry"]) == TELEMETRY_KEYS for info in vector.reset_infos)
        actions = np.zeros((4, 2), dtype=np.float32)
        next_obs, rewards, dones, infos = vector.step(actions)
        assert next_obs.shape == (4, 9)
        assert next_obs.dtype == np.float32
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
        assert all(set(info["telemetry"]) == TELEMETRY_KEYS for info in infos)
        np.testing.assert_array_equal(next_obs[0], next_obs[1])
        assert rewards[0] == rewards[1]
        assert dones[0] == dones[1]
    finally:
        vector.close()


def test_dummy_slot_auto_reset_does_not_end_sibling() -> None:
    vector = make_vector_env(
        _factory(finish_z=0.5, max_steps=100),
        VectorEnvConfig(n_envs=2),
    )
    try:
        initial = vector.reset().copy()
        infos = None
        dones = np.array([False, False])
        for _ in range(20):
            actions = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
            observations, _, dones, infos = vector.step(actions)
            if dones[0]:
                break

        assert bool(dones[0]) is True
        assert bool(dones[1]) is False
        assert infos is not None
        assert "terminal_observation" in infos[0]
        assert infos[1]["tick"] > 0
        np.testing.assert_array_equal(observations[0], initial[0])
    finally:
        vector.close()


def test_dummy_vector_env_close_is_caller_owned_and_succeeds() -> None:
    vector = make_vector_env(_factory(), VectorEnvConfig(n_envs=2))
    vector.reset()
    vector.step(np.zeros((2, 2), dtype=np.float32))
    vector.close()


def _scenario_set(*, max_steps: int = 100):
    return env_factory.TrainingScenarioSet((
        env_factory.TrainingScenario("alpha", _factory(finish_z=100.0, max_steps=max_steps)),
        env_factory.TrainingScenario("beta", _factory(finish_z=101.0, max_steps=max_steps)),
        env_factory.TrainingScenario("gamma", _factory(finish_z=102.0, max_steps=max_steps)),
    ))


def _sample_ids(seed: int, rank: int, count: int, *, reset_seed: bool = False) -> list[str]:
    wrapper = vector_env._SampledF1RLEnv(_scenario_set(), seed, rank)
    try:
        return [
            wrapper.reset(seed=index if reset_seed else None)[1]["track_id"]
            for index in range(count)
        ]
    finally:
        wrapper.close()


def test_sampled_wrapper_uses_stable_rank_seed_and_does_not_sample_at_construction() -> None:
    scenarios = _scenario_set()
    wrapper = vector_env._SampledF1RLEnv(scenarios, 123, 0)
    try:
        assert wrapper.observation_space == scenarios.observation_space
        assert wrapper.action_space == scenarios.action_space
        assert wrapper._active_env is None
        assert wrapper.reset()[1]["track_id"] == "gamma"
        assert _sample_ids(123, 0, 6) == ["gamma", "beta", "alpha", "alpha", "gamma", "beta"]
        assert _sample_ids(123, 1, 6) == ["gamma", "alpha", "alpha", "beta", "alpha", "gamma"]
        assert _sample_ids(124, 0, 6) == ["beta", "beta", "alpha", "alpha", "gamma", "beta"]
    finally:
        wrapper.close()


def test_sampled_wrapper_preserves_nested_telemetry_alongside_track_id() -> None:
    wrapper = vector_env._SampledF1RLEnv(_scenario_set(), 123, 0)
    try:
        _, reset_info = wrapper.reset()
        active = wrapper._active_env
        assert active is not None
        assert set(reset_info["telemetry"]) == TELEMETRY_KEYS
        assert reset_info["track_id"] == wrapper._current_scenario.track_id
        vehicle = active._simulation.snapshot.world.vehicle
        assert reset_info["telemetry"]["vehicle_position"] == (
            float(vehicle.position.x),
            float(vehicle.position.y),
            float(vehicle.position.z),
        )

        _, _, _, _, step_info = wrapper.step(np.array([0.0, 1.0], dtype=np.float32))
        assert set(step_info["telemetry"]) == TELEMETRY_KEYS
        assert step_info["track_id"] == wrapper._current_scenario.track_id
        vehicle = active._simulation.snapshot.world.vehicle
        assert step_info["telemetry"]["vehicle_position"] == (
            float(vehicle.position.x),
            float(vehicle.position.y),
            float(vehicle.position.z),
        )
    finally:
        wrapper.close()


def test_sampled_wrapper_applies_training_telemetry_enabled_before_first_reset() -> None:
    wrapper = vector_env._SampledF1RLEnv(_scenario_set(), 123, 0)
    try:
        wrapper.set_training_telemetry_enabled(True)
        assert wrapper._active_env is None
        wrapper.reset()
        assert wrapper._active_env is not None
        assert wrapper._active_env._training_telemetry_enabled is True
        _, _, _, _, info = wrapper.step(np.array([0.0, 1.0], dtype=np.float32))
        assert "training_telemetry" in info
    finally:
        wrapper.close()


def test_sampled_wrapper_reset_seed_does_not_change_selection_stream() -> None:
    assert _sample_ids(123, 0, 8, reset_seed=True) == _sample_ids(123, 0, 8)


def test_sampled_wrapper_ranks_advance_independent_streams() -> None:
    scenarios = _scenario_set()
    rank_zero = vector_env._SampledF1RLEnv(scenarios, 123, 0)
    rank_one = vector_env._SampledF1RLEnv(scenarios, 123, 1)
    try:
        rank_zero.reset()
        first_rank_one = rank_one.reset()[1]["track_id"]
        rank_zero.reset()
        second_rank_one = rank_one.reset()[1]["track_id"]
        assert [first_rank_one, second_rank_one] == ["gamma", "alpha"]
    finally:
        rank_zero.close()
        rank_one.close()


def test_make_multi_track_vector_env_builds_distinct_dummy_slots() -> None:
    scenarios = _scenario_set()
    vector_env_instance = vector_env.make_multi_track_vector_env(
        scenarios,
        VectorEnvConfig(n_envs=2),
        sampling_seed=123,
    )
    try:
        assert isinstance(vector_env_instance, DummyVecEnv)
        assert vector_env_instance.num_envs == 2
        assert vector_env_instance.envs[0] is not vector_env_instance.envs[1]
        observations = vector_env_instance.reset()
        assert observations.shape == (2, 9)
        assert observations.dtype == np.float32
        assert [env._current_scenario.track_id for env in vector_env_instance.envs] == ["gamma", "gamma"]
    finally:
        vector_env_instance.close()


def test_dummy_auto_reset_preserves_terminal_track_id_before_sampling_next_episode() -> None:
    vector_env_instance = vector_env.make_multi_track_vector_env(
        _scenario_set(max_steps=1),
        VectorEnvConfig(n_envs=2),
        sampling_seed=123,
    )
    try:
        vector_env_instance.reset()
        terminal_ids = [env._current_scenario.track_id for env in vector_env_instance.envs]
        observations, _, dones, infos = vector_env_instance.step(
            np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        )

        assert dones.tolist() == [True, True]
        assert [info["track_id"] for info in infos] == terminal_ids
        assert all("terminal_observation" in info for info in infos)
        assert all(set(info["telemetry"]) == TELEMETRY_KEYS for info in infos)
        assert all(set(info["telemetry"]) == TELEMETRY_KEYS for info in vector_env_instance.reset_infos)
        assert all(
            info["telemetry"]["vehicle_position"]
            != reset_info["telemetry"]["vehicle_position"]
            for info, reset_info in zip(infos, vector_env_instance.reset_infos, strict=True)
        )
        assert observations.shape == (2, 9)
    finally:
        vector_env_instance.close()


@pytest.mark.parametrize(
    ("scenarios", "config", "sampling_seed"),
    [
        (object(), VectorEnvConfig(), 0),
        ("valid", object(), 0),
        ("valid", VectorEnvConfig(), True),
        ("valid", VectorEnvConfig(), "0"),
        ("valid", VectorEnvConfig(), -1),
        ("valid", VectorEnvConfig(), 2**32),
    ],
)
def test_make_multi_track_vector_env_rejects_invalid_inputs(
    scenarios: object,
    config: object,
    sampling_seed: object,
) -> None:
    if scenarios == "valid":
        scenarios = _scenario_set()
    with pytest.raises(ValueError):
        vector_env.make_multi_track_vector_env(
            scenarios, config, sampling_seed  # type: ignore[arg-type]
        )


def test_sampled_wrapper_closes_replaced_and_current_inner_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _scenario_set()
    created: list[F1RLEnv] = []
    closed: list[F1RLEnv] = []
    original_create = F1RLEnvFactory.create

    def create(factory: F1RLEnvFactory) -> F1RLEnv:
        env = original_create(factory)
        created.append(env)
        return env

    def close(env: F1RLEnv) -> None:
        closed.append(env)

    monkeypatch.setattr(F1RLEnvFactory, "create", create)
    monkeypatch.setattr(F1RLEnv, "close", close)
    wrapper = vector_env._SampledF1RLEnv(scenarios, 123, 0)
    wrapper.reset()
    first = wrapper._active_env
    wrapper.reset()
    second = wrapper._active_env
    assert first is not second
    assert first in closed

    wrapper.close()
    assert second in closed


def test_sampled_wrapper_closes_partial_env_and_clears_after_reset_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _scenario_set()
    created: list[F1RLEnv] = []
    closed: list[F1RLEnv] = []
    original_create = F1RLEnvFactory.create
    original_reset = F1RLEnv.reset
    original_close = F1RLEnv.close

    def create(factory: F1RLEnvFactory) -> F1RLEnv:
        env = original_create(factory)
        created.append(env)
        return env

    def reset(env: F1RLEnv, **kwargs):
        if len(created) == 2 and env is created[1]:
            raise RuntimeError("scenario reset failed")
        return original_reset(env, **kwargs)

    def close(env: F1RLEnv) -> None:
        closed.append(env)
        original_close(env)

    monkeypatch.setattr(F1RLEnvFactory, "create", create)
    monkeypatch.setattr(F1RLEnv, "reset", reset)
    monkeypatch.setattr(F1RLEnv, "close", close)
    wrapper = vector_env._SampledF1RLEnv(scenarios, 123, 0)
    wrapper.reset()

    with pytest.raises(RuntimeError, match="scenario reset failed"):
        wrapper.reset()

    assert wrapper._active_env is None
    assert created[0] in closed
    assert created[1] in closed
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        wrapper.step(np.zeros(2, dtype=np.float32))


def test_sampled_wrapper_clears_previous_env_when_factory_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = _scenario_set()
    created: list[F1RLEnv] = []
    closed: list[F1RLEnv] = []
    original_create = F1RLEnvFactory.create
    original_close = F1RLEnv.close
    factory_error = RuntimeError("scenario creation failed")
    create_count = 0
    wrapper = vector_env._SampledF1RLEnv(scenarios, 123, 0)

    def create(factory: F1RLEnvFactory) -> F1RLEnv:
        nonlocal create_count
        create_count += 1
        if create_count == 2:
            assert created[0] in closed
            assert wrapper._active_env is None
            raise factory_error
        env = original_create(factory)
        created.append(env)
        return env

    def close(env: F1RLEnv) -> None:
        closed.append(env)
        original_close(env)

    monkeypatch.setattr(F1RLEnvFactory, "create", create)
    monkeypatch.setattr(F1RLEnv, "close", close)
    wrapper.reset()

    with pytest.raises(RuntimeError) as exc_info:
        wrapper.reset()

    assert exc_info.value is factory_error
    assert wrapper._active_env is None
    assert wrapper._current_scenario is None
    assert created[0] in closed
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        wrapper.step(np.zeros(2, dtype=np.float32))

    wrapper.reset()
    assert wrapper._active_env is created[1]
    assert wrapper._current_scenario is not None
    wrapper.close()
