import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.training.config import VectorEnvConfig
from f1rl.learning.training.env_factory import F1RLEnvFactory
from f1rl.learning.training.vector_env import _RankedEnvFactory, make_vector_env
from f1rl.runtime.simulation import SimulationConfig
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState


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
        actions = np.zeros((4, 2), dtype=np.float32)
        next_obs, rewards, dones, infos = vector.step(actions)
        assert next_obs.shape == (4, 9)
        assert next_obs.dtype == np.float32
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
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
