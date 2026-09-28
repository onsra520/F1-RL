import numpy as np
import pytest

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.gym_environment import F1RLEnv
from f1rl.runtime.simulation import SimulationConfig
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.perception.rangefinder import RangefinderSpec
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState


def _volume(center: Vec3) -> OrientedBoxVolume:
    return OrientedBoxVolume(center, Vec3(1.0, 1.0, 1.0), Quaternion.identity())


def _track() -> TrackWorld:
    return TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(100.0, 0.0, 0.0))),))


def _initial_state() -> WorldState:
    return WorldState(
        vehicle=VehicleState(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            orientation=Quaternion.identity(),
            angular_velocity=Vec3(0.0, 0.0, 0.0),
            grounded=False,
        ),
        track=TrackState(frozenset()),
        interactions=InteractionState(frozenset(), frozenset()),
        score=ScoreState(7),
    )


def _vehicle_parameters() -> VehicleParameters:
    return VehicleParameters(
        wheelbase=2.5,
        ride_height=0.3,
        max_steering_angle=0.6,
        max_acceleration=4.0,
        max_braking=8.0,
        rolling_drag=0.1,
        air_drag=0.01,
        max_ground_speed=50.0,
        gravity=9.81,
    )


def _observation_encoder() -> ObservationEncoder:
    suite = SensorSuite(
        (
            RangefinderSpec("forward", Vec3(0.0, 0.0, 1.0), 20.0),
            RangefinderSpec("right", Vec3(1.0, 0.0, 0.0), 10.0),
        )
    )
    return ObservationEncoder(
        ObservationConfig(
            sensor_suite=suite,
            velocity_scale=20.0,
            target_distance_scale=100.0,
        )
    )


def _env(**overrides: object) -> F1RLEnv:
    kwargs = {
        "track": _track(),
        "initial_state": _initial_state(),
        "vehicle_parameters": _vehicle_parameters(),
        "simulation_config": SimulationConfig(0.1, 0.5),
        "observation_encoder": _observation_encoder(),
        "reward_composer": RewardComposer(()),
        "episode_config": EpisodeConfig(10),
    }
    kwargs.update(overrides)
    return F1RLEnv(**kwargs)


def test_environment_declares_v1_metadata_and_spaces() -> None:
    env = _env()

    assert env.metadata == {"render_modes": []}
    assert env.render_mode is None
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(env.action_space.low, np.array([-1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(env.action_space.high, np.array([1.0, 1.0], dtype=np.float32))

    assert env.observation_space.shape == (11,)
    assert env.observation_space.dtype == np.dtype(np.float32)
    expected_low = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    expected_high = np.ones(11, dtype=np.float32)
    np.testing.assert_array_equal(env.observation_space.low, expected_low)
    np.testing.assert_array_equal(env.observation_space.high, expected_high)


@pytest.mark.parametrize(
    "name",
    [
        "track",
        "initial_state",
        "vehicle_parameters",
        "simulation_config",
        "observation_encoder",
        "reward_composer",
        "episode_config",
    ],
)
def test_constructor_rejects_wrong_argument_types(name: str) -> None:
    with pytest.raises(ValueError):
        _env(**{name: object()})


@pytest.mark.parametrize(
    "track",
    [
        TrackWorld(),
        TrackWorld(
            finish_lines=(
                FinishLine("finish-a", _volume(Vec3(10.0, 0.0, 0.0))),
                FinishLine("finish-b", _volume(Vec3(20.0, 0.0, 0.0))),
            )
        ),
    ],
)
def test_constructor_rejects_tracks_without_exactly_one_finish(track: TrackWorld) -> None:
    with pytest.raises(ValueError):
        _env(track=track)


def test_reset_returns_declared_observation_and_zero_step_info() -> None:
    env = _env()

    observation, info = env.reset()

    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info == {
        "tick": 0,
        "elapsed_time": 0.0,
        "next_checkpoint_index": 0,
        "finished": False,
        "completed_checkpoint_id": None,
        "finished_now": False,
        "game_score": 7,
        "reward_components": {},
    }


def test_reset_is_deterministic_across_repeated_calls_and_seeds() -> None:
    env = _env()

    first_observation, first_info = env.reset(seed=1)
    first_world = env._simulation.snapshot.world
    second_observation, second_info = env.reset(seed=999)

    np.testing.assert_array_equal(second_observation, first_observation)
    assert second_info == first_info
    assert env._simulation.snapshot.world == first_world == _initial_state()


def test_reset_accepts_empty_options_and_rejects_nonempty_options() -> None:
    env = _env()

    observation, _ = env.reset(options={})
    assert env.observation_space.contains(observation)

    with pytest.raises(ValueError):
        env.reset(options={"track": "other"})


@pytest.mark.parametrize(
    "values",
    [
        (0.0,) * 10,
        (float("nan"),) + (0.0,) * 10,
        (2.0,) + (0.0,) * 10,
    ],
)
def test_reset_rejects_malformed_encoded_observation(
    monkeypatch: pytest.MonkeyPatch,
    values: tuple[float, ...],
) -> None:
    env = _env()
    monkeypatch.setattr(ObservationEncoder, "encode", lambda self, track, vehicle, target: values)

    with pytest.raises(ValueError):
        env.reset()
