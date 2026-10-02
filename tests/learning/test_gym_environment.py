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
        "telemetry": {
            "vehicle_position": (0.0, 0.0, 0.0),
            "vehicle_velocity": (0.0, 0.0, 0.0),
            "vehicle_speed": 0.0,
            "grounded": False,
            "obstacle_hit_ids": (),
            "applied_boost_pad_ids": (),
            "applied_slow_zone_ids": (),
        },
    }
    telemetry = info["telemetry"]
    assert set(telemetry) == {
        "vehicle_position",
        "vehicle_velocity",
        "vehicle_speed",
        "grounded",
        "obstacle_hit_ids",
        "applied_boost_pad_ids",
        "applied_slow_zone_ids",
    }
    assert all(type(value) is float for value in telemetry["vehicle_position"])
    assert all(type(value) is float for value in telemetry["vehicle_velocity"])
    assert type(telemetry["vehicle_speed"]) is float
    assert type(telemetry["grounded"]) is bool


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


def test_step_before_reset_is_rejected() -> None:
    env = _env()

    with pytest.raises(RuntimeError):
        env.step(np.array([0.0, 0.0], dtype=np.float32))


@pytest.mark.parametrize(
    "action",
    [
        [0.0, 0.0],
        np.array([0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([False, True], dtype=np.bool_),
        np.array([0.0, "x"], dtype=object),
        np.array([np.nan, 0.0], dtype=np.float32),
        np.array([np.inf, 0.0], dtype=np.float32),
        np.array([-1.01, 0.0], dtype=np.float32),
        np.array([0.0, 1.01], dtype=np.float32),
    ],
)
def test_invalid_action_fails_before_simulation_mutation(action: object) -> None:
    env = _env()
    env.reset()
    state_before = env._simulation.snapshot
    progress_before = env._progress

    with pytest.raises(ValueError):
        env.step(action)

    assert env._simulation.snapshot == state_before
    assert env._progress == progress_before


def test_sampled_action_advances_one_tick_and_returns_gym_tuple() -> None:
    env = _env()
    env.reset()

    result = env.step(env.action_space.sample())

    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert env._simulation.snapshot.tick == 1
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["tick"] == 1


def test_step_telemetry_matches_accepted_post_step_vehicle_state() -> None:
    env = _env()
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    vehicle = env._simulation.snapshot.world.vehicle
    telemetry = info["telemetry"]

    assert telemetry["vehicle_position"] == (
        float(vehicle.position.x),
        float(vehicle.position.y),
        float(vehicle.position.z),
    )
    assert telemetry["vehicle_velocity"] == (
        float(vehicle.velocity.x),
        float(vehicle.velocity.y),
        float(vehicle.velocity.z),
    )
    assert telemetry["vehicle_speed"] == float(vehicle.speed)
    vx, vy, vz = telemetry["vehicle_velocity"]
    assert telemetry["vehicle_speed"] == pytest.approx((vx * vx + vy * vy + vz * vz) ** 0.5)
    assert telemetry["grounded"] is vehicle.grounded
    assert telemetry["obstacle_hit_ids"] == ()
    assert telemetry["applied_boost_pad_ids"] == ()
    assert telemetry["applied_slow_zone_ids"] == ()


def test_simulation_step_failure_leaves_episode_active_and_progress_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from f1rl.runtime.simulation import Simulation

    env = _env()
    env.reset()
    progress_before = env._progress
    original_step = Simulation.step
    monkeypatch.setattr(Simulation, "step", lambda self, control: (_ for _ in ()).throw(RuntimeError("simulation failed")))

    with pytest.raises(RuntimeError, match="simulation failed"):
        env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert env._simulation.snapshot.tick == 0
    assert env._progress == progress_before
    monkeypatch.setattr(Simulation, "step", original_step)
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert env._simulation.snapshot.tick == 1


def test_reward_failure_after_simulation_step_requires_reset() -> None:
    class ExplodingReward:
        @property
        def name(self) -> str:
            return "explode"

        def evaluate(self, transition: object) -> object:
            raise RuntimeError("reward exploded")

    env = _env(reward_composer=RewardComposer((ExplodingReward(),)))
    env.reset()
    progress_before = env._progress

    with pytest.raises(RuntimeError, match="reward exploded"):
        env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert env._simulation.snapshot.tick == 1
    assert env._progress == progress_before
    with pytest.raises(RuntimeError):
        env.step(np.array([0.0, 0.0], dtype=np.float32))

    observation, info = env.reset()
    assert env._simulation.snapshot.tick == 0
    assert env.observation_space.contains(observation)
    assert info["tick"] == 0


def test_bad_post_step_observation_requires_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    env.reset()
    original_encode = ObservationEncoder.encode
    monkeypatch.setattr(ObservationEncoder, "encode", lambda self, track, vehicle, target: (0.0,) * 10)

    with pytest.raises(ValueError):
        env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert env._simulation.snapshot.tick == 1
    with pytest.raises(RuntimeError):
        env.step(np.array([0.0, 0.0], dtype=np.float32))

    monkeypatch.setattr(ObservationEncoder, "encode", original_encode)
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)


def test_learning_package_reexports_f1rl_env_without_regressing_existing_exports() -> None:
    import f1rl.learning as learning
    from f1rl.learning.gym_environment import F1RLEnv as DirectF1RLEnv

    assert learning.F1RLEnv is DirectF1RLEnv
    for name in (
        "ObservationConfig",
        "ObservationEncoder",
        "NavigationTarget",
        "PolicyAction",
        "RewardTransition",
        "RewardComposer",
        "RaceRoute",
        "RaceProgressState",
        "EpisodeConfig",
    ):
        assert hasattr(learning, name)


def test_framework_dependency_boundaries_remain_one_way() -> None:
    from pathlib import Path

    root = Path(__file__).parents[2]
    lower_sources = [
        *(root / "f1rl" / "sandbox").rglob("*.py"),
        *(root / "f1rl" / "runtime").rglob("*.py"),
    ]
    lower_text = "\n".join(path.read_text(encoding="utf-8") for path in lower_sources)
    assert "import gymnasium" not in lower_text
    assert "from gymnasium" not in lower_text
    assert "import numpy" not in lower_text
    assert "from numpy" not in lower_text


def test_gym_adapter_has_no_model_registration_or_rendering_dependencies() -> None:
    from pathlib import Path

    root = Path(__file__).parents[2]
    source = (root / "f1rl" / "learning" / "gym_environment.py").read_text(encoding="utf-8").lower()

    for forbidden in (
        "stable_baselines3",
        "stable-baselines3",
        "import torch",
        "from torch",
        "gymnasium.register",
        "gym.register",
        "pygame",
        "three.js",
        "rgb_array",
    ):
        assert forbidden not in source
