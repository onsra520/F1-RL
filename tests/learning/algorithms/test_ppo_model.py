import numpy as np
import pytest

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
from f1rl.learning.gym_environment import F1RLEnv
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


def _env() -> F1RLEnv:
    track = TrackWorld(
        finish_lines=(FinishLine(
            "finish",
            OrientedBoxVolume(Vec3(100.0, 0.0, 0.0), Vec3(0.5, 5.0, 5.0), Quaternion.identity()),
        ),)
    )
    state = WorldState(
        vehicle=VehicleState(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), False),
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
        ObservationConfig(
            sensor_suite=SensorSuite(()),
            velocity_scale=20.0,
            target_distance_scale=100.0,
        )
    )
    return F1RLEnv(
        track=track,
        initial_state=state,
        vehicle_parameters=parameters,
        simulation_config=SimulationConfig(0.1, 0.5),
        observation_encoder=encoder,
        reward_composer=RewardComposer(()),
        episode_config=EpisodeConfig(100),
    )


def _config(device: str = "cpu") -> PPOConfig:
    return PPOConfig(n_steps=8, batch_size=4, n_epochs=1, device=device)


def test_create_requires_concrete_environment_and_config() -> None:
    env = _env()
    with pytest.raises(ValueError):
        PPOModel.create(object(), _config())
    with pytest.raises(ValueError):
        PPOModel.create(env, object())


def test_create_forwards_baseline_configuration_to_sb3() -> None:
    env = _env()
    config = PPOConfig(
        learning_rate=1e-4,
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        gamma=0.9,
        gae_lambda=0.8,
        clip_range=0.1,
        ent_coef=0.02,
        vf_coef=0.7,
        max_grad_norm=0.4,
        seed=7,
        device="cpu",
    )

    model = PPOModel.create(env, config)
    backend = model._model

    assert backend.policy.__class__.__name__ == "ActorCriticPolicy"
    assert backend.learning_rate == pytest.approx(config.learning_rate)
    assert backend.n_steps == config.n_steps
    assert backend.batch_size == config.batch_size
    assert backend.n_epochs == config.n_epochs
    assert backend.gamma == pytest.approx(config.gamma)
    assert backend.gae_lambda == pytest.approx(config.gae_lambda)
    assert backend.clip_range(1.0) == pytest.approx(config.clip_range)
    assert backend.ent_coef == pytest.approx(config.ent_coef)
    assert backend.vf_coef == pytest.approx(config.vf_coef)
    assert backend.max_grad_norm == pytest.approx(config.max_grad_norm)
    assert backend.seed == config.seed
    assert backend.device.type == "cpu"
    assert backend.verbose == 0
    assert model.num_timesteps == 0


def test_create_accepts_auto_device() -> None:
    model = PPOModel.create(_env(), _config("auto"))
    assert model._model.device.type in {"cpu", "cuda"}


def test_explicit_cuda_must_not_silently_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import f1rl.learning.algorithms.ppo as ppo_module

    class FakePPO:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.device = type("Device", (), {"type": "cpu"})()
            self.num_timesteps = 0

    monkeypatch.setattr(ppo_module, "PPO", FakePPO)
    with pytest.raises(RuntimeError, match="CUDA"):
        PPOModel.create(_env(), _config("cuda"))


@pytest.mark.parametrize(
    "observation",
    [
        [0.0] * 9,
        np.zeros(8, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros(9, dtype=np.bool_),
        np.zeros(9, dtype=object),
        np.array(["0"] * 9),
    ],
)
def test_predict_rejects_invalid_observation_type_shape_or_dtype(observation: object) -> None:
    model = PPOModel.create(_env(), _config())
    with pytest.raises(ValueError):
        model.predict(observation)


def test_predict_rejects_non_finite_observation() -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    observation = observation.copy()
    observation[0] = np.nan
    with pytest.raises(ValueError):
        model.predict(observation)


def test_predict_rejects_observation_outside_space() -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    observation = observation.copy()
    observation[0] = 2.0
    with pytest.raises(ValueError):
        model.predict(observation)


@pytest.mark.parametrize("deterministic", [0, 1, "true", None])
def test_predict_requires_boolean_deterministic(deterministic: object) -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    with pytest.raises(ValueError):
        model.predict(observation, deterministic=deterministic)


def test_predict_returns_valid_float32_action() -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()

    action = model.predict(observation)

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert env.action_space.contains(action)
