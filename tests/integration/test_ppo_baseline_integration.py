import numpy as np

from stable_baselines3.common.env_checker import check_env

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


def _smoke_env() -> F1RLEnv:
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


def test_real_env_is_sb3_compatible_and_accepts_predicted_action() -> None:
    env = _smoke_env()
    check_env(env, warn=False, skip_render_check=True)
    model = PPOModel.create(
        env,
        PPOConfig(n_steps=8, batch_size=4, n_epochs=1, device="cpu"),
    )
    observation, _ = env.reset(seed=0)

    action = model.predict(observation, deterministic=True)
    next_observation, reward, terminated, truncated, info = env.step(action)

    assert env.action_space.contains(action)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["tick"] == 1
    assert np.isfinite(next_observation).all()
