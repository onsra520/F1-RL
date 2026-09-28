import numpy as np
import pytest

from f1rl.learning import (
    EpisodeConfig,
    NavigationTarget,
    ObservationConfig,
    ObservationEncoder,
    RewardComposer,
    RewardTransition,
    TargetProgressReward,
    TimePenalty,
)
from f1rl.learning.gym_environment import F1RLEnv
from f1rl.runtime.simulation import SimulationConfig
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.entities.score_pickup import ScorePickup
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


def _volume(center: Vec3) -> OrientedBoxVolume:
    return OrientedBoxVolume(center, Vec3(0.3, 10.0, 10.0), Quaternion.identity())


def _moving_state() -> WorldState:
    return WorldState(
        vehicle=VehicleState(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(10.0, 0.0, 0.0),
            orientation=Quaternion.identity(),
            angular_velocity=Vec3(0.0, 0.0, 0.0),
            grounded=False,
        ),
        track=TrackState(frozenset()),
        interactions=InteractionState(frozenset(), frozenset()),
        score=ScoreState(),
    )


def _parameters() -> VehicleParameters:
    return VehicleParameters(
        wheelbase=2.5,
        ride_height=0.3,
        max_steering_angle=0.6,
        max_acceleration=0.0,
        max_braking=0.0,
        rolling_drag=0.0,
        air_drag=0.0,
        max_ground_speed=50.0,
        gravity=9.81,
    )


def _encoder() -> ObservationEncoder:
    return ObservationEncoder(
        ObservationConfig(
            sensor_suite=SensorSuite(()),
            velocity_scale=20.0,
            target_distance_scale=100.0,
        )
    )


def _moving_env(
    track: TrackWorld,
    reward_composer: RewardComposer | None = None,
    max_steps: int = 10,
) -> F1RLEnv:
    return F1RLEnv(
        track=track,
        initial_state=_moving_state(),
        vehicle_parameters=_parameters(),
        simulation_config=SimulationConfig(0.1, 0.5),
        observation_encoder=_encoder(),
        reward_composer=reward_composer or RewardComposer(()),
        episode_config=EpisodeConfig(max_steps),
    )


def _action() -> np.ndarray:
    return np.array([0.0, 0.0], dtype=np.float32)


def test_reward_uses_pre_step_target_and_next_observation_uses_post_progress_target() -> None:
    checkpoint_center = Vec3(1.0, 0.0, 0.0)
    finish_center = Vec3(5.0, 0.0, 0.0)
    track = TrackWorld(
        checkpoints=(Checkpoint("cp-0", _volume(checkpoint_center), 0),),
        finish_lines=(FinishLine("finish", _volume(finish_center)),),
    )
    reward_term = TargetProgressReward(1.0)
    env = _moving_env(track, RewardComposer((reward_term, TimePenalty(0.5))))
    env.reset()

    observation, reward, terminated, truncated, info = env.step(_action())

    snapshot = env._simulation.snapshot
    expected_transition = RewardTransition(
        previous_vehicle=_moving_state().vehicle,
        next_vehicle=snapshot.world.vehicle,
        target_position=checkpoint_center,
        events=snapshot.events,
        dt=0.1,
    )
    expected_reward = reward_term.evaluate(expected_transition).value - 0.05
    assert reward == pytest.approx(expected_reward)
    assert info["completed_checkpoint_id"] == "cp-0"
    assert info["next_checkpoint_index"] == 1
    assert list(info["reward_components"]) == ["target_progress", "time"]
    assert terminated is False and truncated is False
    expected_observation = np.asarray(
        env._observation_encoder.encode(
            track,
            snapshot.world.vehicle,
            NavigationTarget(finish_center),
        ),
        dtype=np.float32,
    )
    np.testing.assert_allclose(observation, expected_observation)


def test_valid_finish_terminates_and_requires_reset() -> None:
    track = TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(1.0, 0.0, 0.0))),))
    env = _moving_env(track)
    env.reset()

    _, _, terminated, truncated, info = env.step(_action())

    assert terminated is True
    assert truncated is False
    assert info["finished"] is True
    assert info["finished_now"] is True
    with pytest.raises(RuntimeError):
        env.step(_action())


def test_unfinished_step_budget_truncates_and_requires_reset() -> None:
    track = TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(100.0, 0.0, 0.0))),))
    env = _moving_env(track, max_steps=1)
    env.reset()

    _, _, terminated, truncated, _ = env.step(_action())

    assert terminated is False
    assert truncated is True
    with pytest.raises(RuntimeError):
        env.step(_action())


def test_finish_exactly_at_step_budget_is_terminated_only() -> None:
    track = TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(1.0, 0.0, 0.0))),))
    env = _moving_env(track, max_steps=1)
    env.reset()

    _, _, terminated, truncated, _ = env.step(_action())

    assert terminated is True
    assert truncated is False


def test_game_score_is_not_converted_into_rl_reward() -> None:
    track = TrackWorld(
        finish_lines=(FinishLine("finish", _volume(Vec3(100.0, 0.0, 0.0))),),
        score_pickups=(ScorePickup("pickup", _volume(Vec3(1.0, 0.0, 0.0)), 5),),
    )
    env = _moving_env(track, RewardComposer((TimePenalty(1.0),)))
    env.reset()

    _, reward, terminated, truncated, info = env.step(_action())

    assert reward == pytest.approx(-0.1)
    assert info["game_score"] == 5
    assert info["reward_components"] == {"time": pytest.approx(-0.1)}
    assert terminated is False
    assert truncated is False


def test_gymnasium_env_checker_accepts_environment_without_rendering() -> None:
    from gymnasium.utils.env_checker import check_env

    track = TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(100.0, 0.0, 0.0))),))
    env = _moving_env(track)

    check_env(env, skip_render_check=True)


def test_checker_and_repeated_resets_do_not_replace_or_mutate_bound_track() -> None:
    from gymnasium.utils.env_checker import check_env

    track = TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(100.0, 0.0, 0.0))),))
    env = _moving_env(track)
    expected_track = track

    check_env(env, skip_render_check=True)
    first_observation, first_info = env.reset(seed=11)
    second_observation, second_info = env.reset(seed=22)

    assert env._track is track
    assert env._track == expected_track
    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info


def test_equal_environments_produce_equal_episodes_and_reset_recovery() -> None:
    track_a = TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(100.0, 0.0, 0.0))),))
    track_b = TrackWorld(finish_lines=(FinishLine("finish", _volume(Vec3(100.0, 0.0, 0.0))),))
    first = _moving_env(track_a, RewardComposer((TimePenalty(0.5),)), max_steps=2)
    second = _moving_env(track_b, RewardComposer((TimePenalty(0.5),)), max_steps=2)

    first_reset = first.reset(seed=1)
    second_reset = second.reset(seed=999)
    np.testing.assert_array_equal(first_reset[0], second_reset[0])
    assert first_reset[1] == second_reset[1]

    actions = (
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.25, 0.75], dtype=np.float32),
    )
    for action in actions:
        first_step = first.step(action.copy())
        second_step = second.step(action.copy())
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]

    assert first_step[3] is True
    first_after = first.reset()
    second_after = second.reset()
    np.testing.assert_array_equal(first_after[0], second_after[0])
    assert first_after[1] == second_after[1]
