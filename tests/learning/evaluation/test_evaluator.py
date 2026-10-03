from __future__ import annotations

import numpy as np
import pytest

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.evaluation import EvaluationPolicy, evaluate_episode
from f1rl.learning.gym_environment import F1RLEnv
from f1rl.learning.training import F1RLEnvFactory
from f1rl.runtime import SimulationConfig
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.perception import SensorSuite
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState

from .mocks.environments import ScriptedEvaluationEnv
from .mocks.policies import ConstantPolicy, FailingPolicy


def _factory(*, max_steps: int = 3) -> F1RLEnvFactory:
    track = TrackWorld(
        finish_lines=(
            FinishLine(
                "finish",
                OrientedBoxVolume(
                    Vec3(0.0, 0.0, 100.0),
                    Vec3(1.0, 1.0, 1.0),
                    Quaternion.identity(),
                ),
            ),
        )
    )
    initial_state = WorldState(
        VehicleState(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            False,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )
    return F1RLEnvFactory(
        track=track,
        initial_state=initial_state,
        vehicle_parameters=VehicleParameters(
            wheelbase=2.5,
            ride_height=0.3,
            max_steering_angle=0.6,
            max_acceleration=1.0,
            max_braking=0.0,
            rolling_drag=0.0,
            air_drag=0.0,
            max_ground_speed=50.0,
            gravity=9.81,
        ),
        simulation_config=SimulationConfig(0.1, 0.5),
        observation_encoder=ObservationEncoder(
            ObservationConfig(SensorSuite(()), 20.0, 100.0)
        ),
        reward_composer=RewardComposer(()),
        episode_config=EpisodeConfig(max_steps),
    )


def _reset_info(position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, object]:
    return {
        "telemetry": {
            "vehicle_position": position,
        }
    }


def _step_info(
    *,
    tick: int,
    elapsed: float,
    position: tuple[float, float, float],
    speed: float,
    grounded: bool,
    components: dict[str, float],
    obstacle_ids: tuple[str, ...] = (),
    boost_ids: tuple[str, ...] = (),
    slow_ids: tuple[str, ...] = (),
    next_checkpoint_index: int = 0,
    finished: bool = False,
) -> dict[str, object]:
    return {
        "tick": tick,
        "elapsed_time": elapsed,
        "next_checkpoint_index": next_checkpoint_index,
        "finished": finished,
        "reward_components": components,
        "telemetry": {
            "vehicle_position": position,
            "vehicle_speed": speed,
            "grounded": grounded,
            "obstacle_hit_ids": obstacle_ids,
            "applied_boost_pad_ids": boost_ids,
            "applied_slow_zone_ids": slow_ids,
        },
    }


def test_evaluation_policy_is_structural_and_runtime_checkable() -> None:
    policy = ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32))
    assert isinstance(policy, EvaluationPolicy)
    assert not isinstance(object(), EvaluationPolicy)


@pytest.mark.parametrize(
    ("track_id", "factory", "policy"),
    [
        ("", _factory(), ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32))),
        ("track", object(), ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32))),
        ("track", _factory(), object()),
    ],
)
def test_evaluate_episode_rejects_invalid_static_arguments(
    track_id: object,
    factory: object,
    policy: object,
) -> None:
    with pytest.raises(ValueError):
        evaluate_episode(
            track_id=track_id,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            policy=policy,  # type: ignore[arg-type]
        )


def test_real_timeout_rollout_is_deterministic_and_uses_environment_terminal_contract() -> None:
    policy = ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32))

    result = evaluate_episode(track_id="fixture", factory=_factory(), policy=policy)

    assert policy.deterministic_calls == [True, True, True]
    assert result.success is False
    assert result.truncated is True
    assert result.steps == 3
    assert result.episode_time == pytest.approx(0.3)
    assert result.completion_time is None
    assert result.episode_return == 0.0
    assert result.reward_component_totals == ()
    assert result.collision_count == 0
    assert result.final_normalized_route_progress == pytest.approx(0.0)
    assert result.path_length > 0.0
    assert result.mean_speed > 0.0
    assert result.max_speed >= result.mean_speed
    assert result.airborne_steps == 3
    assert result.boost_applied_steps == 0
    assert result.slow_zone_applied_steps == 0


def test_evaluator_creates_one_fresh_env_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_calls: list[F1RLEnv] = []
    close_calls: list[F1RLEnv] = []
    original_create = F1RLEnvFactory.create
    original_close = F1RLEnv.close

    def tracked_create(self: F1RLEnvFactory) -> F1RLEnv:
        env = original_create(self)
        create_calls.append(env)
        return env

    def tracked_close(self: F1RLEnv) -> None:
        close_calls.append(self)
        original_close(self)

    monkeypatch.setattr(F1RLEnvFactory, "create", tracked_create)
    monkeypatch.setattr(F1RLEnv, "close", tracked_close)

    evaluate_episode(
        track_id="fixture",
        factory=_factory(max_steps=1),
        policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
    )

    assert len(create_calls) == 1
    assert close_calls == create_calls


def test_evaluator_closes_environment_when_policy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[F1RLEnv] = []
    original_close = F1RLEnv.close

    def tracked_close(self: F1RLEnv) -> None:
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(F1RLEnv, "close", tracked_close)

    with pytest.raises(RuntimeError, match="policy failed"):
        evaluate_episode(
            track_id="fixture",
            factory=_factory(),
            policy=FailingPolicy(RuntimeError("policy failed")),
        )

    assert len(closed) == 1


def test_evaluator_closes_environment_when_step_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[F1RLEnv] = []
    original_close = F1RLEnv.close

    def tracked_close(self: F1RLEnv) -> None:
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(F1RLEnv, "close", tracked_close)
    monkeypatch.setattr(
        F1RLEnv,
        "step",
        lambda self, action: (_ for _ in ()).throw(RuntimeError("step failed")),
    )

    with pytest.raises(RuntimeError, match="step failed"):
        evaluate_episode(
            track_id="fixture",
            factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
        )

    assert len(closed) == 1


def test_environment_retains_action_validation_ownership() -> None:
    with pytest.raises(ValueError):
        evaluate_episode(
            track_id="fixture",
            factory=_factory(),
            policy=ConstantPolicy(np.array([2.0, 0.0], dtype=np.float32)),
        )


def test_scripted_evidence_accumulates_exact_episode_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedEvaluationEnv(
        _reset_info(),
        (
            (
                np.zeros(1, dtype=np.float32),
                -3.6,
                False,
                False,
                _step_info(
                    tick=1,
                    elapsed=0.1,
                    position=(0.0, 0.0, 1.0),
                    speed=1.0,
                    grounded=True,
                    components={
                        "target_progress": 1.5,
                        "collision": -5.0,
                        "time": -0.1,
                    },
                    obstacle_ids=("obstacle-a", "obstacle-b"),
                    boost_ids=("boost-a", "boost-b"),
                ),
            ),
            (
                np.zeros(1, dtype=np.float32),
                0.4,
                False,
                True,
                _step_info(
                    tick=2,
                    elapsed=0.2,
                    position=(0.0, 0.0, 3.0),
                    speed=3.0,
                    grounded=False,
                    components={
                        "target_progress": 0.5,
                        "collision": 0.0,
                        "time": -0.1,
                    },
                    slow_ids=("slow-a", "slow-b"),
                ),
            ),
        ),
    )
    monkeypatch.setattr(F1RLEnvFactory, "create", lambda self: scripted)
    policy = ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32))

    result = evaluate_episode(track_id="track-a", factory=_factory(), policy=policy)

    assert scripted.reset_calls == 1
    assert scripted.step_calls == 2
    assert scripted.close_calls == 1
    assert policy.deterministic_calls == [True, True]
    assert result.success is False
    assert result.truncated is True
    assert result.steps == 2
    assert result.episode_time == pytest.approx(0.2)
    assert result.completion_time is None
    assert result.episode_return == pytest.approx(-3.2)
    assert tuple(name for name, _ in result.reward_component_totals) == (
        "target_progress",
        "collision",
        "time",
    )
    assert tuple(value for _, value in result.reward_component_totals) == pytest.approx(
        (2.0, -5.0, -0.2)
    )
    assert result.collision_count == 2
    assert result.path_length == pytest.approx(3.0)
    assert result.final_normalized_route_progress == pytest.approx(0.03)
    assert result.mean_speed == pytest.approx(2.0)
    assert result.max_speed == pytest.approx(3.0)
    assert result.airborne_steps == 1
    assert result.boost_applied_steps == 1
    assert result.slow_zone_applied_steps == 1


def test_reward_component_order_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedEvaluationEnv(
        _reset_info(),
        (
            (
                np.zeros(1, dtype=np.float32),
                1.0,
                False,
                False,
                _step_info(
                    tick=1,
                    elapsed=0.1,
                    position=(0.0, 0.0, 1.0),
                    speed=1.0,
                    grounded=True,
                    components={"a": 1.0},
                ),
            ),
            (
                np.zeros(1, dtype=np.float32),
                1.0,
                False,
                True,
                _step_info(
                    tick=2,
                    elapsed=0.2,
                    position=(0.0, 0.0, 2.0),
                    speed=1.0,
                    grounded=True,
                    components={"b": 1.0},
                ),
            ),
        ),
    )
    monkeypatch.setattr(F1RLEnvFactory, "create", lambda self: scripted)

    with pytest.raises(ValueError, match="reward component"):
        evaluate_episode(
            track_id="track-a",
            factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
        )
    assert scripted.close_calls == 1


def test_missing_metric_telemetry_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _step_info(
        tick=1,
        elapsed=0.1,
        position=(0.0, 0.0, 1.0),
        speed=1.0,
        grounded=True,
        components={},
    )
    telemetry = info["telemetry"]
    assert isinstance(telemetry, dict)
    del telemetry["vehicle_speed"]
    scripted = ScriptedEvaluationEnv(
        _reset_info(),
        ((np.zeros(1, dtype=np.float32), 0.0, False, True, info),),
    )
    monkeypatch.setattr(F1RLEnvFactory, "create", lambda self: scripted)

    with pytest.raises(ValueError, match="vehicle_speed"):
        evaluate_episode(
            track_id="track-a",
            factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
        )


@pytest.mark.parametrize(
    ("tick", "elapsed", "message"),
    [
        (2, 0.1, "tick"),
        (1, 0.2, "elapsed_time"),
    ],
)
def test_tick_and_elapsed_time_must_match_executed_rollout(
    monkeypatch: pytest.MonkeyPatch,
    tick: int,
    elapsed: float,
    message: str,
) -> None:
    scripted = ScriptedEvaluationEnv(
        _reset_info(),
        (
            (
                np.zeros(1, dtype=np.float32),
                0.0,
                False,
                True,
                _step_info(
                    tick=tick,
                    elapsed=elapsed,
                    position=(0.0, 0.0, 1.0),
                    speed=1.0,
                    grounded=True,
                    components={},
                ),
            ),
        ),
    )
    monkeypatch.setattr(F1RLEnvFactory, "create", lambda self: scripted)

    with pytest.raises(ValueError, match=message):
        evaluate_episode(
            track_id="track-a",
            factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
        )


def test_terminal_flags_must_agree_with_finished_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedEvaluationEnv(
        _reset_info(),
        (
            (
                np.zeros(1, dtype=np.float32),
                0.0,
                True,
                False,
                _step_info(
                    tick=1,
                    elapsed=0.1,
                    position=(0.0, 0.0, 1.0),
                    speed=1.0,
                    grounded=True,
                    components={},
                    finished=False,
                ),
            ),
        ),
    )
    monkeypatch.setattr(F1RLEnvFactory, "create", lambda self: scripted)

    with pytest.raises(ValueError, match="terminal"):
        evaluate_episode(
            track_id="track-a",
            factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
        )


def test_episode_return_must_match_reward_component_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedEvaluationEnv(
        _reset_info(),
        (
            (
                np.zeros(1, dtype=np.float32),
                2.0,
                False,
                True,
                _step_info(
                    tick=1,
                    elapsed=0.1,
                    position=(0.0, 0.0, 1.0),
                    speed=1.0,
                    grounded=True,
                    components={"only": 1.0},
                ),
            ),
        ),
    )
    monkeypatch.setattr(F1RLEnvFactory, "create", lambda self: scripted)

    with pytest.raises(ValueError, match="reward"):
        evaluate_episode(
            track_id="track-a",
            factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
        )


def test_reset_must_provide_initial_accepted_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted = ScriptedEvaluationEnv(
        {"telemetry": {}},
        (),
    )
    monkeypatch.setattr(F1RLEnvFactory, "create", lambda self: scripted)

    with pytest.raises(ValueError, match="vehicle_position"):
        evaluate_episode(
            track_id="track-a",
            factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
        )
    assert scripted.close_calls == 1
