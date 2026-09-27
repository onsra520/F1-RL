from math import inf, nan

import pytest

from f1rl.learning.rewards.core import RewardTransition
from f1rl.learning.rewards.terms import CollisionPenalty, TargetProgressReward, TimePenalty
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from f1rl.sandbox.track.state import ScorePickupCollected
from f1rl.sandbox.vehicle import VehicleState


def vehicle(position: Vec3) -> VehicleState:
    return VehicleState(position, Vec3(0.0, 0.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True)


def transition(previous: Vec3, current: Vec3, target: Vec3 | None = None, events: tuple[object, ...] = (), dt: float = 0.1) -> RewardTransition:
    return RewardTransition(vehicle(previous), vehicle(current), target, events, dt)


@pytest.mark.parametrize("term_type", [TargetProgressReward, CollisionPenalty, TimePenalty])
@pytest.mark.parametrize("value", [True, False, "1", object(), nan, inf, -inf, -0.01])
def test_term_configuration_rejects_invalid_values(term_type, value: object) -> None:
    with pytest.raises(ValueError):
        term_type(value)


@pytest.mark.parametrize("term", [TargetProgressReward(0.0), CollisionPenalty(0.0), TimePenalty(0.0)])
def test_zero_configuration_is_valid(term) -> None:
    assert term is not None


def test_target_progress_uses_signed_3d_distance_delta() -> None:
    term = TargetProgressReward(2.0)
    toward = transition(Vec3(0, 0, 0), Vec3(1, 2, 2), Vec3(3, 6, 6))
    away = transition(Vec3(1, 2, 2), Vec3(0, 0, 0), Vec3(3, 6, 6))
    assert term.name == "target_progress"
    assert term.evaluate(toward).value == pytest.approx(6.0)
    assert term.evaluate(away).value == pytest.approx(-6.0)


def test_target_progress_handles_missing_and_equal_distance_targets() -> None:
    term = TargetProgressReward(3.0)
    no_target = transition(Vec3(0, 0, 0), Vec3(5, 0, 0), None)
    equal = transition(Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 0))
    assert term.evaluate(no_target).value == 0.0
    assert term.evaluate(equal).value == 0.0


def test_collision_penalty_fires_once_and_ignores_other_events() -> None:
    term = CollisionPenalty(5.0)
    base = (CheckpointEntered("cp", 0), FinishLineEntered("finish"), ScorePickupCollected("card", 99))
    assert term.name == "collision"
    assert term.evaluate(transition(Vec3(0,0,0), Vec3(0,0,0), events=base)).value == 0.0
    events = (ObstacleHit("a"), ObstacleHit("b"), *base)
    assert term.evaluate(transition(Vec3(0,0,0), Vec3(0,0,0), events=events)).value == -5.0


def test_time_penalty_scales_by_seconds() -> None:
    term = TimePenalty(0.1)
    assert term.name == "time"
    assert term.evaluate(transition(Vec3(0,0,0), Vec3(0,0,0), dt=0.1)).value == pytest.approx(-0.01)
    assert term.evaluate(transition(Vec3(0,0,0), Vec3(0,0,0), dt=0.05)).value == pytest.approx(-0.005)
    assert TimePenalty(0.0).evaluate(transition(Vec3(0,0,0), Vec3(0,0,0))).value == 0.0
