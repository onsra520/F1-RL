from dataclasses import FrozenInstanceError
from math import inf, nan
from typing import get_type_hints

import pytest

from src.rl.rewards.core import RewardContribution, RewardResult, RewardTransition
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from src.simulation.track.state import ScorePickupCollected
from src.simulation.vehicle import VehicleState
from src.simulation.world import WorldEvent


def vehicle(x: float = 0.0, y: float = 0.3, z: float = 0.0) -> VehicleState:
    return VehicleState(Vec3(x, y, z), Vec3(0.0, 0.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True)


def test_reward_transition_accepts_valid_inputs_and_is_frozen() -> None:
    transition = RewardTransition(vehicle(), vehicle(z=1.0), Vec3(0.0, 0.3, 10.0), (ObstacleHit("wall"),), 0.1)
    assert transition.dt == 0.1
    with pytest.raises(FrozenInstanceError):
        transition.dt = 0.2  # type: ignore[misc]


@pytest.mark.parametrize("field,value", [
    ("previous_vehicle", object()), ("next_vehicle", object()), ("target_position", object()),
    ("events", []), ("events", (object(),)),
])
def test_reward_transition_rejects_invalid_domain_inputs(field: str, value: object) -> None:
    kwargs = dict(previous_vehicle=vehicle(), next_vehicle=vehicle(z=1.0), target_position=None, events=(), dt=0.1)
    kwargs[field] = value
    with pytest.raises(ValueError):
        RewardTransition(**kwargs)


@pytest.mark.parametrize("dt", [True, False, "0.1", object(), nan, inf, -inf, 0.0, -0.1])
def test_reward_transition_rejects_invalid_dt(dt: object) -> None:
    with pytest.raises(ValueError):
        RewardTransition(vehicle(), vehicle(z=1.0), None, (), dt)  # type: ignore[arg-type]


def test_reward_transition_accepts_all_current_world_event_variants() -> None:
    events = (ObstacleHit("wall"), CheckpointEntered("cp", 0), FinishLineEntered("finish"), ScorePickupCollected("card", 5))
    assert RewardTransition(vehicle(), vehicle(), None, events, 0.1).events == events


@pytest.mark.parametrize("name", ["", "   ", 3, None])
def test_reward_contribution_rejects_invalid_name(name: object) -> None:
    with pytest.raises(ValueError):
        RewardContribution(name, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_reward_contribution_rejects_non_finite_value(value: float) -> None:
    with pytest.raises(ValueError):
        RewardContribution("term", value)


def test_reward_result_validates_shape_and_is_frozen() -> None:
    contribution = RewardContribution("term", 1.25)
    result = RewardResult(1.25, (contribution,))
    assert result.contributions == (contribution,)
    with pytest.raises(FrozenInstanceError):
        result.total = 2.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        RewardResult(inf, ())
    with pytest.raises(ValueError):
        RewardResult(0.0, [contribution])  # type: ignore[arg-type]


def test_reward_transition_events_annotation_matches_world_event_contract() -> None:
    assert get_type_hints(RewardTransition)["events"] == tuple[WorldEvent, ...]
