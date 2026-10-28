from dataclasses import FrozenInstanceError, fields

import pytest

from src.learning.action import (
    ACTION_HIGH,
    ACTION_LOW,
    ACTION_NAMES,
    ACTION_SIZE,
    PolicyAction,
)


def test_action_metadata_matches_v1_layout_and_field_order() -> None:
    assert ACTION_NAMES == ("steering", "throttle")
    assert ACTION_LOW == (-1.0, 0.0)
    assert ACTION_HIGH == (1.0, 1.0)
    assert ACTION_SIZE == 2
    assert tuple(field.name for field in fields(PolicyAction)) == ACTION_NAMES


@pytest.mark.parametrize(
    "steering,throttle",
    [(-1.0, 0.0), (1.0, 1.0), (0.25, 0.75), (0, 1)],
)
def test_policy_action_accepts_valid_values(steering: float, throttle: float) -> None:
    action = PolicyAction(steering, throttle)
    assert action.steering == steering
    assert action.throttle == throttle


def test_policy_action_is_frozen() -> None:
    action = PolicyAction(0.0, 0.5)
    with pytest.raises(FrozenInstanceError):
        action.steering = 0.25  # type: ignore[misc]


@pytest.mark.parametrize(
    "steering",
    [True, False, "0", object(), float("nan"), float("inf"), float("-inf"), -1.000001, 1.000001],
)
def test_policy_action_rejects_invalid_steering(steering: object) -> None:
    with pytest.raises(ValueError):
        PolicyAction(steering, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "throttle",
    [True, False, "0", object(), float("nan"), float("inf"), float("-inf"), -0.000001, 1.000001],
)
def test_policy_action_rejects_invalid_throttle(throttle: object) -> None:
    with pytest.raises(ValueError):
        PolicyAction(0.0, throttle)  # type: ignore[arg-type]
