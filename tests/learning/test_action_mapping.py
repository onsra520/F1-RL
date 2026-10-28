from copy import deepcopy

import pytest

from src.learning.action import PolicyAction, map_policy_action
from src.sandbox.vehicle import VehicleControl


@pytest.mark.parametrize(
    "steering,throttle",
    [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)],
)
def test_map_policy_action_copies_values_exactly(
    steering: float, throttle: float
) -> None:
    control = map_policy_action(PolicyAction(steering, throttle))
    assert control == VehicleControl(steering, throttle, 0.0)
    assert control.steering == steering
    assert control.throttle == throttle
    assert control.brake == 0.0


def test_zero_throttle_is_not_remapped_to_half_throttle() -> None:
    control = map_policy_action(PolicyAction(0.0, 0.0))
    assert control.throttle == 0.0


class _Lookalike:
    steering = 0.0
    throttle = 0.5


@pytest.mark.parametrize(
    "value",
    [None, (0.0, 0.5), {"steering": 0.0, "throttle": 0.5}, _Lookalike()],
)
def test_map_policy_action_rejects_non_policy_action(value: object) -> None:
    with pytest.raises(ValueError):
        map_policy_action(value)  # type: ignore[arg-type]


def test_mapping_is_deterministic_and_does_not_mutate_action() -> None:
    action = PolicyAction(-0.25, 0.75)
    before = deepcopy(action)
    first = map_policy_action(action)
    second = map_policy_action(action)
    equivalent = map_policy_action(deepcopy(action))
    assert first == second == equivalent
    assert action == before
