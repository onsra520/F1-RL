from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..sandbox.vehicle import VehicleControl

ACTION_NAMES = ("steering", "throttle")
ACTION_LOW = (-1.0, 0.0)
ACTION_HIGH = (1.0, 1.0)
ACTION_SIZE = 2


def _validate_value(value: object, name: str, lower: float, upper: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
    if not isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")


@dataclass(frozen=True, slots=True)
class PolicyAction:
    steering: float
    throttle: float

    def __post_init__(self) -> None:
        _validate_value(self.steering, "steering", -1.0, 1.0)
        _validate_value(self.throttle, "throttle", 0.0, 1.0)


def map_policy_action(action: PolicyAction) -> VehicleControl:
    if not isinstance(action, PolicyAction):
        raise ValueError("action must be a PolicyAction")
    return VehicleControl(action.steering, action.throttle, 0.0)
