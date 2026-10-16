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


@dataclass(frozen=True, slots=True)
class BrakingPolicyAction(PolicyAction):
    brake: float

    def __post_init__(self) -> None:
        PolicyAction.__post_init__(self)
        _validate_value(self.brake,"brake",0.,1.)


@dataclass(frozen=True, slots=True)
class ActionContract:
    version: str = "steering-throttle-brake0-v1"

    def __post_init__(self):
        if self.version not in ("steering-throttle-brake0-v1","steering-throttle-brake-v2"):
            raise ValueError("unregistered action contract")

    @property
    def low(self):
        return ACTION_LOW if self.version == "steering-throttle-brake0-v1" else (*ACTION_LOW,0.)

    @property
    def high(self):
        return (1.,)*len(self.low)


def map_policy_action(action: PolicyAction) -> VehicleControl:
    if not isinstance(action, PolicyAction):
        raise ValueError("action must be a PolicyAction")
    return VehicleControl(action.steering, action.throttle,
                          action.brake if isinstance(action,BrakingPolicyAction) else 0.)
