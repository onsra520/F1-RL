"""Explicit, deterministic route-corridor failure policy.

Horizontal XZ corridor prevents the height of an airborne ramp jump from being
misclassified as a lateral excursion. Config has no implicit training defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class OffTrackConfig:
    lateral_limit_meters: float
    max_consecutive_steps: int

    def __post_init__(self) -> None:
        limit=self.lateral_limit_meters
        steps=self.max_consecutive_steps
        if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not isfinite(limit) or limit <= 0:
            raise ValueError("lateral_limit_meters must be finite positive")
        if isinstance(steps,bool) or not isinstance(steps,int) or steps <= 0:
            raise ValueError("max_consecutive_steps must be positive integer")


@dataclass(frozen=True, slots=True)
class OffTrackState:
    consecutive_steps: int=0
    outside: bool=False
    truncated: bool=False

    def __post_init__(self) -> None:
        if isinstance(self.consecutive_steps,bool) or not isinstance(self.consecutive_steps,int) or self.consecutive_steps < 0:
            raise ValueError("consecutive_steps must be nonnegative integer")
        if not isinstance(self.outside,bool) or not isinstance(self.truncated,bool):
            raise ValueError("outside and truncated must be boolean")

    def advance(self,lateral_meters: float, config: OffTrackConfig) -> OffTrackState:
        if not isinstance(config,OffTrackConfig):
            raise ValueError("config must be OffTrackConfig")
        if isinstance(lateral_meters,bool) or not isinstance(lateral_meters,(int,float)) or not isfinite(lateral_meters) or lateral_meters < 0:
            raise ValueError("lateral_meters must be finite non-negative")
        outside=lateral_meters > config.lateral_limit_meters
        steps=self.consecutive_steps + 1 if outside else 0
        return OffTrackState(steps,outside,steps >= config.max_consecutive_steps)
