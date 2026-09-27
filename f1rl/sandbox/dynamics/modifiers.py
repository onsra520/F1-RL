from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class MotionModifiers:
    longitudinal_acceleration_bonus: float = 0.0
    rolling_drag_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.longitudinal_acceleration_bonus) or self.longitudinal_acceleration_bonus < 0.0:
            raise ValueError("longitudinal_acceleration_bonus must be finite and non-negative")
        if not isfinite(self.rolling_drag_multiplier) or self.rolling_drag_multiplier < 1.0:
            raise ValueError("rolling_drag_multiplier must be finite and at least 1.0")
