from dataclasses import dataclass
from math import isfinite

from ..track.volume import OrientedBoxVolume
from ._validation import validate_id


@dataclass(frozen=True, slots=True)
class SlowZone:
    id: str
    volume: OrientedBoxVolume
    drag_multiplier: float

    def __post_init__(self) -> None:
        validate_id(self.id)
        if isinstance(self.drag_multiplier, bool) or not isinstance(self.drag_multiplier, (int, float)):
            raise ValueError("drag_multiplier must be numeric")
        if not isfinite(self.drag_multiplier) or self.drag_multiplier < 1.0:
            raise ValueError("drag_multiplier must be finite and at least 1.0")
