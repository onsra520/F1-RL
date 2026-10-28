from dataclasses import dataclass

from ..track.volume import OrientedBoxVolume
from ._validation import validate_id, validate_positive_number


@dataclass(frozen=True, slots=True)
class BoostPad:
    id: str
    volume: OrientedBoxVolume
    acceleration_bonus: float

    def __post_init__(self) -> None:
        validate_id(self.id)
        validate_positive_number(self.acceleration_bonus, "acceleration_bonus")
