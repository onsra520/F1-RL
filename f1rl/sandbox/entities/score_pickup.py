from dataclasses import dataclass

from ..track.volume import OrientedBoxVolume
from ._validation import validate_id


@dataclass(frozen=True, slots=True)
class ScorePickup:
    id: str
    volume: OrientedBoxVolume
    points: int

    def __post_init__(self) -> None:
        validate_id(self.id)
        if isinstance(self.points, bool) or not isinstance(self.points, int) or self.points <= 0:
            raise ValueError("points must be a positive integer")
