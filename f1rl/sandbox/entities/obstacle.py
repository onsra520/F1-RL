from dataclasses import dataclass

from ..track.volume import OrientedBoxVolume
from ._validation import validate_id


@dataclass(frozen=True, slots=True)
class Obstacle:
    id: str
    volume: OrientedBoxVolume

    def __post_init__(self) -> None:
        validate_id(self.id)
