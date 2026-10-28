from dataclasses import dataclass

from ..track.volume import OrientedBoxVolume
from ._validation import validate_id


@dataclass(frozen=True, slots=True)
class Checkpoint:
    id: str
    volume: OrientedBoxVolume
    order: int

    def __post_init__(self) -> None:
        validate_id(self.id)
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")
