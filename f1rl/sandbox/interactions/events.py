from dataclasses import dataclass


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CheckpointEntered:
    checkpoint_id: str
    order: int

    def __post_init__(self) -> None:
        _validate_id(self.checkpoint_id, "checkpoint_id")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class FinishLineEntered:
    finish_line_id: str

    def __post_init__(self) -> None:
        _validate_id(self.finish_line_id, "finish_line_id")


@dataclass(frozen=True, slots=True)
class ObstacleHit:
    obstacle_id: str

    def __post_init__(self) -> None:
        _validate_id(self.obstacle_id, "obstacle_id")
