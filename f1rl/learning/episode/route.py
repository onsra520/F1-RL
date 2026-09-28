from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld

if TYPE_CHECKING:
    from .progress import RaceProgressState


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RaceCheckpoint:
    id: str
    order: int
    target_position: Vec3

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")
        if not isinstance(self.target_position, Vec3):
            raise ValueError("target_position must be Vec3")


@dataclass(frozen=True, slots=True)
class RaceFinish:
    id: str
    target_position: Vec3

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not isinstance(self.target_position, Vec3):
            raise ValueError("target_position must be Vec3")


@dataclass(frozen=True, slots=True)
class RaceRoute:
    checkpoints: tuple[RaceCheckpoint, ...]
    finish: RaceFinish

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoints, tuple):
            raise ValueError("checkpoints must be a tuple")
        if not all(isinstance(checkpoint, RaceCheckpoint) for checkpoint in self.checkpoints):
            raise ValueError("checkpoints must contain RaceCheckpoint values")
        if not isinstance(self.finish, RaceFinish):
            raise ValueError("finish must be RaceFinish")

        checkpoints = tuple(sorted(self.checkpoints, key=lambda checkpoint: checkpoint.order))
        if tuple(checkpoint.order for checkpoint in checkpoints) != tuple(range(len(checkpoints))):
            raise ValueError("checkpoint orders must be contiguous starting at zero")
        checkpoint_ids = tuple(checkpoint.id for checkpoint in checkpoints)
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValueError("checkpoint ids must be unique")
        if self.finish.id in checkpoint_ids:
            raise ValueError("finish id must not match a checkpoint id")
        object.__setattr__(self, "checkpoints", checkpoints)

    @classmethod
    def from_track(cls, track: TrackWorld) -> RaceRoute:
        if not isinstance(track, TrackWorld):
            raise ValueError("track must be TrackWorld")
        if len(track.finish_lines) != 1:
            raise ValueError("track must contain exactly one finish line")
        return cls(
            tuple(
                RaceCheckpoint(checkpoint.id, checkpoint.order, checkpoint.volume.center)
                for checkpoint in track.checkpoints
            ),
            RaceFinish(track.finish_lines[0].id, track.finish_lines[0].volume.center),
        )

    def target_for(self, state: RaceProgressState) -> Vec3 | None:
        _validate_route_state(self, state)
        if state.finished:
            return None
        if state.next_checkpoint_index == len(self.checkpoints):
            return self.finish.target_position
        return self.checkpoints[state.next_checkpoint_index].target_position


def _validate_route_state(route: RaceRoute, state: object) -> None:
    try:
        next_checkpoint_index = state.next_checkpoint_index
        finished = state.finished
    except AttributeError as error:
        raise ValueError("state must provide race progress fields") from error
    if isinstance(next_checkpoint_index, bool) or not isinstance(next_checkpoint_index, int) or next_checkpoint_index < 0:
        raise ValueError("next_checkpoint_index must be a non-negative integer")
    if not isinstance(finished, bool):
        raise ValueError("finished must be bool")
    if next_checkpoint_index > len(route.checkpoints):
        raise ValueError("next_checkpoint_index exceeds checkpoint count")
    if finished and next_checkpoint_index != len(route.checkpoints):
        raise ValueError("finished state must complete all checkpoints")
