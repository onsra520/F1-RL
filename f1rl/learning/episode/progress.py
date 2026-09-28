from __future__ import annotations

from dataclasses import dataclass

from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from f1rl.sandbox.track.state import ScorePickupCollected

from .route import RaceRoute, _validate_route_state


@dataclass(frozen=True, slots=True)
class RaceProgressState:
    next_checkpoint_index: int
    finished: bool

    def __post_init__(self) -> None:
        if isinstance(self.next_checkpoint_index, bool) or not isinstance(self.next_checkpoint_index, int) or self.next_checkpoint_index < 0:
            raise ValueError("next_checkpoint_index must be a non-negative integer")
        if not isinstance(self.finished, bool):
            raise ValueError("finished must be bool")


@dataclass(frozen=True, slots=True)
class RaceProgressResult:
    state: RaceProgressState
    target_position: Vec3 | None
    completed_checkpoint_id: str | None
    finished_now: bool

    def __post_init__(self) -> None:
        if not isinstance(self.state, RaceProgressState):
            raise ValueError("state must be RaceProgressState")
        if self.target_position is not None and not isinstance(self.target_position, Vec3):
            raise ValueError("target_position must be Vec3 or None")
        if self.completed_checkpoint_id is not None and (not isinstance(self.completed_checkpoint_id, str) or not self.completed_checkpoint_id.strip()):
            raise ValueError("completed_checkpoint_id must be a non-empty string or None")
        if not isinstance(self.finished_now, bool):
            raise ValueError("finished_now must be bool")


def advance_race_progress(
    route: RaceRoute,
    state: RaceProgressState,
    events: tuple[CheckpointEntered | FinishLineEntered | ObstacleHit | ScorePickupCollected, ...],
) -> RaceProgressResult:
    if not isinstance(route, RaceRoute):
        raise ValueError("route must be RaceRoute")
    if not isinstance(state, RaceProgressState):
        raise ValueError("state must be RaceProgressState")
    if not isinstance(events, tuple) or not all(isinstance(event, (CheckpointEntered, FinishLineEntered, ObstacleHit, ScorePickupCollected)) for event in events):
        raise ValueError("events must be a tuple of WorldEvent values")
    _validate_route_state(route, state)
    if state.finished:
        return RaceProgressResult(state, None, None, False)

    next_state = state
    completed_checkpoint_id = None
    if state.next_checkpoint_index < len(route.checkpoints):
        expected = route.checkpoints[state.next_checkpoint_index]
        if any(
            isinstance(event, CheckpointEntered)
            and event.checkpoint_id == expected.id
            and event.order == expected.order
            for event in events
        ):
            next_state = RaceProgressState(state.next_checkpoint_index + 1, False)
            completed_checkpoint_id = expected.id

    if next_state.next_checkpoint_index == len(route.checkpoints) and any(
        isinstance(event, FinishLineEntered) and event.finish_line_id == route.finish.id
        for event in events
    ):
        next_state = RaceProgressState(len(route.checkpoints), True)
        return RaceProgressResult(next_state, None, completed_checkpoint_id, True)

    return RaceProgressResult(next_state, route.target_for(next_state), completed_checkpoint_id, False)
