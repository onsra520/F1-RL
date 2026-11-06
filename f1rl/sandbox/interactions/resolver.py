from __future__ import annotations

from ..dynamics.modifiers import MotionModifiers
from ..geometry.vector import Vec3
from ..track.query import TrackQuery
from ..vehicle import VehicleState
from .events import CheckpointEntered, FinishLineEntered, ObstacleHit
from .state import InteractionState


def resolve_entry_events(
    query: TrackQuery,
    previous_state: InteractionState,
) -> tuple[InteractionState, tuple[CheckpointEntered, ...], tuple[FinishLineEntered, ...]]:
    checkpoint_ids = frozenset(item.id for item in query.checkpoints)
    finish_ids = frozenset(item.id for item in query.finish_lines)

    checkpoint_events = tuple(
        CheckpointEntered(item.id, item.order)
        for item in sorted(query.checkpoints, key=lambda item: item.id)
        if item.id not in previous_state.active_checkpoint_ids
    )
    finish_events = tuple(
        FinishLineEntered(item.id)
        for item in sorted(query.finish_lines, key=lambda item: item.id)
        if item.id not in previous_state.active_finish_line_ids
    )
    return InteractionState(checkpoint_ids, finish_ids), checkpoint_events, finish_events


def resolve_motion_modifiers(query: TrackQuery) -> MotionModifiers:
    if query.contact is None:
        return MotionModifiers()

    acceleration_bonus = sum(
        item.acceleration_bonus for item in sorted(query.boost_pads, key=lambda item: item.id)
    )
    drag_multiplier = 1.0
    for item in sorted(query.slow_zones, key=lambda item: item.id):
        drag_multiplier *= item.drag_multiplier
    return MotionModifiers(acceleration_bonus, drag_multiplier)


def resolve_obstacle_block(
    previous: VehicleState,
    candidate: VehicleState,
    candidate_query: TrackQuery,
) -> tuple[VehicleState, tuple[ObstacleHit, ...]]:
    if not candidate_query.obstacles:
        return candidate, ()

    stopped = VehicleState(
        previous.position,
        Vec3(0.0, 0.0, 0.0),
        previous.orientation,
        Vec3(0.0, 0.0, 0.0),
        previous.grounded,
    )
    events = tuple(ObstacleHit(item.id) for item in sorted(candidate_query.obstacles, key=lambda item: item.id))
    return stopped, events
