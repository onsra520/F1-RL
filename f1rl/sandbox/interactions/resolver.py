from __future__ import annotations

from ..track.query import TrackQuery
from .events import CheckpointEntered, FinishLineEntered
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
