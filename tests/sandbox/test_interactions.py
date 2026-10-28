import pytest

from src.sandbox.entities.checkpoint import Checkpoint
from src.sandbox.entities.finish_line import FinishLine
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from src.sandbox.interactions.resolver import resolve_entry_events
from src.sandbox.interactions.state import InteractionState
from src.sandbox.track.query import TrackQuery
from src.sandbox.track.volume import OrientedBoxVolume


def volume() -> OrientedBoxVolume:
    return OrientedBoxVolume(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), Quaternion.identity())


def query(*, checkpoints=(), finish_lines=()) -> TrackQuery:
    return TrackQuery(None, (), (), (), tuple(checkpoints), tuple(finish_lines), ())


def empty_state() -> InteractionState:
    return InteractionState(frozenset(), frozenset())


def test_interaction_state_and_events_validate_values() -> None:
    with pytest.raises(ValueError):
        InteractionState(set(), frozenset())
    with pytest.raises(ValueError):
        InteractionState(frozenset({""}), frozenset())
    with pytest.raises(ValueError):
        CheckpointEntered("cp", -1)
    with pytest.raises(ValueError):
        FinishLineEntered("")
    with pytest.raises(ValueError):
        ObstacleHit("")


def test_first_entry_emits_checkpoint_and_finish_events() -> None:
    q = query(checkpoints=(Checkpoint("cp", volume(), 3),), finish_lines=(FinishLine("finish", volume()),))
    next_state, checkpoint_events, finish_events = resolve_entry_events(q, empty_state())
    assert checkpoint_events == (CheckpointEntered("cp", 3),)
    assert finish_events == (FinishLineEntered("finish"),)
    assert next_state == InteractionState(frozenset({"cp"}), frozenset({"finish"}))


def test_staying_inside_does_not_repeat_event_and_leaving_clears_membership() -> None:
    cp = Checkpoint("cp", volume(), 1)
    inside = query(checkpoints=(cp,))
    active, _, _ = resolve_entry_events(inside, empty_state())
    stayed, checkpoint_events, finish_events = resolve_entry_events(inside, active)
    assert checkpoint_events == ()
    assert finish_events == ()
    assert stayed == active

    left, checkpoint_events, finish_events = resolve_entry_events(query(), stayed)
    assert left == empty_state()
    assert checkpoint_events == ()
    assert finish_events == ()


def test_reentry_emits_again_and_simultaneous_ids_are_sorted() -> None:
    a = Checkpoint("a", volume(), 7)
    b = Checkpoint("b", volume(), 2)
    active, events, _ = resolve_entry_events(query(checkpoints=(b, a)), empty_state())
    assert events == (CheckpointEntered("a", 7), CheckpointEntered("b", 2))
    left, _, _ = resolve_entry_events(query(), active)
    _, events, _ = resolve_entry_events(query(checkpoints=(b, a)), left)
    assert events == (CheckpointEntered("a", 7), CheckpointEntered("b", 2))
