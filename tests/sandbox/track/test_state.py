import pytest

from f1rl.sandbox.entities.score_pickup import ScorePickup
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.state import TrackState, collect_score_pickups
from f1rl.sandbox.track.volume import OrientedBoxVolume


def pickup(identifier: str, points: int) -> ScorePickup:
    volume = OrientedBoxVolume(Vec3(0.0, 0.5, 0.0), Vec3(0.5, 0.5, 0.5), Quaternion.identity())
    return ScorePickup(identifier, volume, points)


def test_first_collection_emits_configured_points_and_new_state() -> None:
    state = TrackState(frozenset())

    next_state, events = collect_score_pickups((pickup("card", 100),), state)

    assert state.collected_pickup_ids == frozenset()
    assert next_state.collected_pickup_ids == frozenset({"card"})
    assert [(event.pickup_id, event.points) for event in events] == [("card", 100)]


def test_multiple_pickups_are_emitted_in_id_order() -> None:
    next_state, events = collect_score_pickups(
        (pickup("z-card", 50), pickup("a-card", 25)),
        TrackState(frozenset()),
    )

    assert [event.pickup_id for event in events] == ["a-card", "z-card"]
    assert next_state.collected_pickup_ids == frozenset({"a-card", "z-card"})


def test_repeat_collection_emits_nothing() -> None:
    state = TrackState(frozenset({"card"}))

    next_state, events = collect_score_pickups((pickup("card", 100),), state)

    assert next_state == state
    assert events == ()


def test_duplicate_pickup_ids_fail_explicitly() -> None:
    with pytest.raises(ValueError):
        collect_score_pickups(
            (pickup("card", 100), pickup("card", 200)),
            TrackState(frozenset()),
        )


def test_track_state_requires_frozen_valid_ids() -> None:
    with pytest.raises(ValueError):
        TrackState({"card"})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TrackState(frozenset({""}))
