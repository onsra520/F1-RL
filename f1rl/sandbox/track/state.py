from __future__ import annotations

from dataclasses import dataclass

from ..entities.score_pickup import ScorePickup


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pickup id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TrackState:
    collected_pickup_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.collected_pickup_ids, frozenset):
            raise ValueError("collected_pickup_ids must be a frozenset")
        for pickup_id in self.collected_pickup_ids:
            _validate_id(pickup_id)


@dataclass(frozen=True, slots=True)
class ScorePickupCollected:
    pickup_id: str
    points: int

    def __post_init__(self) -> None:
        _validate_id(self.pickup_id)
        if isinstance(self.points, bool) or not isinstance(self.points, int) or self.points <= 0:
            raise ValueError("points must be a positive integer")


def collect_score_pickups(
    pickups: tuple[ScorePickup, ...],
    state: TrackState,
) -> tuple[TrackState, tuple[ScorePickupCollected, ...]]:
    pickup_ids = [pickup.id for pickup in pickups]
    if len(pickup_ids) != len(set(pickup_ids)):
        raise ValueError("pickup ids must be unique")

    collected = set(state.collected_pickup_ids)
    events: list[ScorePickupCollected] = []
    for pickup in sorted(pickups, key=lambda item: item.id):
        if pickup.id in collected:
            continue
        collected.add(pickup.id)
        events.append(ScorePickupCollected(pickup.id, pickup.points))

    return TrackState(frozenset(collected)), tuple(events)
