from dataclasses import dataclass

from ..track.state import ScorePickupCollected


@dataclass(frozen=True, slots=True)
class ScoreState:
    points: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.points, bool) or not isinstance(self.points, int) or self.points < 0:
            raise ValueError("points must be a non-negative integer")


def apply_score_events(
    score: ScoreState,
    events: tuple[ScorePickupCollected, ...],
) -> ScoreState:
    return ScoreState(score.points + sum(event.points for event in events))
