from dataclasses import FrozenInstanceError, dataclass

import pytest

from f1rl.learning.episode import RaceCheckpoint, RaceFinish, RaceRoute
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.volume import OrientedBoxVolume


@dataclass(frozen=True)
class _ProgressState:
    next_checkpoint_index: int
    finished: bool


def _volume(center: Vec3) -> OrientedBoxVolume:
    return OrientedBoxVolume(center, Vec3(1.0, 1.0, 1.0), Quaternion.identity())


def _checkpoint(id: str, order: int, x: float) -> RaceCheckpoint:
    return RaceCheckpoint(id, order, Vec3(x, 0.0, 0.0))


def _route() -> RaceRoute:
    return RaceRoute((_checkpoint("cp-0", 0, 1.0), _checkpoint("cp-1", 1, 2.0)), RaceFinish("finish", Vec3(3.0, 0.0, 0.0)))


@pytest.mark.parametrize("id", ["", "   ", 1, None])
def test_route_value_objects_reject_invalid_ids(id: object) -> None:
    with pytest.raises(ValueError):
        RaceCheckpoint(id, 0, Vec3(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        RaceFinish(id, Vec3(0.0, 0.0, 0.0))


@pytest.mark.parametrize("order", [True, -1, 1.0, "0"])
def test_checkpoint_rejects_non_nonnegative_integer_order(order: object) -> None:
    with pytest.raises(ValueError):
        RaceCheckpoint("cp", order, Vec3(0.0, 0.0, 0.0))


@pytest.mark.parametrize("target", [None, (0.0, 0.0, 0.0), "target"])
def test_route_value_objects_require_vec3_targets(target: object) -> None:
    with pytest.raises(ValueError):
        RaceCheckpoint("cp", 0, target)
    with pytest.raises(ValueError):
        RaceFinish("finish", target)


def test_route_value_objects_are_immutable() -> None:
    checkpoint = _checkpoint("cp", 0, 1.0)
    finish = RaceFinish("finish", Vec3(2.0, 0.0, 0.0))

    with pytest.raises(FrozenInstanceError):
        checkpoint.id = "other"
    with pytest.raises(FrozenInstanceError):
        finish.id = "other"


def test_route_canonicalizes_checkpoints_by_numeric_order_not_id() -> None:
    route = RaceRoute((_checkpoint("z-first", 0, 1.0), _checkpoint("a-second", 1, 2.0)), RaceFinish("finish", Vec3(3.0, 0.0, 0.0)))

    assert tuple(checkpoint.id for checkpoint in route.checkpoints) == ("z-first", "a-second")


@pytest.mark.parametrize(
    "checkpoints, finish",
    [
        ([_checkpoint("cp", 0, 1.0)], RaceFinish("finish", Vec3(2.0, 0.0, 0.0))),
        ((_checkpoint("cp", 0, 1.0),), "finish"),
        ((_checkpoint("a", 0, 1.0), _checkpoint("b", 0, 2.0)), RaceFinish("finish", Vec3(3.0, 0.0, 0.0))),
        ((_checkpoint("a", 0, 1.0), _checkpoint("b", 2, 2.0)), RaceFinish("finish", Vec3(3.0, 0.0, 0.0))),
        ((_checkpoint("same", 0, 1.0), _checkpoint("same", 1, 2.0)), RaceFinish("finish", Vec3(3.0, 0.0, 0.0))),
        ((_checkpoint("same", 0, 1.0),), RaceFinish("same", Vec3(2.0, 0.0, 0.0))),
    ],
)
def test_route_rejects_invalid_definition(checkpoints: object, finish: object) -> None:
    with pytest.raises(ValueError):
        RaceRoute(checkpoints, finish)


def test_from_track_uses_order_and_volume_centers() -> None:
    first_center = Vec3(10.0, 0.0, 0.0)
    second_center = Vec3(20.0, 0.0, 0.0)
    finish_center = Vec3(30.0, 0.0, 0.0)
    track = TrackWorld(
        checkpoints=(Checkpoint("a-second", _volume(second_center), 1), Checkpoint("z-first", _volume(first_center), 0)),
        finish_lines=(FinishLine("finish", _volume(finish_center)),),
    )

    route = RaceRoute.from_track(track)

    assert tuple((checkpoint.id, checkpoint.target_position) for checkpoint in route.checkpoints) == (("z-first", first_center), ("a-second", second_center))
    assert route.finish.target_position == finish_center


def test_from_track_allows_no_checkpoints() -> None:
    finish_center = Vec3(30.0, 0.0, 0.0)

    route = RaceRoute.from_track(TrackWorld(finish_lines=(FinishLine("finish", _volume(finish_center)),)))

    assert route.checkpoints == ()
    assert route.finish.target_position == finish_center


@pytest.mark.parametrize(
    "finish_lines",
    [(), (FinishLine("finish-a", _volume(Vec3(1.0, 0.0, 0.0))), FinishLine("finish-b", _volume(Vec3(2.0, 0.0, 0.0))))],
)
def test_from_track_requires_exactly_one_finish_line(finish_lines: tuple[FinishLine, ...]) -> None:
    with pytest.raises(ValueError):
        RaceRoute.from_track(TrackWorld(finish_lines=finish_lines))


def test_target_for_selects_next_checkpoint_then_finish_then_none() -> None:
    route = _route()

    assert route.target_for(_ProgressState(0, False)) == Vec3(1.0, 0.0, 0.0)
    assert route.target_for(_ProgressState(1, False)) == Vec3(2.0, 0.0, 0.0)
    assert route.target_for(_ProgressState(2, False)) == Vec3(3.0, 0.0, 0.0)
    assert route.target_for(_ProgressState(2, True)) is None


def test_zero_checkpoint_route_targets_finish_immediately() -> None:
    finish = RaceFinish("finish", Vec3(3.0, 0.0, 0.0))
    route = RaceRoute((), finish)

    assert route.target_for(_ProgressState(0, False)) == finish.target_position


@pytest.mark.parametrize("state", [_ProgressState(3, False), _ProgressState(1, True)])
def test_target_for_rejects_route_incompatible_state(state: _ProgressState) -> None:
    with pytest.raises(ValueError):
        _route().target_for(state)
