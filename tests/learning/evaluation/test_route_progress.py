import math

import pytest

from f1rl.learning.episode import RaceCheckpoint, RaceFinish, RaceRoute
from f1rl.learning.evaluation.metrics import _normalized_route_progress
from f1rl.sandbox.geometry.vector import Vec3


def _straight() -> RaceRoute:
    return RaceRoute((), RaceFinish("finish", Vec3(0.0, 0.0, 10.0)))


def _multi() -> RaceRoute:
    return RaceRoute(
        (
            RaceCheckpoint("cp-0", 0, Vec3(0.0, 0.0, 10.0)),
            RaceCheckpoint("cp-1", 1, Vec3(10.0, 0.0, 10.0)),
        ),
        RaceFinish("finish", Vec3(10.0, 0.0, 20.0)),
    )


@pytest.mark.parametrize(
    ("z", "finished", "expected"),
    [
        (0.0, False, 0.0),
        (5.0, False, 0.5),
        (10.0, False, 1.0),
        (-5.0, False, 0.0),
        (15.0, False, 1.0),
        (2.0, True, 1.0),
    ],
)
def test_zero_checkpoint_route_progress_clamps_current_segment(
    z: float,
    finished: bool,
    expected: float,
) -> None:
    assert _normalized_route_progress(
        _straight(),
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(0.0, 0.0, z),
        next_checkpoint_index=0,
        finished=finished,
    ) == pytest.approx(expected)


def test_multi_checkpoint_progress_uses_only_current_required_segment() -> None:
    route = _multi()

    assert _normalized_route_progress(
        route,
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(0.0, 0.0, 5.0),
        next_checkpoint_index=0,
        finished=False,
    ) == pytest.approx(5.0 / 30.0)

    assert _normalized_route_progress(
        route,
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(5.0, 0.0, 10.0),
        next_checkpoint_index=1,
        finished=False,
    ) == pytest.approx(15.0 / 30.0)

    assert _normalized_route_progress(
        route,
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(10.0, 0.0, 15.0),
        next_checkpoint_index=2,
        finished=False,
    ) == pytest.approx(25.0 / 30.0)


def test_later_geometry_cannot_bypass_ordered_checkpoint_progress() -> None:
    route = _multi()

    progress = _normalized_route_progress(
        route,
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(10.0, 0.0, 19.0),
        next_checkpoint_index=0,
        finished=False,
    )

    assert progress == pytest.approx(10.0 / 30.0)


def test_lateral_offset_does_not_inflate_segment_projection() -> None:
    route = _multi()
    on_segment = _normalized_route_progress(
        route,
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(5.0, 0.0, 10.0),
        next_checkpoint_index=1,
        finished=False,
    )
    laterally_offset = _normalized_route_progress(
        route,
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(5.0, 100.0, 10.0),
        next_checkpoint_index=1,
        finished=False,
    )
    assert laterally_offset == pytest.approx(on_segment)


def test_route_progress_uses_full_3d_projection() -> None:
    route = RaceRoute((), RaceFinish("finish", Vec3(0.0, 6.0, 8.0)))

    progress = _normalized_route_progress(
        route,
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(0.0, 3.0, 4.0),
        next_checkpoint_index=0,
        finished=False,
    )

    assert progress == pytest.approx(0.5)


def test_route_progress_rejects_zero_length_route() -> None:
    route = RaceRoute((), RaceFinish("finish", Vec3(0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="length"):
        _normalized_route_progress(
            route,
            initial_position=(0.0, 0.0, 0.0),
            vehicle_position=(0.0, 0.0, 0.0),
            next_checkpoint_index=0,
            finished=False,
        )


@pytest.mark.parametrize(
    ("position", "next_index"),
    [
        ((0.0, 0.0), 0),
        ((0.0, 0.0, float("nan")), 0),
        ((0.0, 0.0, float("inf")), 0),
        ((0.0, 0.0, 0.0), -1),
        ((0.0, 0.0, 0.0), 3),
    ],
)
def test_route_progress_rejects_invalid_inputs(
    position: tuple[float, ...],
    next_index: int,
) -> None:
    with pytest.raises(ValueError):
        _normalized_route_progress(
            _multi(),
            initial_position=(0.0, 0.0, 0.0),
            vehicle_position=position,  # type: ignore[arg-type]
            next_checkpoint_index=next_index,
            finished=False,
        )


def test_route_progress_is_always_finite_and_bounded() -> None:
    value = _normalized_route_progress(
        _multi(),
        initial_position=(0.0, 0.0, 0.0),
        vehicle_position=(1e6, 1e6, 1e6),
        next_checkpoint_index=1,
        finished=False,
    )
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
