from math import pi, sqrt

import pytest

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.volume import OrientedBoxVolume


def _box(*, orientation: Quaternion | None = None) -> OrientedBoxVolume:
    return OrientedBoxVolume(
        center=Vec3(0.0, 0.0, 0.0),
        half_extents=Vec3(1.0, 1.0, 1.0),
        orientation=orientation or Quaternion.identity(),
    )


def test_axis_aligned_ray_returns_exact_entry_distance() -> None:
    distance = _box().ray_distance(
        Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 10.0
    )
    assert distance == pytest.approx(4.0)


def test_rotated_box_uses_oriented_local_slabs() -> None:
    box = OrientedBoxVolume(
        Vec3(0.0, 0.0, 0.0),
        Vec3(2.0, 1.0, 1.0),        Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 4.0),
    )
    distance = box.ray_distance(
        Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 10.0
    )
    assert distance == pytest.approx(5.0 - sqrt(2.0))


def test_origin_inside_or_on_boundary_returns_zero() -> None:
    box = _box()
    assert box.ray_distance(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 5.0) == 0.0
    assert box.ray_distance(Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 5.0) == 0.0


def test_miss_and_tangential_boundary_hit() -> None:
    box = _box()
    assert box.ray_distance(Vec3(2.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 10.0) is None
    assert box.ray_distance(
        Vec3(1.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 10.0
    ) == pytest.approx(4.0)


def test_hit_at_max_distance_is_included_but_beyond_range_is_not() -> None:
    box = _box()
    origin = Vec3(0.0, 0.0, -5.0)
    direction = Vec3(0.0, 0.0, 1.0)
    assert box.ray_distance(origin, direction, 4.0) == pytest.approx(4.0)
    assert box.ray_distance(origin, direction, 3.999) is None

def test_parallel_axis_hit_and_miss_are_handled_without_division() -> None:
    box = _box()
    direction = Vec3(0.0, 0.0, 1.0)
    assert box.ray_distance(Vec3(0.5, 0.0, -5.0), direction, 10.0) == pytest.approx(4.0)
    assert box.ray_distance(Vec3(1.5, 0.0, -5.0), direction, 10.0) is None


@pytest.mark.parametrize(
    "direction",
    [Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 2.0), Vec3(0.0, 0.0, 1.0 + 2e-9)],
)
def test_direction_must_be_unit_length(direction: Vec3) -> None:
    with pytest.raises(ValueError):
        _box().ray_distance(Vec3(0.0, 0.0, -5.0), direction, 10.0)


def test_direction_within_absolute_tolerance_is_accepted() -> None:
    distance = _box().ray_distance(
        Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0 + 5e-10), 10.0
    )
    assert distance is not None


@pytest.mark.parametrize("max_distance", [-1.0, float("nan"), float("inf"), True])
def test_max_distance_rejects_invalid_values(max_distance: object) -> None:
    with pytest.raises(ValueError):
        _box().ray_distance(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), max_distance)  # type: ignore[arg-type]


def test_zero_max_distance_is_valid() -> None:
    box = _box()
    assert box.ray_distance(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.0) == 0.0
    assert box.ray_distance(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 0.0) is None