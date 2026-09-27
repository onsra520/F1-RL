from math import cos, pi, sin, sqrt

import pytest

from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.patch import PlanarSurfacePatch


def flat_patch() -> PlanarSurfacePatch:
    return PlanarSurfacePatch(
        id="flat",
        origin=Vec3(0.0, 0.0, 0.0),
        forward=Vec3(0.0, 0.0, 1.0),
        normal=Vec3(0.0, 1.0, 0.0),
        width=4.0,
        length=10.0,
    )


def test_flat_patch_projects_position_and_includes_edges() -> None:
    patch = flat_patch()

    center = patch.sample(Vec3(1.0, 0.25, 2.0), max_distance=0.25)
    edge = patch.sample(Vec3(2.0, 0.0, 5.0), max_distance=0.0)

    assert center is not None
    assert center.point == Vec3(1.0, 0.0, 2.0)
    assert center.normal == Vec3(0.0, 1.0, 0.0)
    assert edge is not None


def test_patch_rejects_outside_bounds_and_excess_distance() -> None:
    patch = flat_patch()

    assert patch.sample(Vec3(2.01, 0.0, 0.0), max_distance=1.0) is None
    assert patch.sample(Vec3(0.0, 0.0, 5.01), max_distance=1.0) is None
    assert patch.sample(Vec3(0.0, 0.101, 0.0), max_distance=0.1) is None


def test_sloped_patch_projects_to_ramp_plane() -> None:
    angle = pi / 6.0
    forward = Vec3(0.0, sin(angle), cos(angle))
    normal = Vec3(0.0, cos(angle), -sin(angle))
    patch = PlanarSurfacePatch("ramp", Vec3(0.0, 1.0, 0.0), forward, normal, 4.0, 8.0)
    expected = patch.origin + forward * 2.0

    contact = patch.sample(expected + normal * 0.4, max_distance=0.4)

    assert contact is not None
    assert contact.point.x == pytest.approx(expected.x)
    assert contact.point.y == pytest.approx(expected.y)
    assert contact.point.z == pytest.approx(expected.z)
    assert contact.normal == normal


def test_patch_validates_id_axes_dimensions_and_distance() -> None:
    with pytest.raises(ValueError):
        PlanarSurfacePatch("", Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0), 1.0, 1.0)
    with pytest.raises(ValueError):
        PlanarSurfacePatch("bad", Vec3(0, 0, 0), Vec3(0, 0, 2), Vec3(0, 1, 0), 1.0, 1.0)
    with pytest.raises(ValueError):
        PlanarSurfacePatch("bad", Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, 1), 1.0, 1.0)
    with pytest.raises(ValueError):
        PlanarSurfacePatch("bad", Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0), 0.0, 1.0)
    with pytest.raises(ValueError):
        flat_patch().sample(Vec3(0, 0, 0), max_distance=-0.1)


def test_patch_sampling_is_deterministic() -> None:
    patch = flat_patch()
    point = Vec3(0.5, 0.1, -2.0)

    assert patch.sample(point, 0.1) == patch.sample(point, 0.1)
