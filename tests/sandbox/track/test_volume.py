from dataclasses import FrozenInstanceError
from math import pi

import pytest

from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.track.volume import OrientedBoxVolume


def test_identity_box_contains_interior_boundary_and_excludes_outside() -> None:
    volume = OrientedBoxVolume(
        center=Vec3(1.0, 2.0, 3.0),
        half_extents=Vec3(2.0, 1.0, 0.5),
        orientation=Quaternion.identity(),
    )

    assert volume.contains(Vec3(1.5, 2.0, 3.0))
    assert volume.contains(Vec3(3.0, 3.0, 3.5))
    assert not volume.contains(Vec3(3.000001, 2.0, 3.0))


def test_rotated_box_uses_local_axes() -> None:
    rotation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 2.0)
    volume = OrientedBoxVolume(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 1.0, 0.5), rotation)

    assert volume.contains(rotation.rotate(Vec3(1.9, 0.0, 0.0)))
    assert not volume.contains(rotation.rotate(Vec3(2.1, 0.0, 0.0)))


def test_volume_rejects_non_positive_half_extents() -> None:
    with pytest.raises(ValueError):
        OrientedBoxVolume(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 1.0), Quaternion.identity())

    with pytest.raises(ValueError):
        OrientedBoxVolume(Vec3(0.0, 0.0, 0.0), Vec3(1.0, -1.0, 1.0), Quaternion.identity())


def test_volume_is_immutable() -> None:
    volume = OrientedBoxVolume(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), Quaternion.identity())

    with pytest.raises(FrozenInstanceError):
        volume.center = Vec3(2.0, 0.0, 0.0)  # type: ignore[misc]
