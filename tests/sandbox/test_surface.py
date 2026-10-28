from dataclasses import FrozenInstanceError

import pytest

from src.sandbox.geometry.vector import Vec3
from src.sandbox.surface import SurfaceContact


def test_surface_contact_accepts_unit_normal_and_is_frozen() -> None:
    contact = SurfaceContact(point=Vec3(1.0, 2.0, 3.0), normal=Vec3(0.0, 1.0, 0.0))

    assert contact.point == Vec3(1.0, 2.0, 3.0)
    assert contact.normal == Vec3(0.0, 1.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        contact.normal = Vec3(1.0, 0.0, 0.0)


def test_surface_contact_requires_unit_normal() -> None:
    with pytest.raises(ValueError):
        SurfaceContact(point=Vec3(0.0, 0.0, 0.0), normal=Vec3(0.0, 2.0, 0.0))
    with pytest.raises(ValueError):
        SurfaceContact(point=Vec3(0.0, 0.0, 0.0), normal=Vec3(0.0, 0.0, 0.0))
