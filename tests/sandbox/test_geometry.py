from math import pi, sqrt

import pytest

from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3


def test_vec3_arithmetic_and_scalar_operations() -> None:
    a = Vec3(1.0, 2.0, 3.0)
    b = Vec3(4.0, -1.0, 2.0)

    assert a + b == Vec3(5.0, 1.0, 5.0)
    assert a - b == Vec3(-3.0, 3.0, 1.0)
    assert a * 2.0 == Vec3(2.0, 4.0, 6.0)
    assert 2.0 * a == Vec3(2.0, 4.0, 6.0)
    assert a / 2.0 == Vec3(0.5, 1.0, 1.5)


def test_vec3_dot_cross_and_magnitude() -> None:
    x = Vec3(1.0, 0.0, 0.0)
    y = Vec3(0.0, 1.0, 0.0)

    assert x.dot(y) == 0.0
    assert x.cross(y) == Vec3(0.0, 0.0, 1.0)
    assert Vec3(2.0, 3.0, 6.0).magnitude == pytest.approx(7.0)


def test_vec3_normalization_projection_and_rejection() -> None:
    value = Vec3(3.0, 4.0, 0.0)
    normal = Vec3(0.0, 2.0, 0.0)

    assert value.normalized().magnitude == pytest.approx(1.0)
    assert value.project_onto(normal) == Vec3(0.0, 4.0, 0.0)
    assert value.reject_from(normal) == Vec3(3.0, 0.0, 0.0)


def test_vec3_rejects_undefined_operations() -> None:
    with pytest.raises(ValueError):
        Vec3(0.0, 0.0, 0.0).normalized()
    with pytest.raises(ValueError):
        Vec3(1.0, 0.0, 0.0).project_onto(Vec3(0.0, 0.0, 0.0))
    with pytest.raises(ZeroDivisionError):
        Vec3(1.0, 0.0, 0.0) / 0.0


def test_quaternion_identity_and_normalization() -> None:
    identity = Quaternion.identity()
    value = Quaternion(2.0, 0.0, 0.0, 0.0).normalized()

    assert identity == Quaternion(1.0, 0.0, 0.0, 0.0)
    assert value == identity


def test_quaternion_axis_angle_rotation_and_multiplication() -> None:
    quarter_turn_y = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), pi / 2.0)
    rotated = quarter_turn_y.rotate(Vec3(0.0, 0.0, 1.0))

    assert rotated.x == pytest.approx(1.0, abs=1e-9)
    assert rotated.y == pytest.approx(0.0, abs=1e-9)
    assert rotated.z == pytest.approx(0.0, abs=1e-9)
    assert (Quaternion.identity() * quarter_turn_y) == quarter_turn_y


def test_quaternion_from_basis_preserves_vehicle_axes() -> None:
    right = Vec3(1.0, 0.0, 0.0)
    up = Vec3(0.0, 1.0, 0.0)
    forward = Vec3(0.0, 0.0, 1.0)

    rotation = Quaternion.from_basis(right, up, forward)

    assert rotation.rotate(right) == right
    assert rotation.rotate(up) == up
    assert rotation.rotate(forward) == forward


def test_quaternion_rejects_zero_norm_or_zero_axis() -> None:
    with pytest.raises(ValueError):
        Quaternion(0.0, 0.0, 0.0, 0.0).normalized()
    with pytest.raises(ValueError):
        Quaternion.from_axis_angle(Vec3(0.0, 0.0, 0.0), pi / 2.0)


def test_geometry_rejects_non_finite_components() -> None:
    with pytest.raises(ValueError):
        Vec3(float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError):
        Vec3(0.0, float("inf"), 0.0)
    with pytest.raises(ValueError):
        Quaternion(float("nan"), 0.0, 0.0, 1.0)


def test_quaternion_rejects_left_handed_basis() -> None:
    with pytest.raises(ValueError):
        Quaternion.from_basis(
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, -1.0),
        )
