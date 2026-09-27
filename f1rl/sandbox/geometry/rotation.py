from __future__ import annotations

from dataclasses import dataclass
from math import cos, isclose, isfinite, sin, sqrt

from .vector import Vec3


@dataclass(frozen=True, slots=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.w, self.x, self.y, self.z)):
            raise ValueError("quaternion components must be finite")
        norm = sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if norm == 0.0:
            raise ValueError("quaternion norm must be non-zero")
        object.__setattr__(self, "w", self.w / norm)
        object.__setattr__(self, "x", self.x / norm)
        object.__setattr__(self, "y", self.y / norm)
        object.__setattr__(self, "z", self.z / norm)

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(1.0, 0.0, 0.0, 0.0)

    def normalized(self) -> Quaternion:
        return Quaternion(self.w, self.x, self.y, self.z)

    def __mul__(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle: float) -> Quaternion:
        unit_axis = axis.normalized()
        half_angle = angle / 2.0
        scale = sin(half_angle)
        return cls(cos(half_angle), unit_axis.x * scale, unit_axis.y * scale, unit_axis.z * scale)

    def rotate(self, vector: Vec3) -> Vec3:
        axis = Vec3(self.x, self.y, self.z)
        axis_dot = axis.dot(vector)
        axis_norm_sq = axis.dot(axis)
        return (
            axis * (2.0 * axis_dot)
            + vector * (self.w * self.w - axis_norm_sq)
            + axis.cross(vector) * (2.0 * self.w)
        )

    @classmethod
    def from_basis(cls, right: Vec3, up: Vec3, forward: Vec3) -> Quaternion:
        r = right.normalized()
        u = up.normalized()
        f = forward.normalized()
        if not (
            isclose(r.dot(u), 0.0, abs_tol=1e-9)
            and isclose(r.dot(f), 0.0, abs_tol=1e-9)
            and isclose(u.dot(f), 0.0, abs_tol=1e-9)
        ):
            raise ValueError("basis vectors must be orthogonal")
        if not isclose(r.cross(u).dot(f), 1.0, abs_tol=1e-9):
            raise ValueError("basis vectors must be right-handed")

        m00, m01, m02 = r.x, u.x, f.x
        m10, m11, m12 = r.y, u.y, f.y
        m20, m21, m22 = r.z, u.z, f.z
        trace = m00 + m11 + m22

        if trace > 0.0:
            s = sqrt(trace + 1.0) * 2.0
            return cls(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
        if m00 > m11 and m00 > m22:
            s = sqrt(1.0 + m00 - m11 - m22) * 2.0
            return cls((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
        if m11 > m22:
            s = sqrt(1.0 + m11 - m00 - m22) * 2.0
            return cls((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
        s = sqrt(1.0 + m22 - m00 - m11) * 2.0
        return cls((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
