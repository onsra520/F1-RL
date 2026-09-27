from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("Vec3 components must be finite")

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vec3:
        if scalar == 0.0:
            raise ZeroDivisionError("cannot divide Vec3 by zero")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    @property
    def magnitude(self) -> float:
        return sqrt(self.dot(self))

    def normalized(self) -> Vec3:
        magnitude = self.magnitude
        if magnitude == 0.0:
            raise ValueError("cannot normalize zero-length vector")
        return self / magnitude

    def project_onto(self, normal: Vec3) -> Vec3:
        denominator = normal.dot(normal)
        if denominator == 0.0:
            raise ValueError("cannot project onto zero-length vector")
        return normal * (self.dot(normal) / denominator)

    def reject_from(self, normal: Vec3) -> Vec3:
        return self - self.project_onto(normal)
