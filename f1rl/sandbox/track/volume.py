from __future__ import annotations

from dataclasses import dataclass

from ..geometry.rotation import Quaternion
from ..geometry.vector import Vec3

_LOCAL_RIGHT = Vec3(1.0, 0.0, 0.0)
_LOCAL_UP = Vec3(0.0, 1.0, 0.0)
_LOCAL_FORWARD = Vec3(0.0, 0.0, 1.0)
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class OrientedBoxVolume:
    center: Vec3
    half_extents: Vec3
    orientation: Quaternion

    def __post_init__(self) -> None:
        if self.half_extents.x <= 0.0 or self.half_extents.y <= 0.0 or self.half_extents.z <= 0.0:
            raise ValueError("half_extents must be strictly positive")

    def contains(self, point: Vec3) -> bool:
        offset = point - self.center
        right = self.orientation.rotate(_LOCAL_RIGHT)
        up = self.orientation.rotate(_LOCAL_UP)
        forward = self.orientation.rotate(_LOCAL_FORWARD)

        local_x = offset.dot(right)
        local_y = offset.dot(up)
        local_z = offset.dot(forward)

        return (
            abs(local_x) <= self.half_extents.x + _EPSILON
            and abs(local_y) <= self.half_extents.y + _EPSILON
            and abs(local_z) <= self.half_extents.z + _EPSILON
        )
