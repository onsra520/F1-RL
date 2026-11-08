from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..geometry.rotation import Quaternion
from ..geometry.vector import Vec3

_LOCAL_RIGHT = Vec3(1.0, 0.0, 0.0)
_LOCAL_UP = Vec3(0.0, 1.0, 0.0)
_LOCAL_FORWARD = Vec3(0.0, 0.0, 1.0)
_EPSILON = 1e-12
_UNIT_TOLERANCE = 1e-9


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

    def ray_distance(self, origin: Vec3, direction: Vec3, max_distance: float) -> float | None:
        if isinstance(max_distance, bool) or not isinstance(max_distance, (int, float)):
            raise ValueError("max_distance must be a finite non-negative number")
        if not isfinite(max_distance) or max_distance < 0.0:
            raise ValueError("max_distance must be a finite non-negative number")
        if abs(direction.magnitude - 1.0) > _UNIT_TOLERANCE:
            raise ValueError("direction must be unit length")
        if self.contains(origin):
            return 0.0

        offset = origin - self.center
        axes = (
            self.orientation.rotate(_LOCAL_RIGHT),
            self.orientation.rotate(_LOCAL_UP),
            self.orientation.rotate(_LOCAL_FORWARD),
        )
        local_origin = tuple(offset.dot(axis) for axis in axes)
        local_direction = tuple(direction.dot(axis) for axis in axes)
        half_extents = (self.half_extents.x, self.half_extents.y, self.half_extents.z)

        t_min = 0.0
        t_max = float(max_distance)
        for origin_component, direction_component, half_extent in zip(
            local_origin, local_direction, half_extents, strict=True
        ):
            if abs(direction_component) <= _EPSILON:
                if abs(origin_component) > half_extent + _EPSILON:
                    return None
                continue

            first = (-half_extent - origin_component) / direction_component
            second = (half_extent - origin_component) / direction_component
            entry, exit_ = (first, second) if first <= second else (second, first)
            t_min = max(t_min, entry)
            t_max = min(t_max, exit_)
            if t_min > t_max + _EPSILON:
                return None

        if t_min > max_distance + _EPSILON:
            return None
        return min(t_min, float(max_distance))
