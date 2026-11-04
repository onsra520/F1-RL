from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

from ..geometry.vector import Vec3
from ..surface import SurfaceContact

_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class PlanarSurfacePatch:
    id: str
    origin: Vec3
    forward: Vec3
    normal: Vec3
    width: float
    length: float
    route_index: int | None = None
    route_overlap_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("patch id must be a non-empty string")
        if not isclose(self.forward.magnitude, 1.0, rel_tol=0.0, abs_tol=_TOLERANCE):
            raise ValueError("forward must be unit length")
        if not isclose(self.normal.magnitude, 1.0, rel_tol=0.0, abs_tol=_TOLERANCE):
            raise ValueError("normal must be unit length")
        if not isclose(self.forward.dot(self.normal), 0.0, rel_tol=0.0, abs_tol=_TOLERANCE):
            raise ValueError("forward and normal must be orthogonal")
        if not isfinite(self.width) or self.width <= 0.0:
            raise ValueError("width must be finite and positive")
        if not isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if self.route_index is not None and (type(self.route_index) is not int or self.route_index < -1):
            raise ValueError("route_index must be an integer greater than or equal to -1")
        if not isfinite(self.route_overlap_m) or self.route_overlap_m < 0.0 or self.route_overlap_m >= self.length:
            raise ValueError("route_overlap_m must be finite and within [0, length)")
        if self.route_overlap_m and self.route_index is None:
            raise ValueError("route overlap requires an authored route index")

    @property
    def right(self) -> Vec3:
        return self.normal.cross(self.forward)

    def sample(self, position: Vec3, max_distance: float) -> SurfaceContact | None:
        if not isfinite(max_distance) or max_distance < 0.0:
            raise ValueError("max_distance must be finite and non-negative")

        offset = position - self.origin
        signed_distance = offset.dot(self.normal)
        if max_distance == 0.0:
            if signed_distance != 0.0:
                return None
        elif abs(signed_distance) > max_distance + _TOLERANCE:
            return None

        projected = position - self.normal * signed_distance
        local = projected - self.origin
        local_right = local.dot(self.right)
        local_forward = local.dot(self.forward)

        if abs(local_right) > self.width / 2.0 + _TOLERANCE:
            return None
        if abs(local_forward) > self.length / 2.0 + _TOLERANCE:
            return None
        return SurfaceContact(projected, self.normal)
