from dataclasses import dataclass
from math import isclose

from .geometry.vector import Vec3


@dataclass(frozen=True, slots=True)
class SurfaceContact:
    point: Vec3
    normal: Vec3

    def __post_init__(self) -> None:
        if not isclose(self.normal.magnitude, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("surface normal must be unit length")
