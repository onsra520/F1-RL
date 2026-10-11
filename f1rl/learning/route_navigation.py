"""Pure geometry for authored route guidance and checkpoint-gated progress.

This module does not own simulator lifecycle, reward weights or PPO training.
It computes a deterministic projection; runtime callers own the high-water state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ..sandbox.geometry.vector import Vec3
from ..track_assets.model import AuthoredNavigation


@dataclass(frozen=True, slots=True)
class RouteProjection:
    segment_index: int
    segment_fraction: float
    progress_meters: float
    off_path_meters: float
    endpoint: Vec3

    def __post_init__(self) -> None:
        if isinstance(self.segment_index, bool) or not isinstance(self.segment_index, int) or self.segment_index < 0:
            raise ValueError("segment_index must be non-negative integer")
        for name, value in (
            ("segment_fraction", self.segment_fraction),
            ("progress_meters", self.progress_meters),
            ("off_path_meters", self.off_path_meters),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.segment_fraction <= 1.0 or self.progress_meters < 0.0 or self.off_path_meters < 0.0:
            raise ValueError("projection values must be in range")
        if not isinstance(self.endpoint, Vec3):
            raise ValueError("endpoint must be Vec3")


@dataclass(frozen=True, slots=True)
class AuthoredRoute:
    navigation: AuthoredNavigation
    _distances: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.navigation, AuthoredNavigation):
            raise ValueError("navigation must be authored and validated")
        distances = [0.0]
        for first, second in zip(
            self.navigation.points, self.navigation.points[1:], strict=False
        ):
            length = (second - first).magnitude
            if length <= 1e-8:
                raise ValueError("navigation segments must have positive length")
            distances.append(distances[-1] + length)
        object.__setattr__(self, "_distances", tuple(distances))

    @property
    def points(self) -> tuple[Vec3, ...]:
        return self.navigation.points

    @property
    def total_meters(self) -> float:
        return self._distances[-1]

    @property
    def checkpoint_meters(self) -> tuple[float, ...]:
        return tuple(
            self._distances[i]
            for i in self.navigation.checkpoint_point_indices
        )

    def project(self, position: Vec3, *, next_checkpoint_index: int) -> RouteProjection:
        if not isinstance(position, Vec3):
            raise ValueError("position must be Vec3")
        if (
            isinstance(next_checkpoint_index, bool)
            or not isinstance(next_checkpoint_index, int)
            or not 0 <= next_checkpoint_index <= len(self.checkpoint_meters)
        ):
            raise ValueError("next_checkpoint_index outside route bounds")

        # The vehicle may not earn forward progress on segments *after* an
        # uncompleted race checkpoint. Earlier completed segments remain
        # candidates for deterministic nearest-route projection.
        max_point_index = (
            self.navigation.checkpoint_point_indices[next_checkpoint_index]
            if next_checkpoint_index < len(self.checkpoint_meters)
            else len(self.points) - 1
        )
        best: tuple[float, int, float] | None = None
        for segment in range(max_point_index):
            start = self.points[segment]
            delta = self.points[segment + 1] - start
            squared_length = delta.dot(delta)
            if squared_length <= 0.0:
                raise ValueError("navigation segment has zero length")
            fraction = max(0.0, min(1.0, (position - start).dot(delta) / squared_length))
            closest = start + delta * fraction
            off = (position - closest).magnitude
            # Tie-break consistently by authored segment order; this avoids
            # relying on floating-point hash/dictionary iteration behavior.
            candidate = (off, segment, fraction)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        assert best is not None
        off, segment, fraction = best
        length = self._distances[segment + 1] - self._distances[segment]
        progress = self._distances[segment] + fraction * length
        return RouteProjection(segment, fraction, progress, off, self.points[segment + 1])

    def advance_highwater(
        self,
        previous_meters: float,
        projection: RouteProjection,
        *,
        max_path_distance: float,
    ) -> float:
        if (
            isinstance(previous_meters, bool)
            or not isinstance(previous_meters, (int, float))
            or not isfinite(previous_meters)
            or previous_meters < 0.0
            or previous_meters > self.total_meters + 1e-7
        ):
            raise ValueError("previous_meters must be finite within route")
        if not isinstance(projection, RouteProjection):
            raise ValueError("projection must be RouteProjection")
        if (
            isinstance(max_path_distance, bool)
            or not isinstance(max_path_distance, (int, float))
            or not isfinite(max_path_distance)
            or max_path_distance <= 0.0
        ):
            raise ValueError("max_path_distance must be finite positive")
        if projection.off_path_meters > max_path_distance:
            return float(previous_meters)
        return max(float(previous_meters), projection.progress_meters)
