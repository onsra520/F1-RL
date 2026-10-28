from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .body_collision import (
    VehicleBodyGeometry,
    _slerp,
    obb_separation_lower_bound,
    sweep_body_against_obstacle,
)
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState


@dataclass(frozen=True, slots=True)
class BodyImpactTimeBracket:
    last_proven_clear_s: float
    first_witnessed_contact_by_s: float
    sweep_checks: int

    @property
    def uncertainty_s(self) -> float:
        return self.first_witnessed_contact_by_s - self.last_proven_clear_s


def first_body_impact_bracket(
    previous: VehicleState,
    candidate: VehicleState,
    geometry: VehicleBodyGeometry,
    obstacle: OrientedBoxVolume,
    *,
    duration_s: float,
    max_refinements: int = 12,
) -> BodyImpactTimeBracket | None:
    if not isinstance(previous, VehicleState) or not isinstance(candidate, VehicleState):
        raise ValueError("impact bracket requires two vehicle states")
    if not isinstance(geometry, VehicleBodyGeometry) or not isinstance(obstacle, OrientedBoxVolume):
        raise ValueError("impact bracket requires body and obstacle geometry")
    if type(duration_s) not in (int, float) or not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("impact bracket requires finite positive duration")
    if type(max_refinements) is not int or not 1 <= max_refinements <= 20:
        raise ValueError("impact bracket refinement count is invalid")
    if obb_separation_lower_bound(geometry.box_at(previous.position, previous.orientation), obstacle) <= 1e-10:
        raise ValueError("impact bracket starts from an already overlapping body")

    displacement = candidate.position - previous.position

    def pose_at(fraction: float) -> VehicleState:
        return VehicleState(
            previous.position + displacement * fraction,
            previous.velocity,
            _slerp(previous.orientation, candidate.orientation, fraction),
            previous.angular_velocity,
            previous.grounded,
        )

    checks = 0

    def witness(before: float) -> bool:
        nonlocal checks
        verdict = sweep_body_against_obstacle(previous, pose_at(before), geometry, obstacle)
        checks += verdict.checks
        if verdict.uncertain:
            raise ValueError("body impact time cannot be certified at current sweep resolution")
        return verdict.contact

    if not witness(1.0):
        return None
    lower = 0.0
    upper = 1.0
    for _ in range(max_refinements):
        midpoint = (lower + upper) * 0.5
        if witness(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return BodyImpactTimeBracket(
        last_proven_clear_s=lower * duration_s,
        first_witnessed_contact_by_s=upper * duration_s,
        sweep_checks=checks,
    )
