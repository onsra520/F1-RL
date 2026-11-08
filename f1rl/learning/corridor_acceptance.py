"""Prospective 2m soft / 4m hard authored-route acceptance geometry.

This is an independent, read-only geometric screen, not the legacy
offtrack reward or the frozen scientific training termination policy.
Horizontal XZ distance is 1-Lipschitz in XZ position, enabling a
conservative continuous-segment proof under linear translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from ..sandbox.geometry.vector import Vec3
from .route_navigation import AuthoredRoute


@dataclass(frozen=True,slots=True)
class RouteCorridorVerdict:
    status: str
    max_sampled_deviation_m: float
    checks: int
    soft_zone_encountered: bool

    def __post_init__(self) -> None:
        if self.status not in ("clear", "hard_violation", "uncertain"):
            raise ValueError("invalid route status")
        if type(self.checks) is not int or self.checks < 1:
            raise ValueError("at least one route check required")
        if not isfinite(self.max_sampled_deviation_m) or self.max_sampled_deviation_m < 0:
            raise ValueError("invalid recorded route deviation")

    @property
    def safe_for_route(self) -> bool:
        return self.status == "clear"


def screen_authored_route_sweep(
    route: AuthoredRoute,
    previous: Vec3,
    candidate: Vec3,
    *,
    soft_limit_m: float = 2.0,
    hard_limit_m: float = 4.0,
    max_subdivisions: int = 12,
) -> RouteCorridorVerdict:
    """Certify continuous centre XZ displacement within 4 m authored route.

    Soft-zone telemetry records >2m deviations; the *reward coefficient*
    and hard-violation termination semantics are deliberately NOT set here.
    The whole-body road union validator is a separate mandatory gate.
    """
    if not isinstance(route, AuthoredRoute) or not isinstance(previous, Vec3) or not isinstance(candidate, Vec3):
        raise ValueError("route and two Vec3 endpoints required")
    if (
        type(soft_limit_m) not in (float, int)
        or type(hard_limit_m) not in (float, int)
        or not isfinite(soft_limit_m) or not isfinite(hard_limit_m)
        or soft_limit_m < 0 or hard_limit_m <= soft_limit_m
    ):
        raise ValueError("route soft/hard limits must be finite ordered values")
    if float(soft_limit_m) != 2.0 or float(hard_limit_m) != 4.0:
        raise ValueError("preregistered route limits are 2m soft and 4m hard")
    if type(max_subdivisions) is not int or not 0 <= max_subdivisions <= 20:
        raise ValueError("max_subdivisions must be integer in [0,20]")

    delta = candidate - previous
    total_xz = hypot(delta.x, delta.z)
    checks=0
    peak=0.0
    soft_seen=False

    def at(t:float) -> float:
        nonlocal checks,peak,soft_seen
        checks += 1
        position = previous+delta*t
        distance=route.horizontal_distance(position)
        peak=max(peak,distance)
        soft_seen |= distance > float(soft_limit_m)
        return distance

    if at(0.)>hard_limit_m or at(1.)>hard_limit_m:
        return RouteCorridorVerdict("hard_violation",peak,checks,soft_seen)

    def visit(lo:float,hi:float,depth:int)->str:
        mid=(lo+hi)/2
        distance=at(mid)
        if distance>hard_limit_m:
            return "hard_violation"
        # Distance-to-set is 1-Lipschitz. The midpoint distance bounds
        # every continuous interpolated centre position in this interval.
        if distance + total_xz*(hi-lo)/2 <= hard_limit_m:
            return "clear"
        if depth==0:
            return "uncertain"
        left=visit(lo,mid,depth-1)
        if left=="hard_violation":
            return left
        right=visit(mid,hi,depth-1)
        if right=="hard_violation":
            return right
        return "uncertain" if "uncertain" in (left,right) else "clear"

    status=visit(0.,1.,max_subdivisions)
    return RouteCorridorVerdict(status,peak,checks,soft_seen)
