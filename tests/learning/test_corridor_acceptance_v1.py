"""Synthetic authored-route tests; no training or HELD-OUT asset access."""
import pytest

from src.rl.corridor_acceptance import screen_authored_route_sweep
from src.rl.route_navigation import AuthoredRoute
from src.simulation.geometry.vector import Vec3
from src.tracks.model import AuthoredNavigation


def route_straight():
    return AuthoredRoute(AuthoredNavigation(
        (Vec3(0,0,-10), Vec3(0,0,10)), ("road",), (),
    ))


def route_corner():
    return AuthoredRoute(AuthoredNavigation(
        (Vec3(-5,0,-5), Vec3(-5,0,5), Vec3(5,0,5)),
        ("vertical", "horizontal"), (),
    ))


def test_center_follows_line_without_route_penalty():
    report=screen_authored_route_sweep(route_straight(),Vec3(0,0,-5),Vec3(0,0,5))
    assert report.status=="clear"
    assert report.safe_for_route
    assert not report.soft_zone_encountered


def test_soft_zone_reported_separately_from_hard_violation():
    report=screen_authored_route_sweep(route_straight(),Vec3(3,0,-2),Vec3(3,0,2))
    assert report.status=="clear"
    assert report.soft_zone_encountered
    assert report.max_sampled_deviation_m==pytest.approx(3.)


def test_hard_violation_is_strictly_greater_than_4m():
    report=screen_authored_route_sweep(route_straight(),Vec3(4.001,0,-2),Vec3(4.001,0,2))
    assert report.status=="hard_violation"
    assert not report.safe_for_route


def test_cross_corner_with_valid_endpoints_can_violate_midstep():
    report=screen_authored_route_sweep(
        route_corner(),Vec3(-5,0,-4),Vec3(4,0,5),
    )
    assert report.status=="hard_violation"
    assert report.max_sampled_deviation_m > 4.


def test_low_resolution_uncertainty_fails_closed_not_success():
    route=route_straight()
    a=Vec3(3.99,0,-1)
    b=Vec3(3.99,0,1)
    unresolved=screen_authored_route_sweep(route,a,b,max_subdivisions=0)
    assert unresolved.status=="uncertain" and not unresolved.safe_for_route
    detailed=screen_authored_route_sweep(route,a,b,max_subdivisions=12)
    assert detailed.status=="clear"


def test_lateral_distance_is_horizontal_not_airborne_height():
    report=screen_authored_route_sweep(route_straight(),Vec3(0,80,-2),Vec3(0,80,2))
    assert report.status=="clear"


@pytest.mark.parametrize("limits",[(2.1,4.),(2.,5.),(4.,2.),(-1.,4.),(2.,float("inf"))])
def test_unapproved_route_limits_are_rejected(limits):
    with pytest.raises(ValueError):
        screen_authored_route_sweep(route_straight(),Vec3(0,0,0),Vec3(1,0,0),
                                   soft_limit_m=limits[0],hard_limit_m=limits[1])


@pytest.mark.parametrize("depth",[-1,21,True,1.5])
def test_invalid_subdivision_budget_rejected(depth):
    with pytest.raises(ValueError,match="max_subdivisions"):
        screen_authored_route_sweep(route_straight(),Vec3(0,0,0),Vec3(1,0,0),max_subdivisions=depth)
