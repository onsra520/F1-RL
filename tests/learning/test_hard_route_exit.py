"""First exit from the approved closed 4m tube around an authored route."""
import pytest
from f1rl.learning.hard_route_outcome import first_hard_route_exit
from f1rl.learning.route_navigation import AuthoredRoute
from f1rl.track_assets.model import AuthoredNavigation
from f1rl.sandbox.geometry.vector import Vec3


def route(points):
    return AuthoredRoute(AuthoredNavigation(tuple(Vec3(x,0.,z) for x,z in points),
        ('road',)*(len(points)-1),()))


def test_strict_boundary_and_outward_start():
    r=route([(0.,-20.),(0.,20.)])
    assert first_hard_route_exit(r,Vec3(4.,0.,0.),Vec3(4.,0.,5.)) is None
    assert first_hard_route_exit(r,Vec3(4.,0.,0.),Vec3(5.,0.,0.))==0.
    assert first_hard_route_exit(r,Vec3(0.,0.,0.),Vec3(5.,0.,0.))==pytest.approx(.8)


def test_earliest_exit_is_retained_when_path_returns_inside_later():
    r=route([(0.,0.),(0.,20.),(20.,20.),(20.,0.)])
    assert first_hard_route_exit(r,Vec3(0.,0.,0.),Vec3(20.,0.,0.))==pytest.approx(.2)


def test_endpoint_caps_and_overlapping_tubes_preserve_connected_coverage():
    r=route([(0.,0.),(0.,10.)])
    assert first_hard_route_exit(r,Vec3(0.,0.,0.),Vec3(0.,0.,14.)) is None
    assert first_hard_route_exit(r,Vec3(0.,0.,0.),Vec3(0.,0.,20.))==pytest.approx(.7)
    bend=route([(0.,0.),(0.,8.),(8.,8.)])
    assert first_hard_route_exit(bend,Vec3(0.,0.,0.),Vec3(8.,0.,8.)) is None


def test_an_already_invalid_route_start_is_not_a_valid_failure_sample():
    with pytest.raises(ValueError,match='outside'):
        first_hard_route_exit(route([(0.,-20.),(0.,20.)]),Vec3(5.,0.,0.),Vec3(6.,0.,0.))


def test_capsule_intervals_agree_with_independent_distance_samples_across_bent_routes():
    import random
    rng=random.Random(73021)
    for _ in range(80):
        points=[(0.,0.)]
        for _ in range(4):
            points.append((points[-1][0]+rng.uniform(-12.,12.),points[-1][1]+rng.uniform(-12.,12.)))
        r=route(points)
        start=Vec3(0.,rng.uniform(-5.,5.),0.)
        end=Vec3(rng.uniform(-30.,30.),rng.uniform(-5.,5.),rng.uniform(-30.,30.))
        t=first_hard_route_exit(r,start,end)
        stop=1. if t is None else t
        assert all(r.horizontal_distance(start+(end-start)*(stop*i/100.))<=4.+1e-10 for i in range(101))
        if t is not None:
            assert r.horizontal_distance(start+(end-start)*min(1.,t+1e-6))>4.


def test_vertical_authored_segment_has_a_point_capsule_in_horizontal_plane():
    r=AuthoredRoute(AuthoredNavigation((Vec3(0.,0.,0.),Vec3(0.,3.,0.)),('road',),()))
    assert first_hard_route_exit(r,Vec3(0.,0.,0.),Vec3(5.,10.,0.))==pytest.approx(.8)
