"""C2.2 core route projection, gating and monotone high-water progress."""
import math
from pathlib import Path

import pytest

from src.rl.route_navigation import AuthoredRoute
from src.simulation.geometry.vector import Vec3
from src.tracks import load_track_asset


def _route(track_id="straight-sprint-v1"):
    asset=load_track_asset(Path("assets/tracks/training") / f"{track_id}.json")
    assert asset.navigation is not None
    return AuthoredRoute(asset.navigation)


def test_straight_projection_advances_toward_first_checkpoint() -> None:
    route=_route()
    near=route.project(Vec3(0,0,20), next_checkpoint_index=0)
    assert near.segment_index == 0
    assert near.progress_meters == pytest.approx(16.0)
    assert near.endpoint == Vec3(0,0,35)
    assert near.off_path_meters == pytest.approx(0.0)


def test_projection_clamps_at_next_checkpoint_without_credit_for_skip() -> None:
    route=_route()
    near_finish=route.project(Vec3(0,0,111), next_checkpoint_index=0)
    assert near_finish.progress_meters <= route.checkpoint_meters[0]
    assert near_finish.off_path_meters > 50
    assert route.advance_highwater(0.0, near_finish, max_path_distance=5.0) == 0.0
    after_cp=route.project(Vec3(0,0,60), next_checkpoint_index=1)
    assert after_cp.progress_meters == pytest.approx(56.0)


def test_route_highwater_never_rewards_backtracking() -> None:
    route=_route()
    first=route.project(Vec3(0,0,25), next_checkpoint_index=0)
    high=route.advance_highwater(0.0,first,max_path_distance=5.0)
    assert high == pytest.approx(21.0)
    backward=route.project(Vec3(0,0,12),next_checkpoint_index=0)
    assert route.advance_highwater(high,backward,max_path_distance=5.0) == high


def test_right_turn_endpoint_guidance_is_not_direct_finish_shortcut() -> None:
    route=_route("right-turn-v1")
    corner=route.project(Vec3(0,0,65),next_checkpoint_index=1)
    assert corner.endpoint == Vec3(0,0,70)
    after=route.project(Vec3(10,0,70),next_checkpoint_index=1)
    assert after.endpoint == Vec3(20,0,70)
    assert after.progress_meters > corner.progress_meters


def test_three_dimensional_ramp_distance_uses_arc_length() -> None:
    route=_route("ramp-v1")
    pt=route.points[3]
    projection=route.project(pt,next_checkpoint_index=0)
    assert projection.off_path_meters == pytest.approx(0.0,abs=1e-6)
    assert projection.progress_meters == pytest.approx(route.checkpoint_meters[0])
    assert route.total_meters > (route.points[-1]-route.points[0]).magnitude


def test_tie_breaking_is_deterministic() -> None:
    route=_route()
    a=route.project(Vec3(2,0,35),next_checkpoint_index=1)
    b=route.project(Vec3(2,0,35),next_checkpoint_index=1)
    assert a == b
    assert math.isfinite(a.off_path_meters)


@pytest.mark.parametrize("bad", [-1, 100, True, 0.5])
def test_invalid_checkpoint_index_fails_closed(bad) -> None:
    route=_route()
    with pytest.raises(ValueError):
        route.project(Vec3(0,0,5),next_checkpoint_index=bad)


def test_highwater_rejects_invalid_scalar_input() -> None:
    route=_route()
    point=route.project(Vec3(0,0,20),next_checkpoint_index=0)
    for value in (-1.0, float("nan"), True):
        with pytest.raises(ValueError):
            route.advance_highwater(value,point,max_path_distance=5.0)
    for value in (-1.0, 0.0, float("nan"), True):
        with pytest.raises(ValueError):
            route.advance_highwater(0.0,point,max_path_distance=value)


def test_route_progress_constrained_to_finish() -> None:
    route=_route()
    final=route.project(Vec3(0,0,112),next_checkpoint_index=2)
    assert final.progress_meters == pytest.approx(route.total_meters)
    assert final.endpoint == Vec3(0,0,112)
