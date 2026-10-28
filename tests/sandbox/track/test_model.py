import pytest

from src.sandbox.entities.boost_pad import BoostPad
from src.sandbox.entities.checkpoint import Checkpoint
from src.sandbox.entities.finish_line import FinishLine
from src.sandbox.entities.obstacle import Obstacle
from src.sandbox.entities.score_pickup import ScorePickup
from src.sandbox.entities.slow_zone import SlowZone
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.state import TrackState
from src.sandbox.track.volume import OrientedBoxVolume


def patch(identifier: str, height: float) -> PlanarSurfacePatch:
    return PlanarSurfacePatch(
        identifier,
        Vec3(0.0, height, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        10.0,
        10.0,
    )


def box(center_x: float = 0.0) -> OrientedBoxVolume:
    return OrientedBoxVolume(Vec3(center_x, 0.5, 0.0), Vec3(1.0, 1.0, 1.0), Quaternion.identity())


def test_surface_at_selects_nearest_then_lexical_id() -> None:
    nearest_world = TrackWorld(surfaces=(patch("low", 0.0), patch("high", 0.3)))
    nearest = nearest_world.surface_at(Vec3(0.0, 0.25, 0.0), 1.0)
    assert nearest is not None
    assert nearest.point.y == pytest.approx(0.3)

    tie_world = TrackWorld(surfaces=(patch("z-low", -0.1), patch("a-high", 0.1)))
    tied = tie_world.surface_at(Vec3(0.0, 0.0, 0.0), 1.0)
    assert tied is not None
    assert tied.point.y == pytest.approx(0.1)


def test_surface_at_returns_none_for_gap() -> None:
    world = TrackWorld(surfaces=(patch("road", 0.0),))

    assert world.surface_at(Vec3(20.0, 0.0, 20.0), 1.0) is None


def test_query_reports_all_matching_world_categories() -> None:
    shared = box()
    world = TrackWorld(
        obstacles=(Obstacle("ob", shared),),
        boost_pads=(BoostPad("boost", shared, 5.0),),
        slow_zones=(SlowZone("slow", shared, 2.0),),
        checkpoints=(Checkpoint("cp", shared, 0),),
        finish_lines=(FinishLine("finish", shared),),
        score_pickups=(ScorePickup("score", shared, 100),),
    )

    query = world.query_point(Vec3(0.0, 0.5, 0.0), 0.0, TrackState(frozenset()))

    assert [item.id for item in query.obstacles] == ["ob"]
    assert [item.id for item in query.boost_pads] == ["boost"]
    assert [item.id for item in query.slow_zones] == ["slow"]
    assert [item.id for item in query.checkpoints] == ["cp"]
    assert [item.id for item in query.finish_lines] == ["finish"]
    assert [item.id for item in query.score_pickups] == ["score"]


def test_query_filters_collected_pickups_and_sorts_matches_by_id() -> None:
    world = TrackWorld(
        obstacles=(Obstacle("z", box()), Obstacle("a", box())),
        score_pickups=(ScorePickup("z-card", box(), 50), ScorePickup("a-card", box(), 25)),
    )
    state = TrackState(frozenset({"a-card"}))

    query = world.query_point(Vec3(0.0, 0.5, 0.0), 0.0, state)

    assert [item.id for item in query.obstacles] == ["a", "z"]
    assert [item.id for item in query.score_pickups] == ["z-card"]


def test_world_rejects_duplicate_ids_across_categories() -> None:
    with pytest.raises(ValueError):
        TrackWorld(
            surfaces=(patch("shared", 0.0),),
            obstacles=(Obstacle("shared", box()),),
        )


def test_query_rejects_invalid_surface_distance() -> None:
    world = TrackWorld()

    with pytest.raises(ValueError):
        world.query_point(Vec3(0.0, 0.0, 0.0), -0.1, TrackState(frozenset()))
    with pytest.raises(ValueError):
        world.surface_at(Vec3(0.0, 0.0, 0.0), float("inf"))


def test_repeated_queries_are_exactly_deterministic() -> None:
    world = TrackWorld(
        surfaces=(patch("road", 0.0),),
        boost_pads=(BoostPad("boost", box(), 3.0),),
    )
    state = TrackState(frozenset())
    point = Vec3(0.0, 0.0, 0.0)

    assert world.query_point(point, 0.5, state) == world.query_point(point, 0.5, state)
