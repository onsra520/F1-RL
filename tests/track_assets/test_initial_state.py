import math

import pytest

from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.track_assets import SpawnPose, build_initial_world_state
from f1rl.track_assets.canonical import _build_track_asset


def _surface():
    return PlanarSurfacePatch(
        "road",
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        10.0,
        20.0,
    )


def _asset(position):
    return _build_track_asset(
        schema_version=1,
        track_id="state-test",
        spawn=SpawnPose(position, Quaternion.identity()),
        world=TrackWorld((_surface(),)),
    )


def test_initial_state_preserves_spawn_and_zeroes_mutable_state():
    position = Vec3(1.0, 0.0, 2.0)
    state = build_initial_world_state(_asset(position), max_surface_distance=0.0)
    assert state.vehicle.position == position
    assert state.vehicle.orientation == Quaternion.identity()
    assert state.vehicle.velocity == Vec3(0.0, 0.0, 0.0)
    assert state.vehicle.angular_velocity == Vec3(0.0, 0.0, 0.0)
    assert state.track.collected_pickup_ids == frozenset()
    assert state.interactions.active_checkpoint_ids == frozenset()
    assert state.interactions.active_finish_line_ids == frozenset()
    assert state.score.points == 0


def test_spawn_on_surface_is_grounded_without_snapping_position():
    position = Vec3(2.0, 0.0, 3.0)
    state = build_initial_world_state(_asset(position), max_surface_distance=0.0)
    assert state.vehicle.grounded is True
    assert state.vehicle.position == position


def test_spawn_outside_surface_query_is_not_grounded():
    position = Vec3(0.0, 5.0, 0.0)
    state = build_initial_world_state(_asset(position), max_surface_distance=1.0)
    assert state.vehicle.grounded is False
    assert state.vehicle.position == position


@pytest.mark.parametrize("value", [True, -0.1, math.nan, math.inf, -math.inf, "1"])
def test_invalid_max_surface_distance_is_rejected(value):
    with pytest.raises(ValueError, match="max_surface_distance"):
        build_initial_world_state(_asset(Vec3(0.0, 0.0, 0.0)), max_surface_distance=value)
