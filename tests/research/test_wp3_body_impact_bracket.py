from __future__ import annotations

from dataclasses import replace

import pytest

from src.sandbox.body_collision import SweptCollisionResult, VehicleBodyGeometry
from src.sandbox.body_impact import first_body_impact_bracket
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState


def pose(x: float, z: float = 0) -> VehicleState:
    return VehicleState(Vec3(x, 0, z), Vec3(0, 0, 0), Quaternion.identity(), Vec3(0, 0, 0), True)


BODY = VehicleBodyGeometry(1.0, 1.0, 1.0, Vec3(0, 0, 0))
OBSTACLE = OrientedBoxVolume(Vec3(0, 0, 0), Vec3(0.05, 1.0, 1.0), Quaternion.identity())


def test_fast_sweep_locates_first_contact_before_last_obstacle_exit():
    result = first_body_impact_bracket(pose(-5), pose(5), BODY, OBSTACLE, duration_s=0.1, max_refinements=12)
    assert result is not None
    assert result.last_proven_clear_s < 0.0445 <= result.first_witnessed_contact_by_s
    assert result.uncertainty_s <= 0.1 / 4096 + 1e-15
    assert result.sweep_checks > 0


def test_slalom_obstacle_impact_is_bracketed_without_fabricating_response():
    obstacle = OrientedBoxVolume(
        Vec3(-3.5, 1.0, 100.0), Vec3(2.0, 1.0, 2.0), Quaternion.identity()
    )
    body = VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0.0, 0.3, 0.0))
    orientation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 0.37)
    first = replace(pose(-2.0655, 95.60), orientation=orientation, position=Vec3(-2.0655, 0.5008, 95.60))
    last = replace(pose(-1.9222, 95.815), orientation=orientation, position=Vec3(-1.9222, 0.5008, 95.815))
    result = first_body_impact_bracket(first, last, body, obstacle, duration_s=0.1)
    assert result is not None
    assert 0.0975 < result.last_proven_clear_s < result.first_witnessed_contact_by_s < 0.0978
    assert result.uncertainty_s <= 0.1 / 4096 + 1e-15


def test_moving_far_away_has_no_impact():
    assert first_body_impact_bracket(pose(-5, 10), pose(5, 10), BODY, OBSTACLE, duration_s=0.1) is None


def test_contact_at_initial_pose_is_rejected():
    with pytest.raises(ValueError, match="already overlapping"):
        first_body_impact_bracket(pose(0), pose(5), BODY, OBSTACLE, duration_s=0.1)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True])
def test_invalid_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="finite positive duration"):
        first_body_impact_bracket(pose(-5), pose(5), BODY, OBSTACLE, duration_s=duration)


@pytest.mark.parametrize("count", [0, -1, 21, True, 1.5])
def test_invalid_refinements_are_rejected(count):
    with pytest.raises(ValueError, match="refinement count"):
        first_body_impact_bracket(pose(-5), pose(5), BODY, OBSTACLE, duration_s=0.1, max_refinements=count)


def test_uncertain_sweep_cannot_be_forged_into_impact(monkeypatch):
    import src.sandbox.body_impact as module
    monkeypatch.setattr(module, "sweep_body_against_obstacle", lambda *args: SweptCollisionResult(False, True, 1))
    with pytest.raises(ValueError, match="cannot be certified"):
        first_body_impact_bracket(pose(-5), pose(5), BODY, OBSTACLE, duration_s=0.1)
