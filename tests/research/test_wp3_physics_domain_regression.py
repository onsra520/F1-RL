from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from f1rl.research.wp7_protocol import build_protocol, prepare_evaluation_factory
from f1rl.sandbox.body_collision import sweep_body_against_obstacle
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.multi_surface_support import (
    _atlas, _contacts, _uncovered_area, certify_multisurface_support,
)
from f1rl.sandbox.road_footprint import body_xz_polygon
from f1rl.sandbox.vehicle import VehicleState


ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _protocol() -> dict:
    return build_protocol(ROOT, profile="spatial_rotating_owner_v3")


def _factory(track_id: str):
    factory, _ = prepare_evaluation_factory(
        ROOT, _protocol(), "multitrack-seed0", track_id, "nominal", 0,
    )
    return factory


def _pose(x: float, y: float, z: float, yaw: float) -> VehicleState:
    return VehicleState(
        Vec3(x, y, z),
        Vec3(0.0, 0.0, 0.0),
        Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), yaw),
        Vec3(0.0, 0.0, 0.0),
        True,
    )


def test_right_turn_seam_body_overhang_is_not_four_wheel_support_loss():
    factory = _factory("right-turn-v1")
    contact = factory.physics_config.independent_suspension_policy.contact
    body = factory.physics_config.body_geometry
    pose = _pose(0.49965, 0.50083, 67.93203, 0.10)
    atlas = _atlas(factory.track.surfaces, contact.seam_height_tolerance_m)
    assert atlas is not None
    wheel_contacts = _contacts(pose, contact, atlas, free_extension=True)
    assert wheel_contacts is not None and len(wheel_contacts) == 4
    assert {w.patch_id for w in wheel_contacts} == {"road-entry"}
    assert _uncovered_area(body_xz_polygon(pose, body), atlas[1]) > 0.0
    verdict = certify_multisurface_support(
        pose, pose, body, contact, factory.track.surfaces,
        body_clearance_version="bounded_patch_planes_v2",
        contact_domain="unilateral_free_extension_v1",
    )
    assert verdict.status == "outside"
    assert verdict.reason == "invalid endpoint body or wheel support"


def test_nearby_supported_body_remains_certified_on_frozen_road_atlas():
    factory = _factory("right-turn-v1")
    contact = factory.physics_config.independent_suspension_policy.contact
    pose = _pose(0.49965, 0.50083, 67.93203, 0.08)
    verdict = certify_multisurface_support(
        pose, pose, factory.physics_config.body_geometry,
        contact, factory.track.surfaces,
        body_clearance_version="bounded_patch_planes_v2",
        contact_domain="unilateral_free_extension_v1",
    )
    assert verdict.valid


def test_slalom_obstacle_contact_is_witnessed_not_uncertain_or_false_clear():
    factory = _factory("slalom-v1")
    body = factory.physics_config.body_geometry
    obstacle = next(
        entry.volume for entry in factory.track.obstacles
        if entry.volume.center.z == 100.0
    )
    first = _pose(-2.0655, 0.5008, 95.6, 0.37)
    last = _pose(-1.9222, 0.5015, 95.815, 0.37)
    verdict = sweep_body_against_obstacle(first, last, body, obstacle)
    assert verdict.contact is True
    assert verdict.uncertain is False
    assert verdict.checks >= 2
