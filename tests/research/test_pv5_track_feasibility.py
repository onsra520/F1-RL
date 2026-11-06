from dataclasses import replace
from pathlib import Path

import pytest

from f1rl.research.wp7_track_feasibility import audit_train_tracks, audit_track
from f1rl.track_assets import load_track_asset
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from math import radians

ROOT = Path(__file__).resolve().parents[2]


def test_four_real_train_assets_have_hash_bound_body_aware_findings():
    report = audit_train_tracks(ROOT)
    assert [r["track_id"] for r in report["tracks"]] == [
        "straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"]
    rows = {r["track_id"]: r for r in report["tracks"]}
    assert rows["straight-sprint-v1"]["geometry_status"] == "certified_authored_sweep"
    assert rows["right-turn-v1"]["turning"]["minimum_radius_m"] > 5.
    assert "authored_heading_discontinuity" in rows["right-turn-v1"]["blockers"]
    assert "directed_gate_not_crossed:finish" in rows["right-turn-v1"]["blockers"]
    assert rows["ramp-v1"]["geometry_status"] == "uncertified"
    assert all(len(r["source_sha256"]) == 64 for r in report["tracks"])
    assert report["training_authorized"] is False


def test_narrow_gate_cannot_pass_because_centerline_fits():
    asset = load_track_asset(ROOT / "assets/tracks/training/straight-sprint-v1.json")
    gate = asset.world.checkpoints[0]
    gate = replace(gate, volume=replace(gate.volume, half_extents=Vec3(.4, 2., 1.)))
    world = replace(asset.world, checkpoints=(gate, *asset.world.checkpoints[1:]))
    result = audit_track(replace(asset, world=world))
    assert f"body_aperture:{gate.id}" in result["blockers"]
    assert result["world_sha256"] != asset.world_sha256


def test_narrow_road_rejects_full_body_while_centerline_remains_inside():
    asset = load_track_asset(ROOT / "assets/tracks/training/straight-sprint-v1.json")
    result = audit_track(replace(asset, world=replace(asset.world,
        surfaces=tuple(replace(s, width=1.) for s in asset.world.surfaces))))
    assert result["geometry_status"] == "uncertified"
    assert any(b.startswith("body_road:") for b in result["blockers"])


def test_unknown_track_is_rejected_before_any_asset_access():
    with pytest.raises(ValueError, match="four TRAIN"):
        audit_train_tracks(ROOT, track_ids=("heldout-v1",))


def test_oblique_aperture_checks_entire_body_passage_not_center_pose_projection():
    asset = load_track_asset(ROOT / "assets/tracks/training/straight-sprint-v1.json")
    gate = asset.world.checkpoints[0]
    gate = replace(gate, volume=replace(gate.volume, half_extents=Vec3(2.2, 2., 1.),
        orientation=Quaternion.from_axis_angle(Vec3(0.,1.,0.), radians(80))))
    result = audit_track(replace(asset, world=replace(asset.world,
        checkpoints=(gate, *asset.world.checkpoints[1:]))))
    assert f"body_aperture:{gate.id}" in result["blockers"]
