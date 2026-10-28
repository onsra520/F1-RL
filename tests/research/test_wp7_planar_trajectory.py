"""Certificates use real TRAIN geometry, without PPO or canonical asset edits."""
from pathlib import Path
from dataclasses import replace
import json

import pytest

from src.research.wp7_planar_trajectory import search_planar_trajectory, certify_planar_trajectory
from src.research.wp7_protocol import digest
from src.sandbox.geometry.vector import Vec3
from src.track_assets import load_track_asset

ROOT = Path(__file__).resolve().parents[2]


def test_smooth_alternative_certifies_real_slalom_body_clearance_and_steering():
    asset = load_track_asset(ROOT / "assets/tracks/training/slalom-v1.json")
    result = search_planar_trajectory(asset)
    assert result["geometry_status"] == "certified_continuous_planar_path"
    assert result["curvature_upper_bound_per_m"] < result["steering_curvature_limit_per_m"]
    assert result["directed_gate_ids"] == ["checkpoint-0", "checkpoint-1", "checkpoint-2", "finish"]
    assert result["world_sha256"] == asset.world_sha256
    assert result["speed_contact_status"] == "unresolved"
    assert result["training_authorized"] is False


def straight():
    return load_track_asset(ROOT / "assets/tracks/training/straight-sprint-v1.json")


def test_certificate_is_replayable_after_json_serialization():
    result = search_planar_trajectory(straight())
    assert result["geometry_status"] == "certified_continuous_planar_path"
    record = json.loads(json.dumps(result))
    assert record["sha256"] == digest({k:v for k,v in record.items() if k != "sha256"})
    assert record == json.loads(json.dumps(search_planar_trajectory(straight())))


def test_center_crossing_cannot_certify_a_narrow_body_aperture():
    asset = straight()
    first,*others = asset.world.checkpoints
    gate = replace(first,volume=replace(first.volume,half_extents=Vec3(.4,2.,1.)))
    result = search_planar_trajectory(replace(asset,world=replace(asset.world,checkpoints=(gate,*others))))
    assert result["geometry_status"] == "unresolved"
    assert f"continuous_body_aperture:{gate.id}" in result["blockers"]


def test_narrow_road_is_not_certified_from_center_samples():
    asset = straight()
    road = replace(asset.world.surfaces[0],width=1.)
    result = search_planar_trajectory(replace(asset,world=replace(asset.world,surfaces=(road,))))
    assert result["geometry_status"] == "unresolved"
    assert result["blockers"] == ["continuous_body_road:outside"]


def test_curvature_limit_blocks_an_impossibly_short_lane_change():
    result = certify_planar_trajectory(straight(),((0.,4.),(3.,5.),(0.,116.)))
    assert "steering_curvature_bound_exceeded" in result["blockers"]
    assert result["intervals_certified"] == 0


def test_path_must_continue_until_entire_body_clears_finish():
    result = certify_planar_trajectory(straight(),((0.,4.),(0.,112.1)))
    assert result["geometry_status"] == "unresolved"
    assert "body_does_not_clear_finish_plane" in result["blockers"]


@pytest.mark.parametrize("knots", [((0.,4.),(0.,4.)), ((True,4.),(0.,116.)),
                                   ((0.,float('nan')),(0.,116.)), ((0.,0.),(0.,1e-200))])
def test_invalid_or_degenerate_path_rejected(knots):
    with pytest.raises(ValueError,match="knots"):
        certify_planar_trajectory(straight(),knots)


@pytest.mark.parametrize("track", ["ramp-v1"])
def test_failed_candidate_search_never_claims_general_infeasibility(track):
    asset = load_track_asset(ROOT / f"assets/tracks/training/{track}.json")
    result = search_planar_trajectory(asset)
    assert result["geometry_status"] == "unresolved"
    assert result["blockers"]
    assert result["search_scope"] == "finite_graph_family_not_infeasibility_proof"


def test_unknown_track_rejected_without_any_sealed_asset_read():
    with pytest.raises(ValueError,match="TRAIN"):
        search_planar_trajectory(replace(straight(),track_id="unknown-fixture"))
