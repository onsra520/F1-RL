"""T2: version-bound CUSTOM Auto Drive v4 failure is a certified terminal, not RaceContractError.

The artificial gap here is a development diagnostic only. It MUST NOT be
treated as an editor-valid jump map or independent scientific result.
"""
from dataclasses import replace
from pathlib import Path

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import convert_custom_track
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.session import RuntimeSessionManager
from src.simulation.geometry.vector import Vec3

ROOT = Path(__file__).resolve().parents[2]


def test_development_gap_crash_uses_registered_physics_and_race_v4_terminal():
    catalog = RuntimeCatalog(ROOT)
    custom = convert_custom_track({
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "v4-crash-only-test",
        "name": "V4 crash terminal fixture",
        "width": 12.0,
        "points": [{"x": 0.0, "z": 0.0}, {"x": 0.0, "z": 85.0}],
        "props": [],
    }, catalog)
    original = custom.factory
    surfaces = tuple(
        replace(patch, origin=patch.origin + Vec3(900.0, 0.0, 0.0))
        if patch.route_index is not None and 27.0 <= patch.origin.z <= 31.0
        else patch for patch in original.track.surfaces
    )
    diagnostic_factory = replace(original, track=replace(original.track, surfaces=surfaces))
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None, track_id=custom.track_id, mode="auto", seed=412,
            custom_factory=diagnostic_factory,
            custom_track_sha256=custom.track_sha256,
            physics_profile="experimental_v4",
            auto_drive_config=AutoDriveConfig(
                driving_style="centerline",
                speed_profile="hyper",
                target_max_speed_kmh=120.0,
                acceleration_factor=1.0,
            ),
        )
        initial = session.status_document()
        assert initial["physics"]["ground_motion_version"] == "spatial_road_edge_partial_lifecycle_v4"
        assert initial["race_contract"]["version"] == "wp7_directed_lifecycle_crash_contract_v4"
        session.start()
        assert session.wait(90.0), "development v4 terminal must not hang"
        status = session.status_document()
        assert status["status"] == "finished", status["error"]
        assert status["error"] is None
        result = status["episode_result"]
        assert result["reason"] == "physical_crash", result
        assert result["successful_finish"] is False
        assert result["valid_sample"] is False  # custom development, never scientific Eval
        replay = session.replay()
        assert replay["result"]["reason"] == "physical_crash"
        assert replay["provenance"]["dataset_split"] == "DEVELOPMENT"
    finally:
        manager.shutdown()

def test_actual_editor_gap_takeoff_is_physics_sourced_and_unresolved_pitch_fails_closed():
    """No synthetic landing may turn a genuinely unsafe takeoff into PASS."""
    catalog = RuntimeCatalog(ROOT)
    custom = convert_custom_track({
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "t3-takeoff-regression",
        "name": "T3 original real five metre gap",
        "width": 12.0,
        "points": [{"x": 0.0, "z": 0.0}, {"x": 0.0, "z": 86.0}],
        "props": [{"id": "jump-gap", "kind": "gap",
                   "t": 0.5, "offset": 5.0, "rotation": 0.0}],
    }, catalog)
    assert custom.ppo_compatible is False
    assert custom.factory.track.surface_at(Vec3(0.0, 0.0, 43.0), 0.0) is None
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None, track_id=custom.track_id, mode="auto", seed=412,
            custom_factory=custom.factory,
            custom_track_sha256=custom.track_sha256,
            physics_profile="experimental_v4",
            auto_drive_config=AutoDriveConfig(
                driving_style="centerline", speed_profile="hyper",
                target_max_speed_kmh=120.0, acceleration_factor=1.0,
                braking_factor=0.8),
        )
        session.start()
        assert session.wait(90.0)
        status, replay = session.status_document(), session.replay()
        assert status["status"] == "failed"
        assert "pitch=" in status["error"]
        assert replay["result"]["valid_sample"] is False
        assert replay["result"]["successful_finish"] is False
        frames = replay["frames"]
        assert frames[0]["grounded"] is True
        takeoff = [f for f in frames if "takeoff" in f["physical_event_types"]]
        assert len(takeoff) == 1
        assert takeoff[0]["grounded"] is False
        assert all("landing" not in f["physical_event_types"] for f in frames)
        assert frames[-1]["grounded"] is False
    finally:
        manager.shutdown()

def test_t3_versioned_custom_jump_auto_drive_proves_full_physics_flight_then_finish():
    """Only authoritative Gym V4 takeoff, accepted airborne and landing count."""
    import json

    catalog = RuntimeCatalog(ROOT)
    doc = json.loads(
        (ROOT / "examples/t3-v4-launch-landing-map.json").read_text(encoding="utf-8")
    )
    converted = convert_custom_track(doc, catalog)
    assert converted.ppo_compatible is False
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(
            model_id=None,
            track_id=converted.track_id,
            mode="auto",
            seed=412,
            custom_factory=converted.factory,
            custom_track_sha256=converted.track_sha256,
            physics_profile="experimental_v4",
            auto_drive_config=AutoDriveConfig(
                driving_style="centerline",
                speed_profile="hyper",
                target_max_speed_kmh=120.0,
                acceleration_factor=1.0,
                braking_factor=0.8,
                moderate_grade_threshold=0.3,
                steep_grade_threshold=0.5,
                moderate_grade_speed_mps=15.0,
                steep_grade_speed_mps=15.0,
            ),
        )
        identity = session.status_document()
        assert identity["physics"]["ground_motion_version"] == (
            "spatial_road_edge_partial_lifecycle_v4"
        )
        assert identity["race_contract"]["version"] == (
            "wp7_directed_lifecycle_crash_contract_v4"
        )
        session.start()
        assert session.wait(120.0), "must terminate rather than hang after landing"
        status = session.status_document()
        replay = session.replay()
        assert status["error"] is None, status["error"]
        assert status["status"] == "finished"
        assert replay["result"]["successful_finish"] is True
        assert replay["result"]["reason"] == "finish"
        assert replay["result"]["valid_sample"] is False  # custom development
        assert replay["provenance"]["dataset_split"] == "DEVELOPMENT"
        frames = replay["frames"]
        assert [f["tick"] for f in frames] == list(range(len(frames)))
        assert frames[0]["grounded"] is True
        takeoff = [f for f in frames if "takeoff" in f["physical_event_types"]]
        airborne = [f for f in frames if f["grounded"] is False]
        landing = [f for f in frames if "landing" in f["physical_event_types"]]
        assert len(takeoff) == len(landing) == 1
        assert takeoff[0]["tick"] < airborne[-1]["tick"] < landing[0]["tick"]
        assert all(f["grounded"] is True for f in frames[landing[0]["tick"]:])
        assert status["steps"] >= landing[0]["tick"] + 20
    finally:
        manager.shutdown()
