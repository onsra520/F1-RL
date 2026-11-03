"""Strict source-backed T3 development acceptance; no synthetic flight frames.

Runs one genuine custom Auto Drive V4 session. This accepts only exact
takeoff -> airborne -> landing -> grounded order and a successful finish.
It is NOT independent scientific Eval approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import convert_custom_track
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.session import RuntimeSessionManager

ROOT = Path(__file__).resolve().parents[2]
V4 = "spatial_road_edge_partial_lifecycle_v4"
RACE_V4 = "wp7_directed_lifecycle_crash_contract_v4"


def default_map() -> dict:
    """Frozen versioned development map: real launch/receiver plus force pads."""
    return json.loads(
        (ROOT / "examples/t3-v4-launch-landing-map.json").read_text(encoding="utf-8")
    )


def check_sequence(frames: list[dict]) -> tuple[dict, list[str]]:
    errors = []
    takeoffs = [i for i, frame in enumerate(frames)
                if "takeoff" in frame["physical_event_types"]]
    landings = [i for i, frame in enumerate(frames)
                if "landing" in frame["physical_event_types"]]
    airs = [i for i, frame in enumerate(frames) if frame["grounded"] is False]
    ticks = [frame["tick"] for frame in frames]
    if not frames or ticks != list(range(len(frames))):
        errors.append("missing or nonmonotonic authoritative runtime frames")
    if not frames or frames[0]["grounded"] is not True:
        errors.append("initial state was not physically grounded")
    if len(takeoffs) != 1:
        errors.append(f"expected one sourced takeoff event, got {len(takeoffs)}")
    if not airs:
        errors.append("no accepted airborne frame")
    if len(landings) != 1:
        errors.append(f"expected one sourced landing event, got {len(landings)}")
    if takeoffs and airs and takeoffs[0] > airs[0]:
        errors.append("airborne appeared before certified takeoff")
    if landings and airs and landings[0] <= airs[0]:
        errors.append("landing appeared before airborne")
    if landings and not any(frame["grounded"] is True
                            for frame in frames[landings[0]:]):
        errors.append("no grounded recovery after certified landing")
    if takeoffs and landings and takeoffs[0] >= landings[0]:
        errors.append("invalid event ordering")
    return {"takeoff_ticks": [ticks[i] for i in takeoffs],
            "landing_ticks": [ticks[i] for i in landings],
            "airborne_ticks": [ticks[i] for i in airs],
            "grounded_after_landing": bool(landings and any(
                f["grounded"] for f in frames[landings[0]:]))}, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=412)
    args = parser.parse_args()
    doc = json.loads(args.map.read_text(encoding="utf-8")) if args.map else default_map()
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(doc, catalog)
    if not any(p["kind"] == "jump" for p in converted.source_document["props"]):
        raise ValueError("T3 reference map must have a versioned launch/receiver and physical gap")
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(model_id=None, track_id=converted.track_id,
            seed=args.seed, mode="auto", custom_factory=converted.factory,
            custom_track_sha256=converted.track_sha256, physics_profile="experimental_v4",
            auto_drive_config=AutoDriveConfig(driving_style="centerline",
               speed_profile="hyper", target_max_speed_kmh=120.0,
               acceleration_factor=1.0, braking_factor=0.8,
               moderate_grade_threshold=0.3, steep_grade_threshold=0.5,
               moderate_grade_speed_mps=15.0, steep_grade_speed_mps=15.0))
        started = session.status_document()
        if started["physics"]["ground_motion_version"] != V4 or (
            started["race_contract"] or {}).get("version") != RACE_V4:
            raise ValueError("source profile was not genuine Physics/Race V4")
        session.start()
        if not session.wait(120):
            session.cancel()
            session.wait(10)
        status = session.status_document()
        replay = session.replay()
        frames = replay["frames"]
        phases, errors = check_sequence(frames)
        if status["status"] != "finished" or status["error"]:
            errors.append(f"session not finished: {status['status']} / {status['error']}")
        if status["episode_result"]["successful_finish"] is not True:
            errors.append(f"no successful finish: {status['episode_result']['reason']}")
        if replay["provenance"]["dataset_split"] != "DEVELOPMENT":
            errors.append("unexpected dataset split")
        evidence = {
            "gate": "T3 Auto Drive Physics V4 live flight",
            "decision": "PASS" if not errors else "BLOCKED",
            "reasons": errors,
            "source_sha256": catalog.source_sha256,
            "track_sha256": converted.track_sha256,
            "physics": started["physics"], "race_contract": started["race_contract"],
            "seed": args.seed, "status": status["status"],
            "error": status["error"], "result": status["episode_result"],
            "frame_count": len(frames), "phases": phases,
            "trace_sha256": status["episode_result"]["trace_sha256"],
            "scientific_evidence": False,
        }
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2))
        return 0 if not errors else 1
    finally:
        manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
