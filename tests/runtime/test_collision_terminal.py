"""Collision is a runtime failure, never a fake accepted Physics frame."""
from pathlib import Path

import pytest

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import convert_custom_track
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.session import RuntimeSessionManager
from src.runtime.development_collision_terminal import inspect_collision_error
from src.simulation.ballistic_flight import FlightPhysicsError


def make_track(catalog, kind="cone"):
    return convert_custom_track({
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "collision-terminal-test",
        "name": "Collision terminal", "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [{"id": kind, "kind": kind, "t": 0.5, "offset": 0, "rotation": 0}],
    }, catalog)


def make_session(manager, track):
    return manager.create(
        model_id=None, track_id=track.track_id, seed=412, mode="auto",
        custom_factory=track.factory, custom_track_sha256=track.track_sha256,
        auto_drive_config=AutoDriveConfig(driving_style="centerline"))


@pytest.mark.parametrize("kind", ["cone", "barrier"])
def test_collision_terminal_is_out_of_band_without_fabricated_step(kind):
    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    converted = make_track(catalog, kind)
    manager = RuntimeSessionManager(catalog)
    try:
        session = make_session(manager, converted)
        session.start()
        assert session.wait(180)
        replay = session.replay()
        terminal, frames = replay["collision_terminal"], replay["frames"]
        assert replay["result"]["status"] == "failed"
        assert replay["result"]["reason"] == "collision"
        assert replay["result"]["valid_sample"] is False
        assert session.status_document()["error"] is None
        assert terminal["obstacle_ids"] == [converted.factory.track.obstacles[0].id]
        assert terminal["last_accepted_tick"] == frames[-1]["tick"]
        assert replay["result"]["steps"] == frames[-1]["tick"]
        assert terminal["witness_interval_s"][0] == pytest.approx(frames[-1]["time_s"])
        assert terminal["witness_interval_s"][1] > terminal["witness_interval_s"][0]
        assert terminal["diagnostic_penalty"] == -5.0
        assert terminal["penalty_kind"] == "runtime_diagnostic_not_gym_reward"
        assert terminal["physics_transition_accepted"] is False
        assert terminal["scientific_evidence"] is False
        assert all(not frame["collision_ids"] for frame in frames)
        assert all(b["tick"] == a["tick"] + 1 for a,b in zip(frames, frames[1:]))
        events=[]
        while (event := session.next_event(timeout=0.01)) is not None:
            events.append(event)
        collisions=[e for e in events if e["type"]=="collision"]
        assert len(collisions)==1
        assert collisions[0]["data"]["ids"]==terminal["obstacle_ids"]
        assert collisions[0]["data"]["physics_transition_accepted"] is False
        assert any(e["type"]=="terminal" and e["data"]["status"]=="failed" for e in events)
    finally:
        manager.shutdown()


def test_failed_collision_reset_starts_fresh_episode():
    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    converted = make_track(catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = make_session(manager, converted)
        session.start()
        assert session.wait(180)
        first=session.replay()
        assert first["result"]["reason"]=="collision"
        session.reset()
        assert session.wait(180)
        second=session.replay()
        assert second["result"]["reason"]=="collision"
        assert second["frames"][0]["tick"]==0
        assert second["frames"]==first["frames"]
        assert second["collision_terminal"]==first["collision_terminal"]
        assert second["result"]["trace_sha256"]==first["result"]["trace_sha256"]
    finally:
        manager.shutdown()


def test_only_known_certified_witness_is_mapped_without_mutating_physics():
    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    converted=make_track(catalog)
    env=converted.factory.create()
    try:
        env.reset(seed=412)
        snapshot=env._simulation.snapshot
        obstacle_id=converted.factory.track.obstacles[0].id
        error=FlightPhysicsError("contact", failure_code="verified_obstacle_contact")
        error.witnessed_obstacle_ids=(obstacle_id,)
        evidence=inspect_collision_error(error, env)
        assert evidence is not None
        assert evidence.document()["diagnostic_penalty"]==-5.0
        assert evidence.document()["physics_transition_accepted"] is False
        assert env._simulation.snapshot is snapshot
        error.witnessed_obstacle_ids=("invalid-id",)
        assert inspect_collision_error(error, env) is None
        uncertain=FlightPhysicsError("uncertain", failure_code="uncertified_physics")
        uncertain.witnessed_obstacle_ids=(obstacle_id,)
        assert inspect_collision_error(uncertain, env) is None
    finally:
        env.close()
