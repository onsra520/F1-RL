from pathlib import Path

import pytest

from src.runtime.catalog import RuntimeCatalog
from src.runtime.custom_tracks import convert_custom_track
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.session import RuntimeSessionManager


@pytest.mark.parametrize("kind", ["cone", "barrier"])
def test_verified_contact_ends_with_collision_telemetry(kind):
    root = Path(__file__).resolve().parents[2]
    catalog = RuntimeCatalog(root)
    track = {
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "collision-terminal-test",
        "name": "Collision terminal", "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [{"id": kind, "kind": kind, "t": 0.5, "offset": 0, "rotation": 0}],
    }
    converted = convert_custom_track(track, catalog)
    manager = RuntimeSessionManager(catalog)
    try:
        session = manager.create(model_id=None, track_id=converted.track_id, seed=412, mode="auto",
                                 custom_factory=converted.factory,
                                 custom_track_sha256=converted.track_sha256,
                                 auto_drive_config=AutoDriveConfig(driving_style="centerline"))
        session.start()
        assert session.wait(180)
        replay = session.replay()
        assert replay["result"]["status"] == "failed"
        assert replay["result"]["reason"] == "collision"
        assert replay["result"]["valid_sample"] is False
        assert replay["result"]["successful_finish"] is False
        assert replay["frames"][-1]["collision_ids"] == [converted.factory.track.obstacles[0].id]
        assert replay["frames"][-1]["collision_count"] == 1
        assert replay["frames"][-1]["tick"] == replay["frames"][-2]["tick"] + 1
        assert replay["frames"][-1]["position"] == replay["frames"][-2]["position"]
        assert replay["frames"][-1]["outcome"]["kind"] == "collision_terminal_development"
        assert replay["frames"][-1]["outcome"]["valid_sample"] is False
        events = []
        while (event := session.next_event(timeout=0.01)) is not None:
            events.append(event)
        collisions = [event for event in events if event["type"] == "collision"]
        assert collisions[-1]["data"]["ids"] == [converted.factory.track.obstacles[0].id]
        assert any(event["type"] == "terminal" and event["data"]["status"] == "failed"
                   for event in events)
    finally:
        manager.shutdown()

def test_dev_terminal_uses_only_configured_collision_penalty_without_fake_physics():
    """Contract mapping unit: real SAT witness is covered by the two E2E tests above."""
    from src.runtime.development_collision_terminal import DevelopmentCollisionTerminalEnv

    catalog = RuntimeCatalog(Path(__file__).resolve().parents[2])
    track = {
        "schema": "f1rl_racing_sandbox_draft_v1", "id": "collision-penalty-test",
        "name": "Collision mapping", "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 120}],
        "props": [{"id": "cone", "kind": "cone", "t": 0.5, "offset": 0, "rotation": 0}],
    }
    converted = convert_custom_track(track, catalog)
    env = DevelopmentCollisionTerminalEnv(converted.factory.create())
    try:
        env.reset(seed=412)
        obstacle_id = converted.factory.track.obstacles[0].id
        _, reward, terminated, truncated, info = env._terminal_on_witness((obstacle_id,))
        assert terminated and not truncated
        assert reward == -5.0
        assert info["reward_components"] == {"route_progress": 0.0, "collision": -5.0, "time": 0.0}
        assert info["collision_terminal"]["physical_impulse_applied"] is False
        assert info["collision_terminal"]["certified_contact_time"] is False
        assert info["research_outcome"]["valid_sample"] is False
        assert info["research_outcome"]["reason"] == "collision"
        assert not info["race"].get("terminal_failure", False)
        assert env.runtime_snapshot()["tick"] == 1
        with pytest.raises(RuntimeError, match="absorbing"):
            env.step(env.action_space.sample())
    finally:
        env.close()
