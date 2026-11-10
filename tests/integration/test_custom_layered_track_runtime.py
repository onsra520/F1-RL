from pathlib import Path

from fastapi.testclient import TestClient

from f1rl.local_demo.runtime.api import create_app
from f1rl.sandbox.spatial_contact_dynamics import LAYERED_ROAD_EDGE_LIFECYCLE_V5

ROOT = Path(__file__).resolve().parents[2]


def test_layered_crossing_api_streams_to_successful_finish():
    draft = {
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "layered-v5-crossing-obstacles",
        "name": "Layered V5 crossing with obstacles",
        "width": 11,
        "points": [
            {"x": x, "z": z}
            for x, z in ((0, 0), (100, 100), (0, 100), (100, 0), (160, 0))
        ],
        "props": [
            {"id": "approach-barrier", "kind": "barrier", "t": 0.3,
             "offset": 3.6, "rotation": 0},
            {"id": "exit-cone", "kind": "cone", "t": 0.72,
             "offset": -3.6, "rotation": 0},
        ],
    }

    with TestClient(create_app(repository_root=ROOT)) as client:
        validated = client.post("/api/v1/tracks/validate", json={"track": draft})
        assert validated.status_code == 200, validated.text
        track = validated.json()
        assert track["requires_layered_road_support"] is True
        assert track["obstacle_count"] == 2

        created = client.post("/api/v1/sessions", json={
            "mode": "auto",
            "seed": 412,
            "custom_track": draft,
            "physics_profile": "experimental_v5_layered",
        })
        assert created.status_code == 201, created.text
        status = created.json()
        assert status["physics"]["ground_motion_version"] == LAYERED_ROAD_EDGE_LIFECYCLE_V5
        assert status["track_preview"]["obstacle_count"] == 2

        streamed_ticks = []
        terminal = None
        with client.websocket_connect(
                f"/api/v1/sessions/{status['session_id']}/ws") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({
                "protocol_version": "f1rl_runtime_ws_v1",
                "type": "speed",
                "multiplier": 16.0,
            })
            while True:
                event = socket.receive_json()
                if event["type"] == "ack" and event["data"]["command"] == "speed":
                    break
            socket.send_json({
                "protocol_version": "f1rl_runtime_ws_v1",
                "type": "start",
            })
            while True:
                event = socket.receive_json()
                if event["type"] == "ack" and event["data"]["command"] == "start":
                    break
                if event["type"] == "frame":
                    streamed_ticks.append(event["data"]["tick"])
            while terminal is None:
                event = socket.receive_json()
                if event["type"] == "frame":
                    streamed_ticks.append(event["data"]["tick"])
                elif event["type"] == "terminal":
                    terminal = event["data"]

            replay = client.get(f"/api/v1/sessions/{status['session_id']}/replay")
            assert replay.status_code == 200, replay.text
            assert replay.json()["provenance"]["dataset_split"] == "DEVELOPMENT"
            assert replay.json()["provenance"]["track_sha256"] == track["track_sha256"]
            assert replay.json()["result"]["successful_finish"] is True
            assert all(frame["collision_count"] == 0 for frame in replay.json()["frames"])
            assert all(not frame["collision_ids"] for frame in replay.json()["frames"])

        assert streamed_ticks
        assert min(streamed_ticks) == 0
        assert max(streamed_ticks) == terminal["steps"]
        assert terminal["status"] == "finished"
        assert terminal["successful_finish"] is True
        assert terminal["valid_sample"] is False
        assert terminal["steps"] < status["max_steps"]
