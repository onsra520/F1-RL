import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.rl.pilot_preflight import TRAIN_IDS
from src.runtime.api import create_app

ROOT = Path(__file__).resolve().parents[2]


def _custom_track():
    return {
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "sandbox-e2e",
        "name": "Runtime integration road",
        "width": 12,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 50}],
        "props": [{"id": "cp-e2e", "kind": "checkpoint", "t": 0.55, "offset": 0, "rotation": 0}],
    }


def test_runtime_http_catalog_is_development_only_and_train_only():
    with TestClient(create_app(repository_root=ROOT)) as client:
        health = client.get("/api/v1/health")
        models = client.get("/api/v1/models")
        tracks = client.get("/api/v1/tracks")

    assert health.status_code == 200
    assert health.json()["runtime_mode"] == "development"
    assert health.json()["scientific_admission"] == "disabled"
    assert len(models.json()["models"]) == 12
    assert all(re.fullmatch(r"[0-9a-f]{64}", model["sha256"]) for model in models.json()["models"])
    assert tuple(track["track_id"] for track in tracks.json()["tracks"]) == TRAIN_IDS
    assert all(track["split"] == "TRAIN" and track["purpose"] == "development" for track in tracks.json()["tracks"])
    preview = tracks.json()["tracks"][0]["preview"]
    assert preview["schema"] == "f1rl_racing_sandbox_draft_v1"
    assert len(preview["points"]) >= 2
    samples = tracks.json()["tracks"][0]["road_samples"]
    assert len(samples) >= len(preview["points"])
    assert (samples[0]["x"], samples[0]["z"]) == (preview["points"][0]["x"], preview["points"][0]["z"])
    assert (samples[-1]["x"], samples[-1]["z"]) == (preview["points"][-1]["x"], preview["points"][-1]["z"])


def test_runtime_http_validates_session_ids_and_has_structured_errors():
    with TestClient(create_app(repository_root=ROOT)) as client:
        invalid = client.post("/api/v1/sessions", json={
            "model_id": "../models/other.zip",
            "track_id": "straight-sprint-v1",
            "seed": 0,
        })
        extra = client.post("/api/v1/sessions", json={
            "model_id": "A-seed0",
            "track_id": "straight-sprint-v1",
            "seed": 0,
            "model_path": "C:/anything.zip",
        })
        created = client.post("/api/v1/sessions", json={
            "model_id": "A-seed0",
            "track_id": "straight-sprint-v1",
            "seed": 412,
        })
        session_id = created.json()["session_id"]
        status = client.get(f"/api/v1/sessions/{session_id}")
        deleted = client.delete(f"/api/v1/sessions/{session_id}")
        missing = client.get(f"/api/v1/sessions/{session_id}")

    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_model"
    assert extra.status_code == 422
    assert extra.json()["error"]["code"] == "invalid_request"
    assert created.status_code == 201
    assert status.json()["status"] == "ready"
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_runtime_websocket_v1_rejects_manual_action_on_ppo_and_disconnect_cleans_session():
    with TestClient(create_app(repository_root=ROOT)) as client:
        created = client.post("/api/v1/sessions", json={
            "model_id": "A-seed0", "track_id": "straight-sprint-v1", "seed": 412,
        })
        session_id = created.json()["session_id"]
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/ws") as socket:
            ready = socket.receive_json()
            assert ready["protocol_version"] == "f1rl_runtime_ws_v1"
            assert ready["type"] == "ready"
            socket.send_json({
                "protocol_version": "f1rl_runtime_ws_v1", "type": "action",
                "steering": 0, "throttle": 0, "brake": 0,
            })
            rejected = socket.receive_json()
            assert rejected["type"] == "error"
            assert rejected["data"]["code"] == "invalid_controller"
        # The ASGI disconnect handler runs concurrently with TestClient's close.
        # Require the actual cleanup within a strict bound rather than racing it.
        deadline = time.monotonic() + 2.0
        while True:
            missing = client.get(f"/api/v1/sessions/{session_id}")
            if missing.status_code == 404 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
    assert missing.status_code == 404


def test_live_ppo_websocket_streams_frames_and_supports_pause_resume_stop():
    with TestClient(create_app(repository_root=ROOT)) as client:
        created = client.post("/api/v1/sessions", json={
            "model_id": "A-seed0", "track_id": "straight-sprint-v1", "seed": 412,
        })
        session_id = created.json()["session_id"]
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/ws") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "start"})
            frame = None
            for _ in range(100):
                event = socket.receive_json()
                if event["type"] == "terminal":
                    pytest.fail(f"live PPO ended before streaming a frame: {event['data']}")
                if event["type"] == "frame" and event["data"]["tick"] >= 2:
                    frame = event["data"]
                    break
            assert frame is not None
            assert frame["schema_version"] == "f1rl_runtime_frame_v1"
            assert frame["tick"] >= 2 and len(frame["position"]) == 3
            assert frame["action"]["throttle"] >= 0

            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "pause"})
            assert client.get(f"/api/v1/sessions/{session_id}").json()["status"] == "paused"
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "reset"})
            reset_frame = None
            for _ in range(100):
                event = socket.receive_json()
                if event["type"] == "frame" and event["data"]["tick"] == 0:
                    reset_frame = event["data"]
                    break
            assert reset_frame is not None and reset_frame["time_s"] == 0

            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "pause"})
            assert client.get(f"/api/v1/sessions/{session_id}").json()["status"] == "paused"
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "resume"})
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "stop"})
            terminal = None
            for _ in range(100):
                event = socket.receive_json()
                if event["type"] == "terminal":
                    terminal = event["data"]
                    break
            assert terminal is not None and terminal["status"] == "cancelled"
            replay = client.get(f"/api/v1/sessions/{session_id}/replay")
            assert replay.status_code == 200
            assert replay.json()["frames"][0]["tick"] == 0
            assert replay.json()["result"]["status"] == "cancelled"


def test_manual_runtime_session_uses_same_gym_track_and_accepts_typed_actions():
    with TestClient(create_app(repository_root=ROOT)) as client:
        created = client.post("/api/v1/sessions", json={
            "mode": "manual", "track_id": "straight-sprint-v1", "seed": 73,
        })
        assert created.status_code == 201
        assert created.json()["mode"] == "manual"
        assert created.json()["model_id"] is None
        assert created.json()["provenance"] == "manual-development"
        session_id = created.json()["session_id"]
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/ws") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "start"})
            socket.send_json({
                "protocol_version": "f1rl_runtime_ws_v1", "type": "action",
                "steering": -0.5, "throttle": 1.0, "brake": 0.0,
            })
            received = [socket.receive_json() for _ in range(4)]
            assert any(event.get("type") == "ack" and event["data"].get("command") == "action" for event in received)
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "stop"})
            for _ in range(6):
                if any(event.get("type") == "ack" and event["data"].get("command") == "stop" for event in received):
                    break
                received.append(socket.receive_json())
    assert created.json()["seed"] == 73


def test_auto_runtime_session_is_registered_without_a_ppo_model():
    with TestClient(create_app(repository_root=ROOT)) as client:
        created = client.post("/api/v1/sessions", json={
            "mode": "auto", "track_id": "straight-sprint-v1", "seed": 412,
        })
        assert created.status_code == 201
        assert created.json()["mode"] == "auto"
        assert created.json()["model_id"] is None
        assert created.json()["provenance"] == "auto-development"

        invalid = client.post("/api/v1/sessions", json={
            "mode": "auto", "model_id": "A-seed0", "track_id": "straight-sprint-v1", "seed": 412,
        })
        assert invalid.status_code == 422
        assert client.delete(f"/api/v1/sessions/{created.json()['session_id']}").status_code == 204


def test_custom_track_validation_and_manual_simulation_use_python_gym():
    with TestClient(create_app(repository_root=ROOT)) as client:
        document = _custom_track()
        validation = client.post("/api/v1/tracks/validate", json={"track": document})
        assert validation.status_code == 200
        converted = validation.json()
        assert converted["schema_version"] == "f1rl_custom_track_conversion_v1"
        assert converted["development_only"] is True
        assert converted["ppo_compatible"] is True
        assert converted["checkpoint_count"] == 1

        invalid = client.post("/api/v1/tracks/validate", json={"track": {
            **document, "props": [{"id": "bad", "kind": "checkpoint", "t": 0.01, "offset": 0, "rotation": 0}],
        }})
        malformed_prop = client.post("/api/v1/tracks/validate", json={"track": {
            **document, "props": [{"id": "bad-kind", "kind": [], "t": 0.5, "offset": 0, "rotation": 0}],
        }})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "checkpoint_order"
        assert malformed_prop.status_code == 422
        assert malformed_prop.json()["error"]["code"] == "invalid_props"
        unsupported = client.post("/api/v1/tracks/validate", json={"track": {
            **document,
            "points": [
                {"x": 0, "z": 0}, {"x": 20, "z": 20}, {"x": 0, "z": 20},
                {"x": 20, "z": 0}, {"x": 30, "z": 0},
            ],
        }})
        assert unsupported.status_code == 422
        assert unsupported.json()["error"]["code"] == "unsupported_intersection"

        created = client.post("/api/v1/sessions", json={
            "mode": "manual", "custom_track": document, "seed": 17,
        })
        assert created.status_code == 201
        assert created.json()["track_id"] == converted["track_id"]
        session_id = created.json()["session_id"]
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/ws") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "start"})
            while True:
                event = socket.receive_json()
                if event["type"] == "terminal":
                    pytest.fail(f"custom Gym session failed before streaming frames: {event['data']}")
                if event["type"] == "frame" and event["data"]["tick"] == 0:
                    break
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "action",
                "steering": 0.0, "throttle": 1.0, "brake": 0.0})
            observed_steps = 0
            for _ in range(60):
                event = socket.receive_json()
                if event["type"] == "terminal":
                    pytest.fail(f"custom Gym session ended before live state was streamed: {event['data']}")
                if event["type"] == "frame":
                    observed_steps = max(observed_steps, event["data"]["tick"])
                    if observed_steps >= 8:
                        break
            assert observed_steps >= 8
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "stop"})
            terminal = None
            for _ in range(40):
                event = socket.receive_json()
                if event["type"] == "terminal":
                    terminal = event["data"]
                    break
            assert terminal is not None and terminal["status"] == "cancelled"
            replay = client.get(f"/api/v1/sessions/{session_id}/replay")
            assert replay.status_code == 200
            assert replay.json()["provenance"]["dataset_split"] == "DEVELOPMENT"
            assert replay.json()["provenance"]["model_id"] == "manual"
            assert replay.json()["provenance"]["track_sha256"] == converted["track_sha256"]


def test_ppo_runs_on_custom_track_with_development_provenance():
    with TestClient(create_app(repository_root=ROOT)) as client:
        document = _custom_track()
        created = client.post("/api/v1/sessions", json={
            "mode": "ppo", "model_id": "A-seed0", "custom_track": document, "seed": 29,
        })
        assert created.status_code == 201
        assert created.json()["provenance"] == "custom-development"
        assert created.json()["track_preview"]["ppo_compatible"] is True
        session_id = created.json()["session_id"]
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/ws") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "start"})
            frame = None
            for _ in range(100):
                event = socket.receive_json()
                if event["type"] == "terminal":
                    pytest.fail(f"custom PPO ended before streaming a frame: {event['data']}")
                if event["type"] == "frame" and event["data"]["tick"] >= 1:
                    frame = event["data"]
                    break
            assert frame is not None and frame["schema_version"] == "f1rl_runtime_frame_v1"
            socket.send_json({"protocol_version": "f1rl_runtime_ws_v1", "type": "stop"})
            terminal = None
            for _ in range(100):
                event = socket.receive_json()
                if event["type"] == "terminal":
                    terminal = event["data"]
                    break
            assert terminal is not None and terminal["status"] == "cancelled"
            replay = client.get(f"/api/v1/sessions/{session_id}/replay").json()
            assert replay["provenance"]["dataset_split"] == "DEVELOPMENT"
            assert replay["provenance"]["model_id"] == "A-seed0"
            assert replay["provenance"]["track_sha256"] == created.json()["track_preview"]["track_sha256"]


def test_runtime_cors_allows_only_local_editor_ports():
    with TestClient(create_app(repository_root=ROOT)) as client:
        for origin in ("http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://localhost:5174"):
            response = client.get("/api/v1/health", headers={"origin": origin})
            assert response.status_code == 200
            assert response.headers.get("access-control-allow-origin") == origin
        external = client.get("/api/v1/health", headers={"origin": "https://other.example"})
        assert external.status_code == 200
        assert "access-control-allow-origin" not in external.headers
        preflight = client.options("/api/v1/sessions", headers={
            "origin": "http://127.0.0.1:5173",
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type,x-trace-id",
        })
        assert preflight.status_code == 200
        assert "x-trace-id" in preflight.headers["access-control-allow-headers"].lower()


def test_auto_drive_config_api_is_validated_and_snapshot_is_immutable():
    from src.runtime.drive_config import AutoDriveConfig
    with TestClient(create_app(repository_root=ROOT)) as client:
        payload = {"mode": "auto", "track_id": "straight-sprint-v1", "seed": 412,
                   "auto_drive_config": {"driving_style": "racing", "speed_profile": "hyper",
                                         "target_max_speed_kmh": 240.0}}
        created = client.post("/api/v1/sessions", json=payload)
        assert created.status_code == 201, created.text
        status = created.json()
        assert status["auto_drive_config"]["driving_style"] == "racing"
        assert status["auto_drive_config"]["target_max_speed_kmh"] == 240.0
        assert status["effective_max_speed_kmh"] <= 3.6 * 15.0 + 1e-6
        payload["auto_drive_config"]["driving_style"] = "right"
        repeated = client.get(f"/api/v1/sessions/{status['session_id']}").json()
        assert repeated["auto_drive_config"]["driving_style"] == "racing"
        assert client.delete(f"/api/v1/sessions/{status['session_id']}").status_code == 204
        for bad in ({"target_max_speed_kmh": -1.0}, {"target_max_speed_kmh": 999.0},
                    {"min_lookahead_m": 20.0}, {"edge_margin_m": -0.1}, {"extra_override": True}):
            failed = client.post("/api/v1/sessions", json={"mode": "auto", "track_id": "straight-sprint-v1",
                "seed": 1, "auto_drive_config": bad})
            assert failed.status_code == 422
        ppo = client.post("/api/v1/sessions", json={"mode": "ppo", "model_id": "A-seed0",
            "track_id": "straight-sprint-v1", "seed": 1, "auto_drive_config": {"driving_style": "right"}})
        assert ppo.status_code == 422
        assert AutoDriveConfig().driving_style == "right"
