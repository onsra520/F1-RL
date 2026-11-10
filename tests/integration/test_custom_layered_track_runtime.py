from pathlib import Path
import json
import time

import anyio
import pytest
from fastapi.testclient import TestClient

from f1rl.local_demo.runtime.api import create_app
from f1rl.sandbox.spatial_contact_dynamics import LAYERED_ROAD_EDGE_LIFECYCLE_V5

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_FIXTURE = ROOT / "tests/runtime/fixtures/m1-development-geometry-v1.json"
WALL_TIMEOUT_S = 240.0
EVENT_TIMEOUT_S = 20.0


async def _receive_message(receiver, timeout_s):
    with anyio.fail_after(timeout_s):
        return await receiver.receive()


def _receive_json(socket, timeout_s):
    message = socket.portal.call(_receive_message, socket._send_rx, timeout_s)
    socket._raise_on_close(message)
    return json.loads(message["text"])


def test_websocket_event_wait_has_a_bounded_timeout():
    async def verify_timeout():
        sender, receiver = anyio.create_memory_object_stream(1)
        try:
            with pytest.raises(TimeoutError):
                await _receive_message(receiver, 0.01)
        finally:
            await sender.aclose()
            await receiver.aclose()

    anyio.run(verify_timeout, backend="asyncio")


def test_layered_crossing_api_streams_to_successful_finish():
    draft = json.loads(GEOMETRY_FIXTURE.read_text(encoding="utf-8"))["fixtures"][0]["track"]

    with TestClient(create_app(repository_root=ROOT)) as client:
        session_id = None
        progress = {"ticks": 0, "last_tick": None, "last_event": None}
        started = time.monotonic()

        def receive(socket):
            remaining = WALL_TIMEOUT_S - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("E2E_WALL_CLOCK_TIMEOUT")
            try:
                event = _receive_json(socket, min(EVENT_TIMEOUT_S, remaining))
            except TimeoutError as error:
                if time.monotonic() - started >= WALL_TIMEOUT_S:
                    raise TimeoutError("E2E_WALL_CLOCK_TIMEOUT") from error
                raise TimeoutError("E2E_STREAM_STALLED") from error
            progress["last_event"] = event.get("type")
            if event.get("type") == "frame":
                progress["ticks"] += 1
                progress["last_tick"] = event["data"].get("tick")
            elif event.get("type") == "error":
                pytest.fail(f"API_WS_FAILURE: {event['data']}; progress={progress}")
            return event

        try:
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
            session_id = status["session_id"]
            assert status["physics"]["ground_motion_version"] == LAYERED_ROAD_EDGE_LIFECYCLE_V5
            assert status["track_preview"]["obstacle_count"] == 2

            streamed_ticks = []
            terminal = None
            with client.websocket_connect(f"/api/v1/sessions/{session_id}/ws") as socket:
                assert receive(socket)["type"] == "ready"
                socket.send_json({
                    "protocol_version": "f1rl_runtime_ws_v1",
                    "type": "speed",
                    "multiplier": 16.0,
                })
                while True:
                    event = receive(socket)
                    if event["type"] == "ack" and event["data"]["command"] == "speed":
                        break
                socket.send_json({
                    "protocol_version": "f1rl_runtime_ws_v1",
                    "type": "start",
                })
                while True:
                    event = receive(socket)
                    if event["type"] == "ack" and event["data"]["command"] == "start":
                        break
                    if event["type"] == "frame":
                        streamed_ticks.append(event["data"]["tick"])
                while terminal is None:
                    event = receive(socket)
                    if event["type"] == "frame":
                        streamed_ticks.append(event["data"]["tick"])
                    elif event["type"] == "terminal":
                        terminal = event["data"]

                if terminal["status"] != "finished" or not terminal["successful_finish"]:
                    pytest.fail(f"PHYSICS_FAILURE: terminal={terminal}; progress={progress}")
                replay = client.get(f"/api/v1/sessions/{session_id}/replay")
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
        except TimeoutError as error:
            detail = "Physics terminal failure" if progress["last_event"] == "terminal" else "stream timeout/stall"
            latest = client.get(f"/api/v1/sessions/{session_id}").json() if session_id else {}
            pytest.fail(
                f"{error}: {detail}; elapsed_s={time.monotonic()-started:.1f}; "
                f"progress={progress}; session_status={latest.get('status')}; "
                f"session_steps={latest.get('steps')}"
            )
        finally:
            if session_id:
                deleted = client.delete(f"/api/v1/sessions/{session_id}")
                assert deleted.status_code in (204, 404), deleted.text
                assert client.get(f"/api/v1/sessions/{session_id}").status_code == 404
