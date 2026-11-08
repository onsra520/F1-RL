from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from f1rl.local_demo.runtime.api import CreateSessionRequest, create_app


def test_runtime_accepts_development_modes_without_loading_a_checkpoint(monkeypatch):
    root = Path(__file__).parents[2]
    app = create_app(repository_root=root)

    def forbidden_model_load(*_args, **_kwargs):
        raise AssertionError("creating a development session must not load a PPO checkpoint")

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.catalog, "load_model", forbidden_model_load)
        ppo = client.post("/api/v1/sessions", json={
            "mode": "ppo", "model_id": "A-seed0", "track_id": "straight-sprint-v1", "seed": 7,
        })
        assert ppo.status_code == 201
        assert ppo.json()["mode"] == "ppo"
        assert ppo.json()["episode_result"] is None
        assert client.delete(f"/api/v1/sessions/{ppo.json()['session_id']}").status_code == 204

        hybrid = CreateSessionRequest(
            mode="hybrid", model_id="wp7-calibration", track_id="straight-sprint-v1", seed=7,
        )
        assert hybrid.mode == "hybrid"
        assert hybrid.model_id == "wp7-calibration"

        response = client.post("/api/v1/sessions", json={
            "mode": "manual", "track_id": "straight-sprint-v1", "seed": 7,
        })
        assert response.status_code == 201
        session = app.state.sessions.get(response.json()["session_id"])
        session.max_steps = 1
        session.start()
        assert session.wait(timeout=30)
        result = session.status_document()
        assert result["status"] == "finished"
        assert result["provenance"] == "manual-development"
        assert result["episode_result"]["valid_sample"] is False
        assert result["episode_result"]["reason"] == "step_limit"
        replay = session.replay()
        assert replay["provenance"]["dataset_split"] == "TRAIN"
        assert replay["result"]["valid_sample"] is False


def test_custom_track_uses_v4_runtime_adapter_without_training_or_heldout(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "/assets/tracks/evaluation/" in path.as_posix().lower():
            raise PermissionError("HELD-OUT geometry is forbidden in this test")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    app = create_app(repository_root=Path(__file__).parents[2])
    track = {
        "schema": "f1rl_racing_sandbox_draft_v1",
        "id": "integration-smoke",
        "name": "Integration smoke",
        "width": 10,
        "points": [{"x": 0, "z": 0}, {"x": 0, "z": 70}],
        "props": [],
    }
    with TestClient(app) as client:
        validated = client.post("/api/v1/tracks/validate", json={"track": track})
        assert validated.status_code == 200
        preview = validated.json()
        assert preview["development_only"] is True

        created = client.post("/api/v1/sessions", json={
            "mode": "auto",
            "seed": 9,
            "custom_track": track,
            "physics_profile": "experimental_v4",
        })
        assert created.status_code == 201, created.text
        session = app.state.sessions.get(created.json()["session_id"])
        session.max_steps = 1
        session.start()
        assert session.wait(timeout=30)
        status = session.status_document()
        assert status["physics"]["ground_motion_version"] == "spatial_road_edge_partial_lifecycle_v4"
        assert status["race_contract"]["version"] == "wp7_directed_lifecycle_crash_contract_v4"
        assert status["episode_result"]["valid_sample"] is False
        assert status["provenance"] == "custom-development"
