from pathlib import Path

import pytest

from src.runtime.catalog import RuntimeCatalog
from src.runtime.session import (
    MAX_ACTIVE_SESSIONS,
    ManualActionController,
    RuntimeFault,
    RuntimeSessionManager,
)

ROOT = Path(__file__).resolve().parents[2]


def test_manual_action_controller_returns_only_bounded_latest_action():
    controller = ManualActionController()
    action = controller.predict(None)
    assert action.tolist() == [0.0, 0.0, 0.0]
    controller.set_action(-1.0, 1.0, 0.25)
    action = controller.predict(None)
    assert action.tolist() == [-1.0, 1.0, 0.25]
    controller.stop()
    action = controller.predict(None)
    assert action.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        controller.set_action(1.1, 0.0, 0.0)


@pytest.mark.parametrize(("model_id", "track_id"), [
    ("A-seed0", "straight-sprint-v1"),
    ("B-seed1", "slalom-v1"),
])
def test_frozen_model_runs_real_train_environment_and_returns_replay(model_id, track_id):
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        session = manager.create(model_id=model_id, track_id=track_id, seed=412)
        session.start()
        assert session.wait(120.0)

        status = session.status_document()
        replay = session.replay()
        assert status["status"] in {"finished", "failed"}
        assert status["provenance"] == "TRAIN-development"
        assert replay["schema"] == "f1rl_racing_sandbox_replay_v2"
        assert replay["provenance"]["dataset_split"] == "TRAIN"
        assert replay["provenance"]["runtime_mode"] == "development"
        assert replay["provenance"]["model_id"] == model_id
        assert replay["provenance"]["track_id"] == track_id
        assert replay["provenance"]["seed"] == 412
        assert replay["result"]["status"] == status["status"]
        if status["status"] == "finished":
            assert replay["result"]["valid_sample"] is True
        else:
            assert replay["result"]["valid_sample"] is False
            assert replay["result"]["reason"]
        assert len(replay["frames"]) >= 2
        assert replay["frames"][0]["time_s"] == 0.0
        assert replay["frames"][-1]["tick"] == replay["result"]["steps"]
        assert replay["frames"][-1]["position"] != replay["frames"][0]["position"]
    finally:
        manager.shutdown()


def test_fixed_seed_frozen_rollout_trace_is_repeatable():
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        traces = []
        for _ in range(2):
            session = manager.create(
                model_id="A-seed0", track_id="straight-sprint-v1", seed=8841,
            )
            session.start()
            assert session.wait(120.0)
            traces.append(session.replay()["result"]["trace_sha256"])
            manager.delete(session.session_id)
        assert traces[0] == traces[1]
    finally:
        manager.shutdown()


def test_session_manager_bounds_ready_sessions_and_rejects_repeated_start():
    manager = RuntimeSessionManager(RuntimeCatalog(ROOT))
    try:
        sessions = [
            manager.create(model_id="A-seed0", track_id="straight-sprint-v1", seed=seed)
            for seed in range(MAX_ACTIVE_SESSIONS)
        ]
        with pytest.raises(RuntimeFault, match="session limit"):
            manager.create(model_id="A-seed0", track_id="straight-sprint-v1", seed=10)
        assert all(session.status == "ready" for session in sessions)
    finally:
        manager.shutdown()
