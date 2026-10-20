from pathlib import Path
from types import SimpleNamespace

import pytest

from f1rl.local_demo import runner


class _Environment:
    def __init__(self, *, close_error=False):
        self.closed = False
        self.close_error = close_error

    def close(self):
        self.closed = True
        if self.close_error:
            raise OSError("environment close failed")


class _Model:
    def __init__(self, environment):
        self._model = type("Backend", (), {"get_env": lambda _self: environment})()


class _Telemetry:
    def __init__(self, *, close_error=False, manifest_error=False):
        self.close_error = close_error
        self.manifest_error = manifest_error
        self.closed = False

    def close(self):
        if self.close_error:
            raise OSError("flush failed")
        self.closed = True

    def manifest(self, *, complete):
        if self.manifest_error:
            raise ValueError("manifest failed")
        return {"complete": complete}


@pytest.mark.parametrize(
    ("close_error", "manifest_error", "expected_error"),
    [(True, False, OSError), (False, True, ValueError)],
)
def test_local_demo_finalization_failure_marks_run_failed_and_releases_resources(
    tmp_path, monkeypatch, close_error, manifest_error, expected_error
):
    environment = _Environment()
    model = _Model(environment)
    telemetry = _Telemetry(close_error=close_error, manifest_error=manifest_error)
    revoked = []
    monkeypatch.setattr(runner, "_revoke", lambda vector: revoked.append(vector))
    state = {"status": "learning_completed", "phase": "complete"}

    with pytest.raises(expected_error):
        runner._finalize_run(
            tmp_path,
            state,
            telemetry=telemetry,
            model=model,
            vector=object(),
            session=object(),
            learning_completed=True,
        )

    persisted = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in persisted
    assert state["phase"] == "finalization_failed"
    assert environment.closed is True
    assert len(revoked) == 1


def test_local_demo_keeps_training_error_primary_when_telemetry_cleanup_fails(
    tmp_path, monkeypatch
):
    environment = _Environment()
    vector = object()
    telemetry = _Telemetry(close_error=True)

    class FailingModel(_Model):
        num_timesteps = 0

        def weights_sha256(self):
            return "f" * 64

        def learn(self, *args, **kwargs):
            raise RuntimeError("training failed")

    model = FailingModel(environment)
    preflight = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "experiment_manifest_sha256": "c" * 64,
        "candidate": {"candidate_id": "C", "demo_contract": "straight_route_stability_demo_v2"},
        "demo_reward": {"version": "straight_route_stability_demo_v2"},
        "scenarios": object(),
        "ppo": SimpleNamespace(seed=0),
        "timesteps": 65536,
        "checkpoint_every": 4096,
        "max_wall_seconds": 7200,
        "threads": 8,
        "interop_threads": 1,
        "sampling_seed": 0,
        "reset_seed": 1000,
        "track_rows": [],
        "reset_contracts": {},
        "registration_sha256": "d" * 64,
        "track_manifest_sha256": "e" * 64,
        "physics_sha256": "1" * 64,
        "observation_schema": [],
        "action_contract_version": "v1",
        "ppo_config": {},
        "action_space": {"shape": [3], "low": [-1, -1, -1], "high": [1, 1, 1]},
        "observation_space": {"shape": [2], "low": [-1, -1], "high": [1, 1]},
    }
    monkeypatch.setattr(runner, "_preflight", lambda *args: preflight)
    monkeypatch.setattr(runner, "configure_cpu_runtime", lambda *args: None)
    monkeypatch.setattr(runner, "make_multi_track_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner, "_install_demo_reward", lambda *args: None)
    monkeypatch.setattr(runner, "_issue", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner.PPOModel, "create", staticmethod(lambda *args: model))
    monkeypatch.setattr(runner, "TrainingTelemetryRecorder", lambda *args, **kwargs: telemetry)
    revoked = []
    monkeypatch.setattr(runner, "_revoke", lambda value: revoked.append(value))
    original_write_json = runner._write_json
    fail_failure_state_write = True

    def write_json_with_one_failed_failure_state(path, value):
        nonlocal fail_failure_state_write
        if path.name == "state.json" and value.get("status") == "failed" and fail_failure_state_write:
            fail_failure_state_write = False
            raise OSError("failure state write failed")
        original_write_json(path, value)

    monkeypatch.setattr(runner, "_write_json", write_json_with_one_failed_failure_state)

    with pytest.raises(RuntimeError, match="training failed") as raised:
        runner.run_local_demo(
            Path(__file__).parents[2],
            candidate_id="C",
            tuning_seed_index=0,
            output_directory=tmp_path / "run",
        )

    state = __import__("json").loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error"] == "RuntimeError: training failed"
    assert state["finalization_error"] == "OSError: flush failed"
    assert environment.closed is True
    assert revoked == [vector]
    assert any("failure state write failed" in note for note in raised.value.__notes__)


def test_local_demo_finalization_attempts_all_cleanup_stages_and_reports_each(tmp_path, monkeypatch):
    environment = _Environment(close_error=True)
    telemetry = _Telemetry(close_error=True)
    revoked = []
    vector = object()
    monkeypatch.setattr(runner, "_revoke", lambda value: revoked.append(value))
    state = {"status": "learning_completed", "phase": "complete"}

    with pytest.raises(OSError, match="flush failed"):
        runner._finalize_run(
            tmp_path,
            state,
            telemetry=telemetry,
            model=_Model(environment),
            vector=vector,
            session=object(),
            learning_completed=True,
        )

    persisted = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "OSError: flush failed" in persisted
    assert "OSError: environment close failed" in persisted
    assert state["status"] == "failed"
    assert environment.closed is True
    assert revoked == [vector]


def test_local_demo_admission_cannot_be_minted_by_direct_caller():
    with pytest.raises(PermissionError, match="only be issued by its runner"):
        runner._issue(
            Path(__file__).parents[2],
            object(),
            commit="a" * 40,
            tree_sha256="b" * 64,
            deadline=1.0,
            max_timesteps=1,
        )

