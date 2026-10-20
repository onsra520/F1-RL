from pathlib import Path

import pytest

from f1rl.local_demo import runner


class _Environment:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


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
    assert state["phase"] == "telemetry_finalization"
    assert environment.closed is True
    assert len(revoked) == 1

