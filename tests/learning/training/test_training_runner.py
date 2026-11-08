import json
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import f1rl.learning.training.runner as runner_module
from f1rl.learning.algorithms.ppo import PPOConfig
from f1rl.learning.training.config import TrainingRunConfig, VectorEnvConfig
from f1rl.learning.training.runner import (
    TrainingRunResult,
    TrainingRunner,
    _advance_checkpoint_schedule,
    _artifact_stem,
    _build_run_id,
    _checkpoint_archive_path,
    _checkpoint_thresholds,
    _prepare_run,
    _write_state,
)

from tests.learning.training.mocks.mock_factories import make_factory, make_scenario_set
from tests.learning.training.mocks.mock_io import FailingStateWriter
from tests.learning.training.mocks.mock_ppo import MockPPOModel
from tests.learning.training.mocks.mock_provenance import make_provenance
from tests.learning.training.mocks.mock_vec_env import MockVecEnv


def _config(
    root: Path,
    *,
    seed: int | None = 42,
    sampling_seed: int = 123,
    total_timesteps: int = 50_000,
    checkpoint_every_steps: int | None = 25_000,
) -> TrainingRunConfig:
    return TrainingRunConfig(
        ppo=PPOConfig(seed=seed),
        vector_env=VectorEnvConfig(n_envs=2),
        total_timesteps=total_timesteps,
        checkpoint_every_steps=checkpoint_every_steps,
        sampling_seed=sampling_seed,
        artifacts_root=root,
    )


def _started_at() -> datetime:
    return datetime(2026, 9, 30, 1, 5, 0)


def test_checkpoint_thresholds_are_strictly_before_requested_final() -> None:
    assert _checkpoint_thresholds(50_000, 25_000) == (25_000,)
    assert _checkpoint_thresholds(60_000, 25_000) == (25_000, 50_000)
    assert _checkpoint_thresholds(10_000, 25_000) == ()
    assert _checkpoint_thresholds(50_000, None) == ()


def test_run_and_artifact_names_encode_mode_and_seed_ownership() -> None:
    started = _started_at()

    assert _artifact_stem("singletrack", 42, None) == "ppo-singletrack-seed42"
    assert _build_run_id(started, "singletrack", 42, None) == (
        "2026-09-30_010500_ppo-singletrack-seed42"
    )
    assert _artifact_stem("multitrack", 42, 123) == "ppo-multitrack-seed42-sample123"
    assert _build_run_id(started, "multitrack", 42, 123) == (
        "2026-09-30_010500_ppo-multitrack-seed42-sample123"
    )
    assert _artifact_stem("singletrack", None, None) == "ppo-singletrack-seednone"


def test_checkpoint_progress_coalesces_multiple_crossed_thresholds() -> None:
    thresholds = (25_000, 50_000)

    due, next_index = _advance_checkpoint_schedule(thresholds, 0, 51_200)
    assert due is True
    assert next_index == 2

    due_again, unchanged_index = _advance_checkpoint_schedule(thresholds, next_index, 51_200)
    assert due_again is False
    assert unchanged_index == 2


def test_prepare_single_track_run_writes_exact_initial_state(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")

    artifacts = _prepare_run(
        config,
        started_at=_started_at(),
        mode="singletrack",
        training_track_ids=(),
    )
    state = json.loads(artifacts.state_path.read_text(encoding="utf-8"))

    assert artifacts.run_id == "2026-09-30_010500_ppo-singletrack-seed42"
    assert artifacts.run_directory.is_dir()
    assert artifacts.checkpoint_directory.is_dir()
    assert artifacts.final_model_path.name == "ppo-singletrack-seed42-final-model.zip"
    assert state == {
        "run_id": artifacts.run_id,
        "status": "running",
        "mode": "singletrack",
        "requested_total_timesteps": 50_000,
        "actual_total_timesteps": 0,
        "checkpoint_every_steps": 25_000,
        "latest_checkpoint": None,
        "final_model": None,
        "final_model_cleanup_error": None,
        "ppo": {
            "learning_rate": 0.0003,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "seed": 42,
            "device": "cpu",
        },
        "vector_env": {
            "n_envs": 2,
            "backend": "dummy",
            "start_method": "spawn",
        },
        "sampling_seed": None,
        "training_tracks": [],
        "provenance_file": None,
    }


def test_prepare_multi_track_run_records_declared_track_ids(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")

    artifacts = _prepare_run(
        config,
        started_at=_started_at(),
        mode="multitrack",
        training_track_ids=("alpha", "beta"),
    )
    state = json.loads(artifacts.state_path.read_text(encoding="utf-8"))

    assert artifacts.run_id.endswith("ppo-multitrack-seed42-sample123")
    assert artifacts.final_model_path.name == "ppo-multitrack-seed42-sample123-final-model.zip"
    assert state["sampling_seed"] == 123
    assert state["training_tracks"] == ["alpha", "beta"]


def test_prepare_run_rejects_existing_run_directory_without_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")
    first = _prepare_run(
        config,
        started_at=_started_at(),
        mode="singletrack",
        training_track_ids=(),
    )
    original = first.state_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        _prepare_run(
            config,
            started_at=_started_at(),
            mode="singletrack",
            training_track_ids=(),
        )

    assert first.state_path.read_text(encoding="utf-8") == original


def test_prepare_run_rejects_artifacts_root_file_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.write_text("keep", encoding="utf-8")
    config = _config(root)

    with pytest.raises(ValueError):
        _prepare_run(
            config,
            started_at=_started_at(),
            mode="singletrack",
            training_track_ids=(),
        )

    assert root.read_text(encoding="utf-8") == "keep"


def test_state_write_keeps_previous_json_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    previous_state = {"status": "running"}
    state_path.write_text(json.dumps(previous_state), encoding="utf-8")
    error = OSError("replace failure")

    def fail_replace(source: Path, destination: Path) -> None:
        raise error

    monkeypatch.setattr(
        runner_module,
        "os",
        SimpleNamespace(replace=fail_replace),
        raising=False,
    )

    with pytest.raises(OSError) as caught:
        _write_state(state_path, {"status": "completed"})

    assert caught.value is error
    assert json.loads(state_path.read_text(encoding="utf-8")) == previous_state
    assert not state_path.with_name("state.json.tmp").exists()


def test_state_write_removes_temporary_file_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "state.json"
    previous_state = {"status": "running"}
    state_path.write_text(json.dumps(previous_state), encoding="utf-8")
    error = OSError("write failure")
    original_write_text = Path.write_text

    def write_partial_temporary_file(path: Path, text: str, **kwargs: object) -> int:
        if path.name == "state.json.tmp":
            original_write_text(path, "{", **kwargs)
            raise error
        return original_write_text(path, text, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_partial_temporary_file)

    with pytest.raises(OSError) as caught:
        _write_state(state_path, {"status": "completed"})

    assert caught.value is error
    assert json.loads(state_path.read_text(encoding="utf-8")) == previous_state
    assert not state_path.with_name("state.json.tmp").exists()


def test_checkpoint_archive_path_uses_actual_timestep_once(tmp_path: Path) -> None:
    artifacts = _prepare_run(
        _config(tmp_path / "runs"),
        started_at=_started_at(),
        mode="multitrack",
        training_track_ids=("alpha",),
    )

    archive = _checkpoint_archive_path(artifacts, 51_200)

    assert archive == (
        artifacts.checkpoint_directory
        / "ppo-multitrack-seed42-sample123-step51200.zip"
    )


def test_training_run_result_is_frozen(tmp_path: Path) -> None:
    result = TrainingRunResult(
        run_id="run",
        run_directory=tmp_path,
        final_model_path=tmp_path / "model.zip",
        checkpoint_paths=(),
        requested_total_timesteps=8,
        actual_total_timesteps=8,
    )
    with pytest.raises(FrozenInstanceError):
        result.actual_total_timesteps = 9  # type: ignore[misc]


def test_run_single_writes_checkpoint_final_state_and_closes_vec_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    vector = MockVecEnv()
    factory = make_factory()
    calls: list[tuple[object, VectorEnvConfig]] = []

    def build_vector(actual_factory: object, config: VectorEnvConfig) -> MockVecEnv:
        calls.append((actual_factory, config))
        return vector

    monkeypatch.setattr(runner_module, "make_vector_env", build_vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    config = _config(tmp_path / "runs")
    result = TrainingRunner(config).run_single(factory)

    assert calls == [(factory, config.vector_env)]
    assert vector.closed is True
    assert vector.close_calls == 1
    assert result.run_id == "2026-09-30_010500_ppo-singletrack-seed42"
    assert result.requested_total_timesteps == 50_000
    assert result.actual_total_timesteps == 51_200
    assert len(result.checkpoint_paths) == 1
    checkpoint = result.checkpoint_paths[0]
    assert checkpoint.name == "ppo-singletrack-seed42-step25008.zip"
    assert checkpoint.is_file()
    assert result.final_model_path.name == "ppo-singletrack-seed42-final-model.zip"
    assert result.final_model_path.is_file()

    model = MockPPOModel.instances[-1]
    assert model.learn_calls == [50_000]
    assert model.save_calls == [checkpoint, result.final_model_path]

    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["actual_total_timesteps"] == 51_200
    assert state["latest_checkpoint"] == "checkpoints/ppo-singletrack-seed42-step25008.zip"
    assert state["final_model"] == "ppo-singletrack-seed42-final-model.zip"
    assert state["sampling_seed"] is None
    assert state["training_tracks"] == []


def test_run_single_seed_none_uses_seednone_and_serializable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    vector = MockVecEnv()
    monkeypatch.setattr(runner_module, "make_vector_env", lambda factory, config: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    result = TrainingRunner(_config(tmp_path / "runs", seed=None)).run_single(make_factory())
    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))

    assert "seednone" in result.run_id
    assert "seednone" in result.final_model_path.name
    assert all("seednone" in path.name for path in result.checkpoint_paths)
    assert state["ppo"]["seed"] is None
    assert state["sampling_seed"] is None
    assert state["training_tracks"] == []
    json.dumps(state)


def test_run_single_rejects_non_factory_before_creating_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    runner = TrainingRunner(_config(root))

    with pytest.raises(ValueError):
        runner.run_single(object())  # type: ignore[arg-type]

    assert not root.exists()


def test_run_multi_derives_track_ids_and_uses_both_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    vector = MockVecEnv()
    scenarios = make_scenario_set()
    calls: list[tuple[object, VectorEnvConfig, int]] = []

    def build_multi(actual_scenarios: object, config: VectorEnvConfig, sampling_seed: int) -> MockVecEnv:
        calls.append((actual_scenarios, config, sampling_seed))
        return vector

    monkeypatch.setattr(runner_module, "make_multi_track_vector_env", build_multi, raising=False)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    config = _config(tmp_path / "runs", seed=42, sampling_seed=123)
    result = TrainingRunner(config).run_multi(scenarios)

    assert calls == [(scenarios, config.vector_env, 123)]
    assert result.run_id == "2026-09-30_010500_ppo-multitrack-seed42-sample123"
    assert all("seed42-sample123" in path.name for path in (*result.checkpoint_paths, result.final_model_path))
    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["sampling_seed"] == 123
    assert state["training_tracks"] == ["alpha", "beta", "gamma"]

def test_run_multi_coalesces_crossed_thresholds_at_actual_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    MockPPOModel.progress_points = (51_200, 61_440)
    vector = MockVecEnv()
    monkeypatch.setattr(runner_module, "make_multi_track_vector_env", lambda *args: vector, raising=False)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    result = TrainingRunner(
        _config(tmp_path / "runs", total_timesteps=60_000, checkpoint_every_steps=25_000)
    ).run_multi(make_scenario_set())

    assert [path.name for path in result.checkpoint_paths] == [
        "ppo-multitrack-seed42-sample123-step51200.zip"
    ]
    model = MockPPOModel.instances[-1]
    assert model.save_calls == [result.checkpoint_paths[0], result.final_model_path]
    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["latest_checkpoint"] == "checkpoints/ppo-multitrack-seed42-sample123-step51200.zip"
    assert state["actual_total_timesteps"] == 61_440


def test_run_single_records_actual_200704_final_step_without_live_ppo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    MockPPOModel.progress_points = (200_704,)
    vector = MockVecEnv()
    monkeypatch.setattr(runner_module, "make_vector_env", lambda factory, config: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    config = _config(
        tmp_path / "runs",
        total_timesteps=200_000,
        checkpoint_every_steps=None,
    )
    config = TrainingRunConfig(
        ppo=config.ppo,
        vector_env=VectorEnvConfig(n_envs=8),
        total_timesteps=config.total_timesteps,
        checkpoint_every_steps=config.checkpoint_every_steps,
        sampling_seed=config.sampling_seed,
        artifacts_root=config.artifacts_root,
    )
    result = TrainingRunner(config).run_single(make_factory())

    assert result.requested_total_timesteps == 200_000
    assert result.actual_total_timesteps == 200_704
    assert result.checkpoint_paths == ()
    assert result.final_model_path.is_file()
    assert MockPPOModel.instances[-1].learn_calls == [200_000]
    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["requested_total_timesteps"] == 200_000
    assert state["actual_total_timesteps"] == 200_704
    assert state["final_model"] == "ppo-singletrack-seed42-final-model.zip"
    assert vector.closed is True

def test_run_failure_marks_state_failed_and_closes_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    error = RuntimeError("learn failure")
    MockPPOModel.learn_error = error
    vector = MockVecEnv()
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    root = tmp_path / "runs"
    with pytest.raises(RuntimeError) as caught:
        TrainingRunner(_config(root)).run_single(make_factory())

    assert caught.value is error
    run_directory = next(root.iterdir())
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["actual_total_timesteps"] == 0
    assert state["final_model"] is None
    assert vector.closed is True

def test_checkpoint_save_failure_keeps_latest_state_truthful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    MockPPOModel.save_error_at_call = 1
    vector = MockVecEnv()
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="mock save failure"):
        TrainingRunner(_config(root)).run_single(make_factory())

    run_directory = next(root.iterdir())
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["actual_total_timesteps"] == 25_008
    assert state["latest_checkpoint"] is None
    assert state["final_model"] is None
    assert list((run_directory / "checkpoints").glob("*.zip")) == []
    assert vector.closed is True

def test_state_update_failure_marks_failed_and_preserves_checkpoint_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SentinelError(RuntimeError):
        pass

    MockPPOModel.reset()
    vector = MockVecEnv()
    original_write_state = runner_module._write_state
    writer = FailingStateWriter(original_write_state, fail_on_call=2, error=SentinelError("state failure"))
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_write_state", writer)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    root = tmp_path / "runs"
    with pytest.raises(SentinelError, match="state failure"):
        TrainingRunner(_config(root)).run_single(make_factory())

    run_directory = next(root.iterdir())
    checkpoints = list((run_directory / "checkpoints").glob("*.zip"))
    assert [path.name for path in checkpoints] == ["ppo-singletrack-seed42-step25008.zip"]
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["actual_total_timesteps"] == 25_008
    assert state["latest_checkpoint"] == "checkpoints/ppo-singletrack-seed42-step25008.zip"
    assert state["final_model"] is None
    assert vector.closed is True


def test_completed_state_write_failure_removes_uncommitted_final_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SentinelError(RuntimeError):
        pass

    MockPPOModel.reset()
    MockPPOModel.progress_points = ()
    vector = MockVecEnv()
    error = SentinelError("completed state failure")
    writer = FailingStateWriter(runner_module._write_state, fail_on_call=2, error=error)
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_write_state", writer)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    root = tmp_path / "runs"
    with pytest.raises(SentinelError) as caught:
        TrainingRunner(_config(root)).run_single(make_factory())

    assert caught.value is error
    run_directory = next(root.iterdir())
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["final_model"] is None
    assert state["final_model_cleanup_error"] is None
    assert list(run_directory.glob("*-final-model.zip")) == []


def test_final_model_cleanup_failure_preserves_completed_state_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SentinelError(RuntimeError):
        pass

    MockPPOModel.reset()
    MockPPOModel.progress_points = ()
    vector = MockVecEnv()
    state_error = SentinelError("completed state failure")
    cleanup_error = PermissionError("final model cleanup failure")
    writer = FailingStateWriter(runner_module._write_state, fail_on_call=2, error=state_error)
    original_unlink = Path.unlink

    def fail_final_model_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.endswith("-final-model.zip"):
            raise cleanup_error
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_write_state", writer)
    monkeypatch.setattr(runner_module, "_now", _started_at)
    monkeypatch.setattr(Path, "unlink", fail_final_model_cleanup)

    root = tmp_path / "runs"
    with pytest.raises(SentinelError) as caught:
        TrainingRunner(_config(root)).run_single(make_factory())

    assert caught.value is state_error
    run_directory = next(root.iterdir())
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["final_model"] is None
    assert state["final_model_cleanup_error"] == "PermissionError: final model cleanup failure"
    assert list(run_directory.glob("*-final-model.zip"))

def test_cleanup_failure_after_success_propagates_but_keeps_completed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    vector = MockVecEnv(close_error=RuntimeError("close failure"))
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="close failure"):
        TrainingRunner(_config(root)).run_single(make_factory())

    run_directory = next(root.iterdir())
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["final_model"] == "ppo-singletrack-seed42-final-model.zip"
    assert (run_directory / state["final_model"]).is_file()
    assert vector.close_calls == 1


def test_training_error_remains_primary_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    training_error = RuntimeError("training failure")
    MockPPOModel.learn_error = training_error
    vector = MockVecEnv(close_error=RuntimeError("close failure"))
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    with pytest.raises(RuntimeError) as caught:
        TrainingRunner(_config(tmp_path / "runs")).run_single(make_factory())

    assert caught.value is training_error
    assert vector.close_calls == 1

def test_initial_state_write_failure_retries_failed_snapshot_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SentinelError(RuntimeError):
        pass

    original_write_state = runner_module._write_state
    writer = FailingStateWriter(original_write_state, fail_on_call=1, error=SentinelError("initial state failure"))
    monkeypatch.setattr(runner_module, "_write_state", writer)
    monkeypatch.setattr(runner_module, "_now", _started_at)
    monkeypatch.setattr(
        runner_module,
        "make_vector_env",
        lambda *args: (_ for _ in ()).throw(AssertionError("env must not be created")),
    )

    root = tmp_path / "runs"
    with pytest.raises(SentinelError, match="initial state failure"):
        TrainingRunner(_config(root)).run_single(make_factory())

    run_directory = next(root.iterdir())
    state_path = run_directory / "state.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["actual_total_timesteps"] == 0
    assert state["final_model"] is None


def test_training_runner_accepts_optional_provenance_and_rejects_wrong_type(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "runs")
    assert isinstance(TrainingRunner(config), TrainingRunner)
    assert isinstance(TrainingRunner(config, provenance=None), TrainingRunner)
    with pytest.raises(ValueError, match="provenance"):
        TrainingRunner(config, provenance=object())  # type: ignore[arg-type]


def test_training_run_result_defaults_provenance_path_to_none(tmp_path: Path) -> None:
    result = TrainingRunResult(
        run_id="run",
        run_directory=tmp_path,
        final_model_path=tmp_path / "model.zip",
        checkpoint_paths=(),
        requested_total_timesteps=8,
        actual_total_timesteps=8,
    )
    assert result.provenance_path is None


def test_run_single_persists_supplied_provenance_before_env_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    vector = MockVecEnv()
    observed: list[bool] = []
    provenance = make_provenance(("authored-track",))

    def build_vector(*args: object) -> MockVecEnv:
        run_directory = next((tmp_path / "runs").iterdir())
        provenance_path = run_directory / "provenance.json"
        observed.append(provenance_path.is_file())
        return vector

    monkeypatch.setattr(runner_module, "make_vector_env", build_vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "_now", _started_at)

    result = TrainingRunner(
        _config(tmp_path / "runs"),
        provenance=provenance,
    ).run_single(make_factory())

    assert observed == [True]
    assert result.provenance_path == result.run_directory / "provenance.json"
    assert result.provenance_path.is_file()
    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["provenance_file"] == "provenance.json"
    before = result.provenance_path.read_bytes()
    assert before
    assert result.provenance_path.read_bytes() == before


def test_run_multi_rejects_provenance_track_order_mismatch_before_artifacts(
    tmp_path: Path,
) -> None:
    scenarios = make_scenario_set()
    actual_ids = tuple(scenario.track_id for scenario in scenarios.scenarios)
    provenance = make_provenance(tuple(reversed(actual_ids)))
    root = tmp_path / "runs"

    with pytest.raises(ValueError, match="track IDs"):
        TrainingRunner(
            _config(root),
            provenance=provenance,
        ).run_multi(scenarios)

    assert not root.exists()


def test_run_multi_rejects_missing_or_extra_provenance_tracks_before_artifacts(
    tmp_path: Path,
) -> None:
    scenarios = make_scenario_set()
    actual_ids = tuple(scenario.track_id for scenario in scenarios.scenarios)
    for supplied in (actual_ids[:-1], (*actual_ids, "extra")):
        root = tmp_path / ("runs-" + str(len(supplied)))
        with pytest.raises(ValueError, match="track IDs"):
            TrainingRunner(
                _config(root),
                provenance=make_provenance(supplied),
            ).run_multi(scenarios)
        assert not root.exists()


def test_provenance_write_failure_prevents_env_and_marks_state_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OSError("provenance write failure")
    root = tmp_path / "runs"
    monkeypatch.setattr(runner_module, "_now", _started_at)
    monkeypatch.setattr(
        runner_module,
        "write_run_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        runner_module,
        "make_vector_env",
        lambda *args: (_ for _ in ()).throw(AssertionError("env must not be created")),
    )

    with pytest.raises(OSError) as caught:
        TrainingRunner(
            _config(root),
            provenance=make_provenance(("authored-track",)),
        ).run_single(make_factory())

    assert caught.value is error
    run_directory = next(root.iterdir())
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["provenance_file"] == "provenance.json"
    assert not (run_directory / "provenance.json").exists()


def test_training_failure_preserves_written_provenance_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    MockPPOModel.reset()
    MockPPOModel.learn_error = RuntimeError("learn failure")
    vector = MockVecEnv()
    root = tmp_path / "runs"
    provenance = make_provenance(("authored-track",))
    monkeypatch.setattr(runner_module, "_now", _started_at)
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: vector)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)

    with pytest.raises(RuntimeError, match="learn failure"):
        TrainingRunner(
            _config(root),
            provenance=provenance,
        ).run_single(make_factory())

    run_directory = next(root.iterdir())
    provenance_path = run_directory / "provenance.json"
    before = provenance_path.read_bytes()
    assert before
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["provenance_file"] == "provenance.json"
    assert provenance_path.read_bytes() == before
