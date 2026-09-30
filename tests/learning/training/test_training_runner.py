import json
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from f1rl.learning.algorithms.ppo import PPOConfig
from f1rl.learning.training.config import TrainingRunConfig, VectorEnvConfig
from f1rl.learning.training.runner import (
    TrainingRunResult,
    _advance_checkpoint_schedule,
    _artifact_stem,
    _build_run_id,
    _checkpoint_archive_path,
    _checkpoint_thresholds,
    _prepare_run,
)


def _config(root: Path, *, seed: int | None = 42, sampling_seed: int = 123) -> TrainingRunConfig:
    return TrainingRunConfig(
        ppo=PPOConfig(seed=seed),
        vector_env=VectorEnvConfig(n_envs=2),
        total_timesteps=50_000,
        checkpoint_every_steps=25_000,
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
