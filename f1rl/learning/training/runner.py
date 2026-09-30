from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Literal

from .config import TrainingRunConfig

TrainingMode = Literal["singletrack", "multitrack"]


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    run_id: str
    run_directory: Path
    final_model_path: Path
    checkpoint_paths: tuple[Path, ...]
    requested_total_timesteps: int
    actual_total_timesteps: int


@dataclass(frozen=True, slots=True)
class _RunArtifacts:
    run_id: str
    run_directory: Path
    checkpoint_directory: Path
    state_path: Path
    final_model_path: Path
    artifact_stem: str


def _seed_text(seed: int | None) -> str:
    return "none" if seed is None else str(seed)


def _artifact_stem(
    mode: TrainingMode,
    ppo_seed: int | None,
    sampling_seed: int | None,
) -> str:
    if mode == "singletrack":
        if sampling_seed is not None:
            raise ValueError("singletrack naming must not include sampling_seed")
        return f"ppo-singletrack-seed{_seed_text(ppo_seed)}"
    if mode == "multitrack":
        if sampling_seed is None:
            raise ValueError("multitrack naming requires sampling_seed")
        return f"ppo-multitrack-seed{_seed_text(ppo_seed)}-sample{sampling_seed}"
    raise ValueError("mode must be singletrack or multitrack")


def _build_run_id(
    started_at: datetime,
    mode: TrainingMode,
    ppo_seed: int | None,
    sampling_seed: int | None,
) -> str:
    timestamp = started_at.strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{_artifact_stem(mode, ppo_seed, sampling_seed)}"


def _checkpoint_thresholds(
    total_timesteps: int,
    interval: int | None,
) -> tuple[int, ...]:
    if interval is None:
        return ()
    return tuple(range(interval, total_timesteps, interval))


def _advance_checkpoint_schedule(
    thresholds: tuple[int, ...],
    next_index: int,
    actual_timesteps: int,
) -> tuple[bool, int]:
    initial_index = next_index
    while next_index < len(thresholds) and actual_timesteps >= thresholds[next_index]:
        next_index += 1
    return next_index > initial_index, next_index


def _checkpoint_archive_path(
    artifacts: _RunArtifacts,
    actual_timesteps: int,
) -> Path:
    return artifacts.checkpoint_directory / (
        f"{artifacts.artifact_stem}-step{actual_timesteps}.zip"
    )


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )


def _initial_state(
    config: TrainingRunConfig,
    *,
    run_id: str,
    mode: TrainingMode,
    training_track_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "running",
        "mode": mode,
        "requested_total_timesteps": config.total_timesteps,
        "actual_total_timesteps": 0,
        "checkpoint_every_steps": config.checkpoint_every_steps,
        "latest_checkpoint": None,
        "final_model": None,
        "ppo": asdict(config.ppo),
        "vector_env": asdict(config.vector_env),
        "sampling_seed": config.sampling_seed if mode == "multitrack" else None,
        "training_tracks": list(training_track_ids),
    }


def _prepare_run(
    config: TrainingRunConfig,
    *,
    started_at: datetime,
    mode: TrainingMode,
    training_track_ids: tuple[str, ...],
) -> _RunArtifacts:
    sampling_seed = config.sampling_seed if mode == "multitrack" else None
    artifact_stem = _artifact_stem(mode, config.ppo.seed, sampling_seed)
    run_id = _build_run_id(started_at, mode, config.ppo.seed, sampling_seed)
    root = config.artifacts_root
    if root.exists() and not root.is_dir():
        raise ValueError("artifacts_root must be a directory path")
    root.mkdir(parents=True, exist_ok=True)

    run_directory = root / run_id
    if run_directory.exists():
        raise FileExistsError(f"run directory already exists: {run_directory}")
    run_directory.mkdir()
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir()

    artifacts = _RunArtifacts(
        run_id=run_id,
        run_directory=run_directory,
        checkpoint_directory=checkpoint_directory,
        state_path=run_directory / "state.json",
        final_model_path=run_directory / f"{artifact_stem}-final-model.zip",
        artifact_stem=artifact_stem,
    )
    _write_state(
        artifacts.state_path,
        _initial_state(
            config,
            run_id=run_id,
            mode=mode,
            training_track_ids=training_track_ids,
        ),
    )
    return artifacts
