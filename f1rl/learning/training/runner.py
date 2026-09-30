from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Literal

from ..algorithms.ppo import PPOModel
from .config import TrainingRunConfig
from .env_factory import F1RLEnvFactory, TrainingScenarioSet
from .vector_env import make_multi_track_vector_env, make_vector_env

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


def _read_state(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError("run state must be a JSON object")
    return document


def _update_state(
    artifacts: _RunArtifacts,
    **updates: object,
) -> None:
    state = _read_state(artifacts.state_path)
    state.update(updates)
    _write_state(artifacts.state_path, state)


def _relative_artifact_path(artifacts: _RunArtifacts, path: Path) -> str:
    return path.relative_to(artifacts.run_directory).as_posix()


def _now() -> datetime:
    return datetime.now()


class TrainingRunner:
    def __init__(self, config: TrainingRunConfig) -> None:
        if not isinstance(config, TrainingRunConfig):
            raise ValueError("config must be TrainingRunConfig")
        self._config = config

    def run_single(self, factory: F1RLEnvFactory) -> TrainingRunResult:
        if not isinstance(factory, F1RLEnvFactory):
            raise ValueError("factory must be F1RLEnvFactory")
        return self._run(
            mode="singletrack",
            training_track_ids=(),
            build_env=lambda: make_vector_env(factory, self._config.vector_env),
        )

    def run_multi(self, scenarios: TrainingScenarioSet) -> TrainingRunResult:
        if not isinstance(scenarios, TrainingScenarioSet):
            raise ValueError("scenarios must be TrainingScenarioSet")
        return self._run(
            mode="multitrack",
            training_track_ids=tuple(scenario.track_id for scenario in scenarios.scenarios),
            build_env=lambda: make_multi_track_vector_env(
                scenarios,
                self._config.vector_env,
                self._config.sampling_seed,
            ),
        )

    def _run(
        self,
        *,
        mode: TrainingMode,
        training_track_ids: tuple[str, ...],
        build_env: Callable[[], object],
    ) -> TrainingRunResult:
        artifacts = _prepare_run(
            self._config,
            started_at=_now(),
            mode=mode,
            training_track_ids=training_track_ids,
        )
        thresholds = _checkpoint_thresholds(
            self._config.total_timesteps,
            self._config.checkpoint_every_steps,
        )
        next_threshold_index = 0
        checkpoint_paths: list[Path] = []
        env: object | None = None
        model: PPOModel | None = None
        primary_error: Exception | None = None

        try:
            env = build_env()
            model = PPOModel.create(env, self._config.ppo)  # type: ignore[arg-type]

            def on_progress(actual_timesteps: int) -> None:
                nonlocal next_threshold_index
                due, advanced_index = _advance_checkpoint_schedule(
                    thresholds,
                    next_threshold_index,
                    actual_timesteps,
                )
                if not due:
                    return
                checkpoint = model.save(
                    _checkpoint_archive_path(artifacts, actual_timesteps)
                )
                checkpoint_paths.append(checkpoint)
                next_threshold_index = advanced_index
                _update_state(
                    artifacts,
                    actual_total_timesteps=actual_timesteps,
                    latest_checkpoint=_relative_artifact_path(artifacts, checkpoint),
                )

            model.learn(
                self._config.total_timesteps,
                progress_hook=on_progress,
            )
            actual_total_timesteps = model.num_timesteps
            final_model_path = model.save(artifacts.final_model_path)
            _update_state(
                artifacts,
                status="completed",
                actual_total_timesteps=actual_total_timesteps,
                final_model=_relative_artifact_path(artifacts, final_model_path),
            )
            return TrainingRunResult(
                run_id=artifacts.run_id,
                run_directory=artifacts.run_directory,
                final_model_path=final_model_path,
                checkpoint_paths=tuple(checkpoint_paths),
                requested_total_timesteps=self._config.total_timesteps,
                actual_total_timesteps=actual_total_timesteps,
            )
        except Exception as error:
            primary_error = error
            actual_timesteps = 0
            if model is not None:
                try:
                    actual_timesteps = model.num_timesteps
                except Exception:
                    actual_timesteps = 0
            try:
                _update_state(
                    artifacts,
                    status="failed",
                    actual_total_timesteps=actual_timesteps,
                    final_model=None,
                )
            except Exception:
                pass
            raise
        finally:
            if env is not None:
                try:
                    env.close()  # type: ignore[attr-defined]
                except Exception:
                    if primary_error is None:
                        raise
