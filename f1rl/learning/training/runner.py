from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Literal

from ..algorithms.ppo import PPOModel
from ..provenance import RunProvenance, write_run_provenance
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
    provenance_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _RunArtifacts:
    run_id: str
    run_directory: Path
    checkpoint_directory: Path
    state_path: Path
    provenance_path: Path
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
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(state, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _initial_state(
    config: TrainingRunConfig,
    *,
    run_id: str,
    mode: TrainingMode,
    training_track_ids: tuple[str, ...],
    provenance_file: str | None = None,
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
        "final_model_cleanup_error": None,
        "ppo": asdict(config.ppo),
        "vector_env": asdict(config.vector_env),
        "sampling_seed": config.sampling_seed if mode == "multitrack" else None,
        "training_tracks": list(training_track_ids),
        "provenance_file": provenance_file,
    }


def _prepare_run(
    config: TrainingRunConfig,
    *,
    started_at: datetime,
    mode: TrainingMode,
    training_track_ids: tuple[str, ...],
    provenance_file: str | None = None,
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
        provenance_path=run_directory / "provenance.json",
        final_model_path=run_directory / f"{artifact_stem}-final-model.zip",
        artifact_stem=artifact_stem,
    )
    initial_state = _initial_state(
        config,
        run_id=run_id,
        mode=mode,
        training_track_ids=training_track_ids,
        provenance_file=provenance_file,
    )
    try:
        _write_state(artifacts.state_path, initial_state)
    except Exception:
        failed_state = dict(initial_state)
        failed_state["status"] = "failed"
        try:
            _write_state(artifacts.state_path, failed_state)
        except Exception:
            pass
        raise
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
    def __init__(
        self,
        config: TrainingRunConfig,
        *,
        provenance: RunProvenance | None = None,
        checkpoint_observer: Callable[[Path, int], None] | None = None,
        final_model_observer: Callable[[Path, int], None] | None = None,
    ) -> None:
        if not isinstance(config, TrainingRunConfig):
            raise ValueError("config must be TrainingRunConfig")
        if provenance is not None and not isinstance(provenance, RunProvenance):
            raise ValueError("provenance must be RunProvenance or None")
        if checkpoint_observer is not None and not callable(checkpoint_observer):
            raise ValueError("checkpoint_observer must be callable or None")
        if final_model_observer is not None and not callable(final_model_observer):
            raise ValueError("final_model_observer must be callable or None")
        self._config = config
        self._provenance = provenance
        self._checkpoint_observer = checkpoint_observer
        self._final_model_observer = final_model_observer

    def run_registered(self, binding: object) -> TrainingRunResult:
        """Validate a policy-bound WP7 execution before any model/artifact creation.

        This does not waive the independent research admission guard.
        """
        from ...research.wp7_execution import PolicyBoundTrainingConfig

        if not isinstance(binding, PolicyBoundTrainingConfig):
            raise ValueError("registered execution requires PolicyBoundTrainingConfig")
        binding.validate(self._config, binding.scenarios)
        if len(binding.scenarios.scenarios) == 1:
            return self.run_single(binding.scenarios.scenarios[0].factory)
        return self.run_multi(binding.scenarios)

    def run_single(self, factory: F1RLEnvFactory) -> TrainingRunResult:
        if not isinstance(factory, F1RLEnvFactory):
            raise ValueError("factory must be F1RLEnvFactory")
        if factory.requires_scientific_admission:
            from ...research.wp7_admission import reject_research_training
            reject_research_training(
                "full-body candidate training is blocked until versioned "
                "provenance, route-corridor and independent-seed protocol approval"
            )
        return self._run(
            mode="singletrack",
            training_track_ids=(),
            build_env=lambda: make_vector_env(factory, self._config.vector_env),
        )

    def run_multi(self, scenarios: TrainingScenarioSet) -> TrainingRunResult:
        if not isinstance(scenarios, TrainingScenarioSet):
            raise ValueError("scenarios must be TrainingScenarioSet")
        if any(scenario.factory.requires_scientific_admission for scenario in scenarios.scenarios):
            from ...research.wp7_admission import reject_research_training
            reject_research_training(
                "full-body candidate training is blocked until versioned "
                "provenance, route-corridor and independent-seed protocol approval"
            )
        training_track_ids = tuple(
            scenario.track_id for scenario in scenarios.scenarios
        )
        if self._provenance is not None:
            provenance_track_ids = tuple(
                track.track_id for track in self._provenance.tracks
            )
            if provenance_track_ids != training_track_ids:
                raise ValueError(
                    "provenance track IDs must exactly match scenario track IDs"
                )
        return self._run(
            mode="multitrack",
            training_track_ids=training_track_ids,
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
        provenance_file = (
            "provenance.json" if self._provenance is not None else None
        )
        artifacts = _prepare_run(
            self._config,
            started_at=_now(),
            mode=mode,
            training_track_ids=training_track_ids,
            provenance_file=provenance_file,
        )
        thresholds = _checkpoint_thresholds(
            self._config.total_timesteps,
            self._config.checkpoint_every_steps,
        )
        next_threshold_index = 0
        checkpoint_paths: list[Path] = []
        env: object | None = None
        model: PPOModel | None = None
        final_model_path: Path | None = None
        primary_error: Exception | None = None

        try:
            if self._provenance is not None:
                write_run_provenance(
                    artifacts.provenance_path,
                    self._provenance,
                )
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
                if self._checkpoint_observer is not None:
                    # The observer sees only an already-saved checkpoint and
                    # the *actual* timestep (which may exceed the nominal
                    # threshold). Its failure aborts the run and preserves
                    # the saved artifact for forensic recovery.
                    self._checkpoint_observer(checkpoint, actual_timesteps)
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
            if self._final_model_observer is not None:
                # Never publish "completed" before the final checkpoint
                # passes the same authenticity/evaluation gate as intermediate ones.
                self._final_model_observer(final_model_path, actual_total_timesteps)
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
                provenance_path=(
                    artifacts.provenance_path
                    if self._provenance is not None
                    else None
                ),
            )
        except Exception as error:
            primary_error = error
            final_model_cleanup_error: str | None = None
            if final_model_path is not None:
                try:
                    final_model_path.unlink()
                except Exception as cleanup_error:
                    final_model_cleanup_error = (
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            actual_timesteps = 0
            if model is not None:
                try:
                    actual_timesteps = model.num_timesteps
                except Exception:
                    actual_timesteps = 0
            failure_updates: dict[str, object] = {
                "status": "failed",
                "actual_total_timesteps": actual_timesteps,
                "final_model": None,
                "final_model_cleanup_error": final_model_cleanup_error,
            }
            if checkpoint_paths:
                failure_updates["latest_checkpoint"] = _relative_artifact_path(
                    artifacts,
                    checkpoint_paths[-1],
                )
            try:
                _update_state(artifacts, **failure_updates)
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
