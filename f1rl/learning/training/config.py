from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ..algorithms.ppo import MAX_SEED, PPOConfig


@dataclass(frozen=True, slots=True)
class VectorEnvConfig:
    n_envs: int = 8
    backend: str = "dummy"
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        if isinstance(self.n_envs, bool) or not isinstance(self.n_envs, int) or self.n_envs < 1:
            raise ValueError("n_envs must be an integer >= 1")
        if self.backend not in {"dummy", "subproc"}:
            raise ValueError('backend must be "dummy" or "subproc"')
        if not isinstance(self.start_method, str) or not self.start_method.strip():
            raise ValueError("start_method must be a non-empty string")
        if self.backend == "subproc" and self.start_method != "spawn":
            raise ValueError('subproc backend requires start_method="spawn"')


@dataclass(frozen=True, slots=True)
class VectorTrainingConfig:
    ppo: PPOConfig
    vector_env: VectorEnvConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VectorTrainingConfig":
        if not isinstance(path, (str, Path)):
            raise ValueError("path must be a string or Path")
        if isinstance(path, str) and not path.strip():
            raise ValueError("path must not be blank")
        config_path = Path(path)
        if not config_path.is_file():
            raise ValueError("configuration path must exist as a file")
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError("configuration file could not be loaded") from error

        if not isinstance(document, dict):
            raise ValueError("configuration root must be a mapping")
        if set(document) != {"ppo", "vector_env"}:
            raise ValueError("configuration root must contain exactly ppo and vector_env")
        raw_ppo: Any = document["ppo"]
        raw_vector: Any = document["vector_env"]
        if not isinstance(raw_ppo, dict) or not isinstance(raw_vector, dict):
            raise ValueError("ppo and vector_env must be mappings")

        ppo_fields = {field.name for field in fields(PPOConfig)}
        vector_fields = {field.name for field in fields(VectorEnvConfig)}
        if set(raw_ppo) - ppo_fields:
            raise ValueError("unknown PPO configuration key")
        if set(raw_vector) - vector_fields:
            raise ValueError("unknown vector_env configuration key")

        try:
            return cls(
                ppo=PPOConfig(**raw_ppo),
                vector_env=VectorEnvConfig(**raw_vector),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid vector training configuration") from error


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    ppo: PPOConfig
    vector_env: VectorEnvConfig
    total_timesteps: int = 50_000
    checkpoint_every_steps: int | None = 25_000
    sampling_seed: int = 0
    artifacts_root: Path = Path("artifacts/runs")

    def __post_init__(self) -> None:
        if not isinstance(self.ppo, PPOConfig):
            raise ValueError("ppo must be PPOConfig")
        if not isinstance(self.vector_env, VectorEnvConfig):
            raise ValueError("vector_env must be VectorEnvConfig")
        if (
            isinstance(self.total_timesteps, bool)
            or not isinstance(self.total_timesteps, int)
            or self.total_timesteps <= 0
        ):
            raise ValueError("total_timesteps must be a positive integer")
        if self.checkpoint_every_steps is not None and (
            isinstance(self.checkpoint_every_steps, bool)
            or not isinstance(self.checkpoint_every_steps, int)
            or self.checkpoint_every_steps <= 0
        ):
            raise ValueError("checkpoint_every_steps must be None or a positive integer")
        if (
            isinstance(self.sampling_seed, bool)
            or not isinstance(self.sampling_seed, int)
            or not 0 <= self.sampling_seed <= MAX_SEED
        ):
            raise ValueError(f"sampling_seed must be an integer in [0, {MAX_SEED}]")
        if not isinstance(self.artifacts_root, Path):
            raise ValueError("artifacts_root must be Path")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingRunConfig":
        if not isinstance(path, (str, Path)):
            raise ValueError("path must be a string or Path")
        if isinstance(path, str) and not path.strip():
            raise ValueError("path must not be blank")
        config_path = Path(path)
        if not config_path.is_file():
            raise ValueError("configuration path must exist as a file")
        config_path = config_path.resolve()
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError("configuration file could not be loaded") from error

        if not isinstance(document, dict):
            raise ValueError("configuration root must be a mapping")
        if document.get("schema_version") == 2:
            if set(document) != {"schema_version", "training", "ppo", "vector_env", "exploratory"}:
                raise ValueError("invalid schema v2 runtime configuration sections")
        elif set(document) != {"training", "ppo", "vector_env"}:
            raise ValueError("configuration root must contain exactly training, ppo, and vector_env")
        raw_training: Any = document["training"]
        raw_ppo: Any = document["ppo"]
        raw_vector: Any = document["vector_env"]
        if not all(isinstance(section, dict) for section in (raw_training, raw_ppo, raw_vector)):
            raise ValueError("training, ppo, and vector_env must be mappings")

        training_fields = {
            "total_timesteps",
            "checkpoint_every_steps",
            "sampling_seed",
            "artifacts_root",
        }
        ppo_fields = {field.name for field in fields(PPOConfig)}
        vector_fields = {field.name for field in fields(VectorEnvConfig)}
        if set(raw_training) - training_fields:
            raise ValueError("unknown training configuration key")
        if set(raw_ppo) - ppo_fields:
            raise ValueError("unknown PPO configuration key")
        if set(raw_vector) - vector_fields:
            raise ValueError("unknown vector_env configuration key")

        training_values = dict(raw_training)
        raw_artifacts_root = training_values.pop("artifacts_root", "artifacts/runs")
        if not isinstance(raw_artifacts_root, str) or not raw_artifacts_root.strip():
            raise ValueError("artifacts_root must be a non-blank string")
        artifacts_root = Path(raw_artifacts_root)
        if not artifacts_root.is_absolute():
            artifacts_root = config_path.parent / artifacts_root

        try:
            return cls(
                ppo=PPOConfig(**raw_ppo),
                vector_env=VectorEnvConfig(**raw_vector),
                artifacts_root=artifacts_root,
                **training_values,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid training run configuration") from error
