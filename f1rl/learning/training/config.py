from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ..algorithms.ppo import PPOConfig


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
