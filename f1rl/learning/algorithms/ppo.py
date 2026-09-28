from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import yaml


def _validate_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _validate_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int | None = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        learning_rate = _validate_real(self.learning_rate, "learning_rate")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        n_steps = _validate_int(self.n_steps, "n_steps", minimum=2)
        batch_size = _validate_int(self.batch_size, "batch_size", minimum=2)
        _validate_int(self.n_epochs, "n_epochs", minimum=1)
        if batch_size > n_steps:
            raise ValueError("batch_size must be <= n_steps")
        if n_steps % batch_size != 0:
            raise ValueError("batch_size must divide n_steps exactly")

        gamma = _validate_real(self.gamma, "gamma")
        gae_lambda = _validate_real(self.gae_lambda, "gae_lambda")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")

        clip_range = _validate_real(self.clip_range, "clip_range")
        ent_coef = _validate_real(self.ent_coef, "ent_coef")
        vf_coef = _validate_real(self.vf_coef, "vf_coef")
        max_grad_norm = _validate_real(self.max_grad_norm, "max_grad_norm")
        if not 0.0 < clip_range <= 1.0:
            raise ValueError("clip_range must be in (0, 1]")
        if ent_coef < 0.0:
            raise ValueError("ent_coef must be non-negative")
        if vf_coef < 0.0:
            raise ValueError("vf_coef must be non-negative")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")

        if self.seed is not None:
            _validate_int(self.seed, "seed", minimum=0)
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError('device must be one of "cpu", "cuda", or "auto"')

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PPOConfig":
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
        unknown_root = set(document) - {"ppo"}
        if unknown_root:
            raise ValueError(f"unknown configuration keys: {sorted(unknown_root)!r}")

        raw_ppo: Any = document.get("ppo", {})
        if not isinstance(raw_ppo, dict):
            raise ValueError("ppo configuration must be a mapping")

        allowed = {field.name for field in fields(cls)}
        unknown = set(raw_ppo) - allowed
        if unknown:
            raise ValueError(f"unknown PPO configuration keys: {sorted(unknown)!r}")
        try:
            return cls(**raw_ppo)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid PPO configuration") from error
