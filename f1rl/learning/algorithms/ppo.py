from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from stable_baselines3 import PPO
except ImportError as error:
    raise ImportError(
        "PPO support requires the optional RL dependencies; install f1-rl with the `rl` extra."
    ) from error

from ..gym_environment import F1RLEnv


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


class PPOModel:
    def __init__(self, model: PPO) -> None:
        self._model = model

    @classmethod
    def create(cls, env: F1RLEnv, config: PPOConfig) -> "PPOModel":
        if not isinstance(env, F1RLEnv):
            raise ValueError("env must be F1RLEnv")
        if not isinstance(config, PPOConfig):
            raise ValueError("config must be PPOConfig")

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            device=config.device,
            verbose=0,
        )
        if config.device == "cuda" and getattr(model.device, "type", None) != "cuda":
            raise RuntimeError("CUDA device was explicitly requested but could not be honored")
        return cls(model)

    @property
    def num_timesteps(self) -> int:
        return int(self._model.num_timesteps)

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        if not isinstance(observation, np.ndarray):
            raise ValueError("observation must be a numpy.ndarray")
        if observation.shape != self._model.observation_space.shape:
            raise ValueError("observation has unexpected shape")
        if observation.dtype.kind not in "iuf":
            raise ValueError("observation must contain real numeric values")
        if not np.isfinite(observation).all():
            raise ValueError("observation values must be finite")
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be bool")

        normalized = np.asarray(observation, dtype=np.float32)
        if not self._model.observation_space.contains(normalized):
            raise ValueError("observation is outside model observation space")

        action, _ = self._model.predict(normalized, deterministic=deterministic)
        result = np.asarray(action, dtype=np.float32)
        if result.shape != self._model.action_space.shape:
            raise RuntimeError("backend returned action with unexpected shape")
        if not np.isfinite(result).all():
            raise RuntimeError("backend returned non-finite action")
        if not self._model.action_space.contains(result):
            raise RuntimeError("backend returned action outside action space")
        return result

    def learn(self, total_timesteps: int) -> "PPOModel":
        if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int):
            raise ValueError("total_timesteps must be a positive integer")
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
        if self._model.get_env() is None:
            raise RuntimeError("an environment is required to continue learning")
        self._model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
        )
        return self

    @staticmethod
    def _archive_path(path: str | Path) -> Path:
        if not isinstance(path, (str, Path)):
            raise ValueError("path must be a string or Path")
        if isinstance(path, str) and not path.strip():
            raise ValueError("path must not be blank")
        archive = Path(path)
        if not str(archive).lower().endswith(".zip"):
            archive = Path(str(archive) + ".zip")
        return archive

    def save(self, path: str | Path) -> Path:
        archive = self._archive_path(path)
        if not archive.parent.is_dir():
            raise ValueError("model archive parent directory must exist")
        if archive.exists() and archive.is_dir():
            raise ValueError("model archive destination must not be a directory")
        self._model.save(str(archive))
        return archive

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        env: F1RLEnv | None = None,
        device: str = "cpu",
    ) -> "PPOModel":
        archive = cls._archive_path(path)
        if not archive.is_file():
            raise ValueError("model archive must exist as a file")
        if env is not None and not isinstance(env, F1RLEnv):
            raise ValueError("env must be F1RLEnv or None")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError('device must be one of "cpu", "cuda", or "auto"')

        model = PPO.load(str(archive), env=env, device=device)
        if device == "cuda" and getattr(model.device, "type", None) != "cuda":
            raise RuntimeError("CUDA device was explicitly requested but could not be honored")
        return cls(model)
