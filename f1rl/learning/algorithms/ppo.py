from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from stable_baselines3 import PPO as _SB3PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as error:
    raise ImportError(
        "PPO support requires the optional RL dependencies; install f1-rl with the `rl` extra."
    ) from error

from ..gym_environment import F1RLEnv

MAX_SEED = 2**32 - 1


class _ProgressHookCallback(BaseCallback):
    def __init__(self, hook: Callable[[int], None]) -> None:
        super().__init__(verbose=0)
        self._hook = hook

    def _on_step(self) -> bool:
        self._hook(int(self.model.num_timesteps))
        return True


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

        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= MAX_SEED
        ):
            raise ValueError(f"seed must be None or an integer in [0, {MAX_SEED}]")
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
        if set(document) != {"ppo"}:
            raise ValueError("configuration root must contain exactly one ppo mapping")
        raw_ppo: Any = document["ppo"]
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


PPOTrainingEnv = F1RLEnv | VecEnv


def _validate_rollout_size(env: PPOTrainingEnv, n_steps: int, batch_size: int) -> None:
    if isinstance(env, F1RLEnv):
        num_envs = 1
    elif isinstance(env, VecEnv):
        num_envs = env.num_envs
    else:
        raise ValueError("env must be F1RLEnv or VecEnv")
    rollout_size = n_steps * num_envs
    if batch_size > rollout_size:
        raise ValueError("batch_size must be <= n_steps * num_envs")
    if rollout_size % batch_size != 0:
        raise ValueError("batch_size must divide n_steps * num_envs exactly")


def _validate_training_admission(env: object) -> None:
    if isinstance(env, F1RLEnv):
        blocked = env.requires_scientific_admission
    elif isinstance(env, VecEnv):
        try:
            flags = env.get_attr("requires_scientific_admission")
        except AttributeError as exc:
            raise ValueError("vector training requires explicit admission metadata") from exc
        if len(flags) != env.num_envs or any(type(flag) is not bool for flag in flags):
            raise ValueError("invalid vector training admission metadata")
        blocked = any(flags)
    else:
        raise ValueError("unregistered training environment")
    if blocked:
        raise ValueError("research physics training is blocked until scientific admission is reviewed")


class PPO(_SB3PPO):
    """Project backend: prediction-only loading never grants learning admission."""

    def __init__(self, policy, env, **kwargs):
        # SB3 load reconstructs with setup disabled, including inference envs.
        if env is not None and kwargs.get("_init_setup_model", True):
            _validate_training_admission(env)
        super().__init__(policy, env, **kwargs)

    def learn(self, *args, **kwargs):
        env = self.get_env()
        if env is None:
            raise RuntimeError("an environment is required to continue learning")
        _validate_training_admission(env)
        return super().learn(*args, **kwargs)


class PPOModel:
    def __init__(self, model: PPO) -> None:
        self._model = model

    @classmethod
    def create(cls, env: PPOTrainingEnv, config: PPOConfig) -> "PPOModel":
        if not isinstance(env, (F1RLEnv, VecEnv)):
            raise ValueError("env must be F1RLEnv or VecEnv")
        if not isinstance(config, PPOConfig):
            raise ValueError("config must be PPOConfig")
        _validate_training_admission(env)
        _validate_rollout_size(env, config.n_steps, config.batch_size)

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

    def learn(
        self,
        total_timesteps: int,
        *,
        progress_hook: Callable[[int], None] | None = None,
    ) -> "PPOModel":
        if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int):
            raise ValueError("total_timesteps must be a positive integer")
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
        if progress_hook is not None and not callable(progress_hook):
            raise ValueError("progress_hook must be callable or None")
        env = self._model.get_env()
        if env is None:
            raise RuntimeError("an environment is required to continue learning")
        _validate_training_admission(env)

        kwargs: dict[str, object] = {
            "total_timesteps": total_timesteps,
            "reset_num_timesteps": False,
        }
        if progress_hook is not None:
            kwargs["callback"] = _ProgressHookCallback(progress_hook)
        self._model.learn(**kwargs)
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
        env: PPOTrainingEnv | None = None,
        device: str = "cpu",
    ) -> "PPOModel":
        archive = cls._archive_path(path)
        if not archive.is_file():
            raise ValueError("model archive must exist as a file")
        if env is not None and not isinstance(env, (F1RLEnv, VecEnv)):
            raise ValueError("env must be F1RLEnv, VecEnv, or None")
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError('device must be one of "cpu", "cuda", or "auto"')

        model = PPO.load(str(archive), env=env, device=device)
        if env is not None:
            _validate_rollout_size(env, int(model.n_steps), int(model.batch_size))
        if device == "cuda" and getattr(model.device, "type", None) != "cuda":
            raise RuntimeError("CUDA device was explicitly requested but could not be honored")
        return cls(model)
