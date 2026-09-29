from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from ..gym_environment import F1RLEnv
from .config import VectorEnvConfig
from .env_factory import F1RLEnvFactory


@dataclass(frozen=True, slots=True)
class _RankedEnvFactory:
    factory: F1RLEnvFactory
    rank: int

    def __call__(self) -> F1RLEnv:
        return self.factory.create()


def make_vector_env(
    factory: F1RLEnvFactory,
    config: VectorEnvConfig,
) -> VecEnv:
    if not isinstance(factory, F1RLEnvFactory):
        raise ValueError("factory must be F1RLEnvFactory")
    if not isinstance(config, VectorEnvConfig):
        raise ValueError("config must be VectorEnvConfig")

    factories = [_RankedEnvFactory(factory, rank) for rank in range(config.n_envs)]
    if config.backend == "dummy":
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method=config.start_method)
