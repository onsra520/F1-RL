from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from ..gym_environment import F1RLEnv
from .config import MAX_SEED, VectorEnvConfig
from .env_factory import F1RLEnvFactory, TrainingScenario, TrainingScenarioSet


@dataclass(frozen=True, slots=True)
class _RankedEnvFactory:
    factory: F1RLEnvFactory
    rank: int

    def __call__(self) -> F1RLEnv:
        if self.factory.reset_config is not None:
            return self.factory.create(reset_rank=self.rank)
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


class _SampledF1RLEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, scenarios: TrainingScenarioSet, sampling_seed: int, rank: int) -> None:
        if not isinstance(scenarios, TrainingScenarioSet):
            raise ValueError("scenarios must be TrainingScenarioSet")
        if (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or not 0 <= sampling_seed <= MAX_SEED
        ):
            raise ValueError(f"sampling_seed must be an integer in [0, {MAX_SEED}]")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        super().__init__()
        rank_seed = int.from_bytes(
            hashlib.sha256(
                f"f1rl/multi-track-sampling/v1/{sampling_seed}/{rank}".encode("utf-8")
            ).digest(),
            "big",
        )
        self._scenarios = scenarios
        self._sampling_seed = sampling_seed
        self._selection_rng = random.Random(rank_seed)
        self._rank = rank
        self._reset_sequence = 0
        self._active_env: F1RLEnv | None = None
        self._training_telemetry_enabled = False
        self._current_scenario: TrainingScenario | None = None
        self.action_space = scenarios.action_space
        self.observation_space = scenarios.observation_space

    @property
    def requires_scientific_admission(self) -> bool:
        return any(s.factory.requires_scientific_admission for s in self._scenarios.scenarios)

    @property
    def training_reset_contracts(self) -> dict[str, dict[str, object]]:
        contracts = {}
        for scenario in self._scenarios.scenarios:
            config = scenario.factory.reset_config
            if config is None:
                raise ValueError("sampled TRAIN telemetry requires reset contracts for every scenario")
            contracts[scenario.track_id] = config.identity_for_track(scenario.track_id)
        return contracts

    def set_training_telemetry_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("training telemetry flag must be bool")
        self._training_telemetry_enabled = enabled
        if self._active_env is not None:
            self._active_env.set_training_telemetry_enabled(enabled)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        previous_env = self._active_env
        self._active_env = None
        self._current_scenario = None
        if previous_env is not None:
            previous_env.close()

        scenario = self._selection_rng.choice(self._scenarios.scenarios)
        env = (
            scenario.factory.create(reset_rank=self._rank, reset_sequence=self._reset_sequence)
            if scenario.factory.reset_config is not None else scenario.factory.create()
        )
        try:
            env.set_training_telemetry_enabled(self._training_telemetry_enabled)
            observation, info = env.reset(seed=seed, options=options)
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            raise

        self._active_env = env
        self._current_scenario = scenario
        self._reset_sequence += 1
        return observation, {**info, "track_id": scenario.track_id}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        from ...research.wp7_train5_internal_execution import check_active_step
        check_active_step(self)
        if self._active_env is None or self._current_scenario is None:
            raise RuntimeError("reset() must be called before step()")
        observation, reward, terminated, truncated, info = self._active_env.step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, "track_id": self._current_scenario.track_id},
        )

    def close(self) -> None:
        active_env = self._active_env
        self._active_env = None
        self._current_scenario = None
        if active_env is not None:
            active_env.close()


@dataclass(frozen=True, slots=True)
class _RankedScenarioEnvFactory:
    scenarios: TrainingScenarioSet
    sampling_seed: int
    rank: int

    def __call__(self) -> _SampledF1RLEnv:
        return _SampledF1RLEnv(self.scenarios, self.sampling_seed, self.rank)


def make_multi_track_vector_env(
    scenarios: TrainingScenarioSet,
    config: VectorEnvConfig,
    sampling_seed: int,
) -> VecEnv:
    if not isinstance(scenarios, TrainingScenarioSet):
        raise ValueError("scenarios must be TrainingScenarioSet")
    if not isinstance(config, VectorEnvConfig):
        raise ValueError("config must be VectorEnvConfig")
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or not 0 <= sampling_seed <= MAX_SEED
    ):
        raise ValueError(f"sampling_seed must be an integer in [0, {MAX_SEED}]")

    factories = [
        _RankedScenarioEnvFactory(scenarios, sampling_seed, rank)
        for rank in range(config.n_envs)
    ]
    if config.backend == "dummy":
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method=config.start_method)
