from __future__ import annotations

import numpy as np


class ConstantPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.deterministic_calls: list[bool] = []
        self.observations: list[np.ndarray] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        self.deterministic_calls.append(deterministic)
        self.observations.append(np.asarray(observation).copy())
        return self.action.copy()


class ScriptedPolicy:
    def __init__(self, actions: tuple[np.ndarray, ...]) -> None:
        self.actions = tuple(np.asarray(action, dtype=np.float32) for action in actions)
        self.deterministic_calls: list[bool] = []
        self._index = 0

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        del observation
        self.deterministic_calls.append(deterministic)
        if self._index >= len(self.actions):
            raise RuntimeError("scripted policy exhausted")
        action = self.actions[self._index]
        self._index += 1
        return action.copy()


class FailingPolicy:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        del observation, deterministic
        self.calls += 1
        raise self.error
