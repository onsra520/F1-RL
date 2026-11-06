from __future__ import annotations

from copy import deepcopy

import numpy as np


Transition = tuple[np.ndarray, float, bool, bool, dict[str, object]]


class ScriptedEvaluationEnv:
    def __init__(
        self,
        reset_info: dict[str, object],
        transitions: tuple[Transition, ...],
    ) -> None:
        self.reset_info = deepcopy(reset_info)
        self.transitions = transitions
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        self.reset_calls += 1
        return np.zeros(1, dtype=np.float32), deepcopy(self.reset_info)

    def step(self, action: np.ndarray) -> Transition:
        del action
        if self.step_calls >= len(self.transitions):
            raise RuntimeError("scripted environment exhausted")
        transition = self.transitions[self.step_calls]
        self.step_calls += 1
        observation, reward, terminated, truncated, info = transition
        return (
            observation.copy(),
            reward,
            terminated,
            truncated,
            deepcopy(info),
        )

    def close(self) -> None:
        self.close_calls += 1
