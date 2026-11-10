"""Isolated research reward wrapper; does not grant training authority."""
from dataclasses import dataclass
from math import isfinite

import gymnasium as gym

from f1rl.learning.crash_terminal_registered import validate_registered_crash_terminal_info
from f1rl.learning.evaluation.crash_contract import validate_crash_terminal_info


@dataclass(frozen=True)
class TerminalCrashPenaltyConfig:
    amount: float

    def __post_init__(self):
        if isinstance(self.amount, bool) or not isinstance(self.amount, (int, float)) or not isfinite(self.amount) or self.amount < 0:
            raise ValueError('terminal crash penalty must be finite and nonnegative')


class TerminalCrashPenalty(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        if not isinstance(config, TerminalCrashPenaltyConfig):
            raise ValueError('terminal crash penalty configuration required')
        self.config = config

    def step(self, action):
        from scripts.wp7_penalty_trial_capability import before_step
        before_step(self)
        obs, reward, terminated, truncated, info = self.env.step(action)
        outcome = info.get('research_outcome')
        reason = outcome.get('reason') if isinstance(outcome, dict) else None
        crash = reason in ('physical_crash', 'physical_crash_development')
        penalty = 0.0
        if crash:
            if not terminated or truncated or not (validate_registered_crash_terminal_info(info) or validate_crash_terminal_info(info)):
                raise ValueError('invalid physical crash evidence')
            penalty = float(self.config.amount)
        return obs, float(reward) - penalty, terminated, truncated, {
            **info, 'terminal_crash_penalty': -penalty, 'pre_penalty_reward': float(reward),
        }
