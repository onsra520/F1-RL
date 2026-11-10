import gymnasium as gym
import pytest

from scripts import wp7_terminal_crash_penalty as module


class Episode(gym.Env):
    def __init__(self, reason, terminated=True, truncated=False):
        self.reason, self.terminated, self.truncated = reason, terminated, truncated

    def step(self, action):
        return 1, 0.25, self.terminated, self.truncated, {'research_outcome': {'reason': self.reason}}


def test_certified_crash_penalized_once(monkeypatch):
    monkeypatch.setattr(module, 'validate_registered_crash_terminal_info', lambda info: True)
    env = module.TerminalCrashPenalty(Episode('physical_crash'), module.TerminalCrashPenaltyConfig(5))
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == -4.75 and terminated and not truncated
    assert info['pre_penalty_reward'] == 0.25 and info['terminal_crash_penalty'] == -5


@pytest.mark.parametrize('reason', ['finish', 'hard_route_violation', 'step_limit'])
def test_other_outcomes_unchanged(reason):
    env = module.TerminalCrashPenalty(Episode(reason), module.TerminalCrashPenaltyConfig(5))
    assert env.step(0)[1] == 0.25


@pytest.mark.parametrize('terminated,truncated', [(True, False), (False, False), (True, True)])
def test_uncertified_or_inconsistent_crash_fails_closed(monkeypatch, terminated, truncated):
    monkeypatch.setattr(module, 'validate_registered_crash_terminal_info', lambda info: False)
    monkeypatch.setattr(module, 'validate_crash_terminal_info', lambda info: False)
    env = module.TerminalCrashPenalty(Episode('physical_crash', terminated, truncated), module.TerminalCrashPenaltyConfig(5))
    with pytest.raises(ValueError, match='invalid physical crash'):
        env.step(0)


@pytest.mark.parametrize('amount', [-1, float('nan'), float('inf'), True])
def test_invalid_amount_rejected(amount):
    with pytest.raises(ValueError):
        module.TerminalCrashPenaltyConfig(amount)
