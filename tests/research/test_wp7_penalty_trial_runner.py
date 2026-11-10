import gymnasium as gym
import numpy as np
import pytest
from scripts import wp7_penalty_trial_runner as runner


class Scenarios:
    scenarios = tuple(range(5))


class Sampled(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    def __init__(self, scenarios, seed, rank):
        self.scenarios, self.seed, self.rank = scenarios, seed, rank
    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(1, dtype=np.float32), 1., False, False, {'research_outcome': {'reason': None}}


@pytest.mark.parametrize('intervention,amount', [('baseline', 0), ('terminal_crash_penalty', 5)])
def test_fake_vector_environment_uses_eight_wrapped_selectors(monkeypatch, intervention, amount):
    monkeypatch.setattr(runner, 'TrainingScenarioSet', Scenarios)
    monkeypatch.setattr(runner, '_SampledF1RLEnv', Sampled)
    scenarios = Scenarios()
    env = runner.build_trial_environment(scenarios, sampling_seed=12, intervention=intervention)
    try:
        assert len(env.envs) == 8
        assert [e.unwrapped.rank for e in env.envs] == list(range(8))
        assert all(e.config.amount == amount and e.unwrapped.seed == 12 for e in env.envs)
        env.reset()
        _, reward, _, _ = env.step(np.zeros(8, dtype=np.int64))
        assert list(reward) == [1.] * 8
    finally:
        env.close()


def test_runner_remains_fail_closed_without_capability():
    with pytest.raises(TypeError):
        runner.run_trial({})
