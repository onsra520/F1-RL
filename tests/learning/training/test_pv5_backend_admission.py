"""No training: blocked paths must reject before entering SB3 learn."""
import pytest

from f1rl.learning.algorithms.ppo import PPO, PPOConfig, PPOModel
from tests.learning.training.mocks.mock_factories import make_factory
from tests.integration.test_wp7_physics_integration import factory, BODY, SUSPENSION
from f1rl.runtime.physics import PhysicsConfig


def research_env():
    return factory(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION)).create()


def test_direct_project_backend_constructor_cannot_bypass_guard():
    env = research_env()
    try:
        with pytest.raises(ValueError, match="blocked"):
            PPO("MlpPolicy", env, n_steps=2, batch_size=2)
    finally:
        env.close()


def test_raw_backend_learn_rechecks_reattached_env_without_collecting_samples(monkeypatch):
    from stable_baselines3 import PPO as SB3PPO
    baseline, research = make_factory().create(), research_env()
    try:
        model = PPOModel.create(baseline, PPOConfig(n_steps=2, batch_size=2))
        model._model.set_env(research)
        def forbidden(*args, **kwargs):
            raise AssertionError("SB3 learn reached before admission")
        monkeypatch.setattr(SB3PPO, "learn", forbidden)
        with pytest.raises(ValueError, match="blocked"):
            model._model.learn(2)
    finally:
        baseline.close()
        research.close()


def test_prediction_only_load_cannot_later_train_research_env(tmp_path, monkeypatch):
    from stable_baselines3 import PPO as SB3PPO
    baseline, research = make_factory().create(), research_env()
    try:
        path = PPOModel.create(baseline, PPOConfig(n_steps=2, batch_size=2)).save(tmp_path / "untrained")
        model = PPOModel.load(path, env=research)
        def forbidden(*args, **kwargs):
            raise AssertionError("SB3 learn reached after load")
        monkeypatch.setattr(SB3PPO, "learn", forbidden)
        with pytest.raises(ValueError, match="blocked"):
            model._model.learn(2)
    finally:
        baseline.close()
        research.close()
