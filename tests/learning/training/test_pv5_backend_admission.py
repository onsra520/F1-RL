"""No training: blocked paths must reject before entering SB3 learn."""
import pytest

from src.learning.algorithms.ppo import PPO, PPOConfig, PPOModel
from tests.learning.training.mocks.mock_factories import make_factory
from tests.integration.test_wp7_physics_integration import factory, BODY, SUSPENSION
from src.runtime.physics import PhysicsConfig


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


@pytest.mark.parametrize("backend", ["dummy", "subproc"])
@pytest.mark.parametrize("entrypoint", ["create", "backend_create", "raw_learn"])
def test_real_vector_backend_cannot_bypass_any_project_learning_entrypoint(backend, entrypoint, monkeypatch):
    from stable_baselines3 import PPO as SB3PPO
    from src.learning.training.config import VectorEnvConfig
    from src.learning.training.vector_env import make_vector_env
    candidate = factory(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION))
    vector = make_vector_env(candidate, VectorEnvConfig(n_envs=1, backend=backend))
    baseline = make_factory().create()
    try:
        assert vector.get_attr("requires_scientific_admission") == [True]
        if entrypoint == "raw_learn":
            model = PPOModel.create(baseline, PPOConfig(n_steps=2, batch_size=2))
            model._model.set_env(vector)
            def forbidden(*args, **kwargs):
                raise AssertionError("SB3 collected a rollout before admission")
            monkeypatch.setattr(SB3PPO, "learn", forbidden)
            invoke = lambda: model._model.learn(2)
        elif entrypoint == "create":
            invoke = lambda: PPOModel.create(vector, PPOConfig(n_steps=2, batch_size=2))
        else:
            invoke = lambda: PPO("MlpPolicy", vector, n_steps=2, batch_size=2)
        with pytest.raises(ValueError, match="blocked") as caught:
            invoke()
        assert caught.value.report["training_authorized"] is False
    finally:
        vector.close()
        baseline.close()


@pytest.mark.parametrize("flags", [[], [0], ["false"], [None], [False, False]])
def test_malformed_vector_admission_metadata_never_allocates_backend(flags, monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv
    vector = DummyVecEnv([make_factory().create])
    try:
        monkeypatch.setattr(vector, "get_attr", lambda *args, **kwargs: flags)
        with pytest.raises(ValueError, match="metadata"):
            PPOModel.create(vector, PPOConfig(n_steps=2, batch_size=2))
    finally:
        vector.close()


def test_missing_vector_admission_metadata_fails_closed(monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv
    vector = DummyVecEnv([make_factory().create])
    def missing(*args, **kwargs):
        raise AttributeError("requires_scientific_admission")
    try:
        monkeypatch.setattr(vector, "get_attr", missing)
        with pytest.raises(ValueError, match="metadata"):
            PPOModel.create(vector, PPOConfig(n_steps=2, batch_size=2))
    finally:
        vector.close()
