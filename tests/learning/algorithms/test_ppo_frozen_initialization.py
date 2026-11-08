from types import SimpleNamespace

import pytest

import f1rl.learning.algorithms.ppo as ppo
import f1rl.research.wp7_train5_v4_runner as v4


def test_initialization_copies_only_policy_into_fresh_model(monkeypatch):
    loaded = []
    policy = SimpleNamespace(load_state_dict=lambda state, strict: loaded.append((state, strict)))
    backend = SimpleNamespace(num_timesteps=0, observation_space="obs", action_space="act", policy=policy)
    model = ppo.PPOModel(backend)
    monkeypatch.setattr(model, "policy_weights_finite", lambda: True)

    def load(archive, *, device):
        assert archive.read() == b"verified bytes"
        assert device == "cpu"
        return {"observation_space": "obs", "action_space": "act"}, {"policy": {"weight": 3}, "policy.optimizer": {"old": 1}}, None

    monkeypatch.setattr(v4, "load_from_zip_file", load)
    v4.initialize_policy_from_archive(model, b"verified bytes")
    assert loaded == [({"weight": 3}, True)]
    assert backend.num_timesteps == 0


def test_initialization_rejects_incompatible_space_and_nonfresh_model(monkeypatch):
    policy = SimpleNamespace(load_state_dict=lambda *args, **kwargs: pytest.fail("must not load"))
    backend = SimpleNamespace(num_timesteps=0, observation_space="obs", action_space="act", policy=policy)
    model = ppo.PPOModel(backend)
    monkeypatch.setattr(v4, "load_from_zip_file", lambda *args, **kwargs: (
        {"observation_space": "other", "action_space": "act"}, {"policy": {}}, None,
    ))
    with pytest.raises(ValueError, match="spaces"):
        v4.initialize_policy_from_archive(model, b"bytes")
    backend.num_timesteps = 1
    with pytest.raises(ValueError, match="fresh"):
        v4.initialize_policy_from_archive(model, b"bytes")


def test_initialization_rejects_nonfinite_policy(monkeypatch):
    policy = SimpleNamespace(load_state_dict=lambda *args, **kwargs: None)
    backend = SimpleNamespace(num_timesteps=0, observation_space="obs", action_space="act", policy=policy)
    model = ppo.PPOModel(backend)
    monkeypatch.setattr(v4, "load_from_zip_file", lambda *args, **kwargs: (
        {"observation_space": "obs", "action_space": "act"}, {"policy": {}}, None,
    ))
    monkeypatch.setattr(model, "policy_weights_finite", lambda: False)
    with pytest.raises(ValueError, match="non-finite"):
        v4.initialize_policy_from_archive(model, b"bytes")
