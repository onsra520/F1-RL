from types import SimpleNamespace

import pytest
import torch

from f1rl.learning.algorithms import ppo


def test_runtime_configuration_and_limits_use_the_sb3_runtime(monkeypatch):
    assert ppo._sb3_utils.th is torch
    calls = []
    for name in ('set_num_threads', 'set_num_interop_threads', 'use_deterministic_algorithms'):
        monkeypatch.setattr(torch, name, lambda value, name=name: calls.append((name, value)))
    ppo.configure_cpu_runtime(4, 1)
    assert calls == [('set_num_threads', 4), ('set_num_interop_threads', 1),
                     ('use_deterministic_algorithms', True)]
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 4)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 1)
    ppo.validate_cpu_runtime(4)
    monkeypatch.setattr(torch, 'get_num_interop_threads', lambda: 5)
    with pytest.raises(ValueError, match='thread budget exceeded'):
        ppo.validate_cpu_runtime(4)


def test_optimizer_hook_adapter_preserves_callback_and_removable_handle():
    callbacks = []
    handle = SimpleNamespace(remove=lambda: None)
    def register(callback):
        callbacks.append(callback)
        return handle
    model = ppo.PPOModel(SimpleNamespace(policy=SimpleNamespace(
        optimizer=SimpleNamespace(register_step_post_hook=register))))
    def check_budget(*args, **kwargs):
        raise RuntimeError('deadline reached')
    assert model.register_optimizer_step_hook(check_budget) is handle
    with pytest.raises(RuntimeError, match='deadline reached'):
        callbacks[0]()
