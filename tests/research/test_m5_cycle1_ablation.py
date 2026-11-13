from types import SimpleNamespace

import pytest
import torch

from scripts.m5_cycle1_ablation import scale_policy_std


def test_std_ablation_changes_only_learned_gaussian_log_std():
    log_std = torch.nn.Parameter(torch.tensor([-0.2, -0.1, 0.0]))
    action_mean_weights = torch.tensor([1.0, 2.0, 3.0])
    model = SimpleNamespace(_model=SimpleNamespace(
        policy=SimpleNamespace(log_std=log_std, action_net=action_mean_weights),
    ))
    before_log_std = log_std.detach().clone()
    before_mean = action_mean_weights.clone()

    scale_policy_std(model, 0.5)

    assert torch.allclose(log_std, before_log_std + torch.log(torch.tensor(0.5)))
    assert torch.equal(action_mean_weights, before_mean)


@pytest.mark.parametrize("scale", [0, 1, -0.5, float("nan"), float("inf")])
def test_std_ablation_rejects_invalid_or_non_reducing_scales(scale):
    model = SimpleNamespace(_model=SimpleNamespace(
        policy=SimpleNamespace(log_std=torch.nn.Parameter(torch.zeros(3))),
    ))

    with pytest.raises(ValueError, match="standard-deviation scale"):
        scale_policy_std(model, scale)


def test_std_ablation_rejects_non_gaussian_policy():
    model = SimpleNamespace(_model=SimpleNamespace(policy=SimpleNamespace()))

    with pytest.raises(ValueError, match="learned log_std"):
        scale_policy_std(model, 0.5)
