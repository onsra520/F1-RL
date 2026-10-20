import json
from types import SimpleNamespace

import pytest
import torch

from f1rl.local_demo.runner import (
    _assert_finite_policy_weights,
    _load_smoke_profile,
    _preflight,
)


def test_smoke_profile_is_strictly_bounded_to_one_registered_rollout(tmp_path):
    source = __import__("pathlib").Path(__file__).parents[2] / "configs/research/local-demo-ppo-smoke-v1.json"

    profile, digest = _load_smoke_profile(source)

    assert profile["candidate"] == "C"
    assert profile["tuning_seed_index"] == 0
    assert profile["timesteps"] == profile["rollout_length"] == 256
    assert profile["n_envs"] == profile["rollout_batches"] == 1
    assert profile["tracks"] == ["straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"]
    assert profile["final_checkpoint_required"] is True
    assert profile["heldout_access"] is False
    assert profile["scientific_evaluation"] is False
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("key", "value"),
    [("timesteps", 65536), ("candidate", "D"), ("tuning_seed_index", 1),
     ("n_envs", 2), ("heldout_access", True), ("scientific_evaluation", True)],
)
def test_smoke_profile_rejects_scope_escalation(tmp_path, key, value):
    source = __import__("pathlib").Path(__file__).parents[2] / "configs/research/local-demo-ppo-smoke-v1.json"
    profile = json.loads(source.read_text(encoding="utf-8"))
    profile[key] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="smoke profile"):
        _load_smoke_profile(path)


@pytest.mark.parametrize(("candidate", "seed_index"), [("D", 0), ("C", 1), ("A", 2)])
def test_smoke_preflight_rejects_other_candidates_and_seed_bundles(candidate, seed_index):
    root = __import__("pathlib").Path(__file__).parents[2]

    with pytest.raises(ValueError, match="requires candidate C and tuning seed index 0"):
        _preflight(root, candidate, seed_index, profile="smoke")


def test_smoke_policy_weight_check_rejects_nonfinite_parameters():
    policy = SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0, float("nan")])})
    model = SimpleNamespace(_model=SimpleNamespace(policy=policy))

    with pytest.raises(RuntimeError, match="non-finite PPO policy weight"):
        _assert_finite_policy_weights(model)
