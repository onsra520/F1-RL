"""Registered policy binding and admission must fail closed."""
from dataclasses import replace
from pathlib import Path
import pytest

from f1rl.learning.training.runner import TrainingRunner
from f1rl.research.wp7_execution import build_policy_execution

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("index", (0, 1, 2))
def test_replicate_seeds_bind_to_config_and_factories(index):
    b = build_policy_execution(ROOT, f"multitrack-seed{index}")
    assert (b.configuration.ppo.seed, b.configuration.sampling_seed) == (index, index)
    assert {s.factory.reset_config.seed for s in b.scenarios.scenarios} == {1000 + index}
    assert b.receipt["seed_bundle"] == {"ppo": index, "sampling": index, "perturbation": 1000 + index}
    b.validate(b.configuration, b.scenarios)


def test_singletrack_roster_is_bound():
    b = build_policy_execution(ROOT, "singletrack-slalom-v1-seed2")
    assert tuple(s.track_id for s in b.scenarios.scenarios) == ("slalom-v1",)


@pytest.mark.parametrize("mutation", ("ppo", "sampling", "roster", "receipt"))
def test_mismatch_denied_before_artifact_creation(tmp_path, mutation):
    b = build_policy_execution(ROOT, "multitrack-seed2")
    cfg = b.configuration
    if mutation == "ppo":
        cfg = replace(cfg, ppo=replace(cfg.ppo, seed=0))
    elif mutation == "sampling":
        cfg = replace(cfg, sampling_seed=0)
    elif mutation == "roster":
        b = replace(b, scenarios=replace(b.scenarios, scenarios=b.scenarios.scenarios[:1]))
    else:
        b = replace(b, receipt={**b.receipt, "training_authorized": True})
    cfg = replace(cfg, artifacts_root=tmp_path / "no-artifacts")
    with pytest.raises(ValueError, match="registered|configuration"):
        TrainingRunner(cfg).run_registered(b)
    assert not cfg.artifacts_root.exists()


def test_exact_binding_does_not_grant_permission():
    b = build_policy_execution(ROOT, "multitrack-seed1")
    from f1rl.research.wp7_admission import ScientificAdmissionError
    with pytest.raises(ScientificAdmissionError):
        TrainingRunner(b.configuration).run_registered(b)
