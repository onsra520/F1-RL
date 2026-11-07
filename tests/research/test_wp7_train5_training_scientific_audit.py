"""TRAIN-5 source-bound training audit: test gates, never train/consume seeds."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from f1rl.research.wp7_train5_v2_protocol import (
    require_training_admission,
    require_heldout_admission,
)
from f1rl.learning.training import runner as runner_module
from f1rl.learning.training.env_factory import TrainingScenarioSet
from f1rl.learning.algorithms import ppo as ppo_module

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "configs/research/wp7-train5-training-contract-scientific-audit-v1.json"
REGISTRATION = ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
SEED_REGISTRY = ROOT / "configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json"

EXPECTED_TRAIN = [
    "straight-sprint-v1", "slalom-v1", "right-turn-train-v2",
    "ramp-v1", "left-turn-train-v2",
]
EXPECTED_HELDOUT = ["left-turn-v1", "offset-slalom-v1"]


def data(path):
    return json.loads(path.read_text(encoding="utf8"))


def test_exact_scientific_audit_read_only_and_training_not_admitted():
    c = data(AUDIT)
    draft = data(REGISTRATION)
    registry = data(SEED_REGISTRY)
    assert c["schema"] == "f1rl_train5_training_contract_scientific_audit_v1"
    assert c["scientific_decision"] == "BLOCKED_NOT_ADMITTED"
    assert c["source_anchor_commit"] == "01ef1e3e75de8df60fc2159ed70b38d0ee777253"
    assert c["source_lock_registration_status"] == draft["status"] == "draft_not_authorized"
    assert registry["registry_gate_status"] == c["global_seed_registry_decision"]
    assert c["training_track_ids"] == draft["train_ids"] == EXPECTED_TRAIN
    assert c["heldout_ids"] == draft["heldout_ids"] == EXPECTED_HELDOUT
    assert c["smoke_cases_finished"] == 6
    assert c["smoke_total_steps"] == 1117
    assert c["smoke_training_seed_count_consumed"] == 0
    assert c["smoke_reset_seed"] == 20261106


@pytest.mark.parametrize("flag", ("training", "finetune", "smoke_rerun", "heldout", "full06", "scientific_release"))
def test_each_scientific_permission_remains_denied(flag):
    assert data(AUDIT)["permissions"][flag] is False


@pytest.mark.parametrize("flag", (
    "owner_separate_scientific_training_protocol_signoff",
    "physical_validation_empirical",
    "train_reset_disturbance_calibration",
    "pre_registered_development_metric_thresholds",
    "compatible_runner_scientific_admission",
    "PPO_backend_scientific_admission",
    "all_required_conditions_pass",
))
def test_missing_scientific_admission_cannot_be_declared_pass(flag):
    assert data(AUDIT)["gates"][flag] is False


def test_ppo_hyperparams_and_five_track_contract_are_precisely_locked_as_proposals():
    c = data(AUDIT)
    plan = c["training_planned"]
    assert plan["policy_runs"] == plan["from_scratch_replicates"] + plan["finetune_replicates"] == 6
    assert plan["timesteps_per_run"] == 200_000
    assert plan["total_timesteps"] == 1_200_000
    assert plan["checkpoint_every"] == 25_000
    assert plan["vector_env_count"] == 8
    assert plan["ppo_hyperparameters"]["n_steps"] == 256
    assert plan["ppo_hyperparameters"]["batch_size"] == 64
    assert plan["ppo_hyperparameters"]["device"] == "cpu"
    assert plan["execution_status"] == "NOT_EXECUTED"
    assert c["training_reset"]["all_five_train_factories_reset_config_is_none"] is True
    assert c["training_reset"]["stochastic_train_reset_protocol_present"] is False
    assert c["training_reset"]["reset_contract_accepted"] is False
    assert c["reward_contract"]["verified_five_track_equal"] is True
    assert c["reward_contract"]["empirical_reward_robustness_validated"] is False
    assert c["action_contract"]["low"] == [-1, 0, 0]
    assert c["action_contract"]["high"] == [1, 1, 1]
    assert c["observation_contract"]["shape"] == [16]
    actual = data(REGISTRATION)["runtime_profile"]
    assert c["runtime_contract"]["ground_motion_version"] == actual["physics_version"]
    assert {key: value for key, value in c["runtime_contract"].items()
            if key != "ground_motion_version"} == {
            key: value for key, value in actual.items()
            if key != "physics_version"}


def test_18_reserved_seeds_still_separate_from_smoke_and_have_no_training_allocation():
    c = data(AUDIT)
    draft = data(REGISTRATION)
    grouped = draft["proposed_training_seeds"]
    assert set(grouped) == {"train5_from_scratch", "train5_from_frozen_checkpoint"}
    seeds = []
    for rows in grouped.values():
        assert len(rows) == 3
        for row in rows:
            assert set(row) == {"ppo", "sampling", "perturbation"}
            seeds.extend(row.values())
    assert len(seeds) == len(set(seeds)) == 18
    assert c["seed_protocol"]["consumed"] == 0
    assert c["seed_protocol"]["seed_authorization_specific_training"] is False
    assert c["smoke_reset_seed"] not in seeds
    assert draft["training_permission"] is False
    assert draft["smoke_permission"] is False


@pytest.mark.parametrize("function", (require_training_admission, require_heldout_admission))
def test_frozen_source_admission_denies_even_with_approved_registry(function):
    with pytest.raises(PermissionError, match="CLOSED"):
        function(True, owner_approval="forged", seed_registry_status="CLOSED")


def test_project_train_runner_hard_stops_before_model_allocation(monkeypatch):
    class Trap(RuntimeError):
        pass
    def deny(*args, **kwargs):
        raise Trap("RESEARCH_TRAINING_DENIED_BEFORE_MODEL")
    monkeypatch.setattr("f1rl.research.wp7_admission.reject_research_training", deny)
    scenarios = object.__new__(TrainingScenarioSet)
    object.__setattr__(
        scenarios,
        "scenarios",
        (SimpleNamespace(factory=SimpleNamespace(requires_scientific_admission=True)),),
    )
    runner = object.__new__(runner_module.TrainingRunner)
    with pytest.raises(Trap, match="RESEARCH_TRAINING_DENIED_BEFORE_MODEL"):
        runner.run_multi(scenarios)


def test_ppo_backend_independently_hard_stops_research_physics(monkeypatch):
    class DummyVecEnv:
        num_envs = 8
        def get_attr(self, field):
            assert field == "requires_scientific_admission"
            return [True] * 8
    class Trap(RuntimeError):
        pass
    def deny(*args, **kwargs):
        raise Trap("PPO_BACKEND_RESEARCH_TRAINING_DENIED")
    monkeypatch.setattr(ppo_module, "VecEnv", DummyVecEnv)
    monkeypatch.setattr(ppo_module, "_local_capability_admitted", lambda env: False)
    monkeypatch.setattr("f1rl.research.wp7_admission.reject_research_training", deny)
    with pytest.raises(Trap, match="PPO_BACKEND_RESEARCH_TRAINING_DENIED"):
        ppo_module._validate_training_admission(DummyVecEnv())
