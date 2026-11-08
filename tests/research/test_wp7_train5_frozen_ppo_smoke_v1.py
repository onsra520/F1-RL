"""A separately versioned six-case smoke cannot confer TRAIN-5 admission."""
import json
from pathlib import Path

import pytest

from scripts import wp7_train5_frozen_ppo_smoke_v1 as smoke
from f1rl.research import wp7_train5_v2_protocol as draft

ROOT = Path(__file__).resolve().parents[2]


def test_scoped_smoke_has_exact_train_v2_models_seed_and_one_shot():
    p = json.loads(smoke.PROTOCOL.read_text(encoding="utf8"))
    registration = json.loads((
        ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
    ).read_text(encoding="utf8"))
    assert p["schema"] == "f1rl_train5_immutable_six_case_smoke_v1"
    assert p["status"] == "APPROVED_BY_OWNER_FOR_FROZEN_PPO_SMOKE_ONLY"
    assert p["branch"] == "handoff/physics-v4-khanh-bang-20261106"
    assert p["frozen_source_base_commit"] == smoke.ANCHOR
    assert tuple(p["model_ids"]) == ("C-seed0", "B-seed1", "A-seed2")
    assert tuple(p["track_ids"]) == ("right-turn-train-v2", "left-turn-train-v2")
    assert set(p["track_ids"]).issubset(registration["train_ids"])
    assert not (set(p["track_ids"]) & set(registration["heldout_ids"]))
    assert p["reset_seed"] == 20261106
    assert p["max_steps_per_case"] == 2500
    assert p["required_case_count"] == 6
    assert p["one_shot"] is True
    assert p["device"] == "cpu"
    assert p["deterministic_prediction"] is True
    assert p["separate_training_protocol_required"] is True
    assert p["runtime_profile"] == registration["runtime_profile"]
    assert p["allowed_output_root"] == str(smoke.EVIDENCE)


@pytest.mark.parametrize("key", (
    "reuse_training_seeds",
    "read_heldout_geometry",
    "train_ppo",
    "allow_training_after_smoke",
))
def test_smoke_never_admits_training_or_heldout(key):
    p = json.loads(smoke.PROTOCOL.read_text(encoding="utf8"))
    assert p[key] is False
    with pytest.raises(PermissionError, match="CLOSED"):
        draft.require_training_admission("forged")
    with pytest.raises(PermissionError, match="CLOSED"):
        draft.require_heldout_admission("forged")


def test_classification_is_conservative_for_unknown_termination():
    assert smoke.classify({"finished": True}, True, False) == "finished"
    assert smoke.classify({"research_outcome": {"reason": "step_limit"}},
                          False, True) == "valid_step_limit"
    assert smoke.classify({"research_outcome": {"reason": "hard_route_violation"}},
                          True, False) == "valid_hard_route_violation"
    assert smoke.classify({"research_outcome": {"reason": "physical_crash"}},
                          True, False) == "uncertified_physical_crash"
    assert smoke.classify({}, True, False) == "runtime_or_contract_exception"
    assert smoke.classify({}, False, False) is None


def test_other_training_seeds_not_used_for_smoke():
    p = json.loads(smoke.PROTOCOL.read_text(encoding="utf8"))
    reg = json.loads((
        ROOT / "configs/research/wp7-train5-paired-v2-registration.json"
    ).read_text(encoding="utf8"))
    reserved = {
        seed
        for rows in reg["proposed_training_seeds"].values()
        for bundle in rows
        for seed in bundle.values()
    }
    assert len(reserved) == 18
    assert p["reset_seed"] not in reserved
    assert set(p["model_ids"]) == {"C-seed0", "B-seed1", "A-seed2"}


def test_smoke_script_has_no_learner_execution_path():
    text = (ROOT / "scripts/wp7_train5_frozen_ppo_smoke_v1.py").read_text("utf8")
    assert ".learn(" not in text
    assert ".train(" not in text
    assert "require_training_admission(" not in text
    assert "one-shot smoke already attempted" in text
    assert "PPOModel.load(" in text
    assert "deterministic=True" in text
