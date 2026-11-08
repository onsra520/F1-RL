"""TRAIN-5 v4 future budget and metric contract; no learning or held-out."""
from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path

import pytest
import yaml

from f1rl.research.wp7_train5_v4_budget import (
    BudgetContractError, load_v4_budget, validate_v4_budget,
)
from f1rl.research.wp7_train5_v4_dev_metrics import (
    DevelopmentMetricError, evaluate_dev_metrics_v4,
)
from f1rl.research.wp7_train5_v3_dev_metrics import (
    evaluate_dev_metrics as evaluate_v3, DevelopmentMetricError as V3Error,
    load_registered_development_contract,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY = load_registered_development_contract(ROOT)
PILOT = yaml.safe_load((ROOT / "configs/recovery/wp7-pilot-200k.yml").read_text("utf8"))
TRACKS = LEGACY["track_ids"]


def rows(final_step=200704):
    result = []
    for cohort, replicate, track, condition in product(
        ("scratch", "frozen_initialization"), range(3), TRACKS,
        ("nominal", "perturbation"),
    ):
        digest = sha256(f"{cohort}:{replicate}".encode()).hexdigest()
        for i in range(60):
            result.append(dict(
                cohort=cohort, replicate=replicate, track_id=track,
                condition=condition, episode_index=i,
                checkpoint_step=final_step, checkpoint_sha256=digest,
                outcome="finished", valid_physics=True, episode_return=1.0,
            ))
    return result


def test_rollout_budget_v4_is_actual_not_rounded_down():
    p = load_v4_budget(ROOT)
    result = validate_v4_budget(p, PILOT)
    assert result["rollout_quantum"] == 2048
    assert result["rollouts_per_policy"] == 98
    assert result["actual_final_checkpoint_step"] == 200704
    assert result["overshoot_per_policy"] == 704
    assert result["total_actual_transitions"] == 1204224
    assert result["checkpoint_steps"] == [200704]
    assert result["training_permission"] is False
    assert result["heldout_permission"] is False


@pytest.mark.parametrize("field, value", [
    ("requested_steps", 200704),
    ("rollout_quantum", 2000),
    ("actual_final_checkpoint_step", 200000),
    ("rollouts_per_policy", 97),
    ("checkpoint_steps", [25000, 200704]),
])
def test_budget_tampering_is_rejected(field, value):
    p = deepcopy(load_v4_budget(ROOT))
    p["budget"][field] = value
    with pytest.raises(BudgetContractError):
        validate_v4_budget(p, PILOT)


def test_v4_is_explicitly_not_admitted():
    p = deepcopy(load_v4_budget(ROOT))
    p["permissions"]["ppo_training"] = True
    with pytest.raises(BudgetContractError):
        validate_v4_budget(p, PILOT)


def test_v4_evaluator_accepts_realized_step_but_not_research_admission():
    result = evaluate_dev_metrics_v4(rows(), LEGACY, load_v4_budget(ROOT), PILOT)
    assert result["input_episode_count"] == 3600
    assert result["final_checkpoint_step"] == 200704
    assert result["development_quality_result"] == "METRICS_PASS_NOT_TRAINING_ADMISSION"
    assert result["scientific_training_permission"] is False
    assert result["heldout_permission"] is False


def test_legacy_v3_still_requires_200000_and_is_untouched():
    assert evaluate_v3(rows(200000), LEGACY)["input_episode_count"] == 3600
    with pytest.raises(V3Error):
        evaluate_v3(rows(), LEGACY)


@pytest.mark.parametrize("wrong_step", [200000, 198656, 202752, True])
def test_v4_evaluator_refuses_wrong_checkpoint_steps(wrong_step):
    bad = rows()
    bad[0]["checkpoint_step"] = wrong_step
    with pytest.raises(DevelopmentMetricError):
        evaluate_dev_metrics_v4(bad, LEGACY, load_v4_budget(ROOT), PILOT)


def test_v4_rejects_heldout_track_and_unverified_checkpoint():
    bad = rows()
    bad[0]["track_id"] = "left-turn-v1"
    with pytest.raises(DevelopmentMetricError):
        evaluate_dev_metrics_v4(bad, LEGACY, load_v4_budget(ROOT), PILOT)
    bad = rows()
    bad[0]["checkpoint_sha256"] = "bad"
    with pytest.raises(DevelopmentMetricError):
        evaluate_dev_metrics_v4(bad, LEGACY, load_v4_budget(ROOT), PILOT)
