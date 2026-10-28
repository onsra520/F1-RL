"""Verify exact one-sided Bernoulli lower bound and non-pooling."""
from math import isclose

import pytest

from src.learning.robustness_acceptance import (
    clopper_pearson_lower_bound,
    evaluate_preregistered_track,
)


def test_exact_one_sided_endpoints() -> None:
    assert clopper_pearson_lower_bound(0, 60) == 0.
    assert clopper_pearson_lower_bound(60, 60) == pytest.approx(.05**(1./60.))
    assert clopper_pearson_lower_bound(1,1) == pytest.approx(.05)


def test_lower_bound_is_strictly_below_observed_success_rate() -> None:
    for successes in (10,30,50,59):
        lower=clopper_pearson_lower_bound(successes,60)
        assert 0. < lower < successes/60.


def test_strictly_increasing_with_successes_and_sample_count() -> None:
    assert clopper_pearson_lower_bound(50,60) < clopper_pearson_lower_bound(55,60)
    assert clopper_pearson_lower_bound(60,60) < clopper_pearson_lower_bound(120,120)


def test_prespecified_90_percent_bound_not_raw_rate() -> None:
    fully_successful=evaluate_preregistered_track(
        track_id="synthetic-train", policy_id="policy-001",
        condition="nominal", successes=60, episodes=60)
    assert fully_successful.accepted
    assert fully_successful.lower_bound>=.90
    raw_success_rate_90_percent=evaluate_preregistered_track(
        track_id="synthetic-train",policy_id="policy-001",
        condition="randomized_train",successes=54,episodes=60)
    assert not raw_success_rate_90_percent.accepted
    assert raw_success_rate_90_percent.lower_bound < .90


def test_nominal_and_randomized_are_separate_strata() -> None:
    nominal=evaluate_preregistered_track(
        track_id="train-1", policy_id="model-A",
        condition="nominal",successes=60,episodes=60)
    randomized=evaluate_preregistered_track(
        track_id="train-1", policy_id="model-A",
        condition="randomized_train",successes=50,episodes=60)
    assert nominal.accepted
    assert not randomized.accepted
    assert nominal.condition!=randomized.condition


@pytest.mark.parametrize("successes,episodes",[
    (-1,60),(61,60),(5,0),(True,60),(5,True),(3.1,60),(3,-2),
])
def test_invalid_count_rejected(successes,episodes) -> None:
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(successes,episodes)


@pytest.mark.parametrize("confidence",[0.,1.,-1.,float("nan"),True])
def test_invalid_confidence_rejected(confidence) -> None:
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(55,60,confidence=confidence)


@pytest.mark.parametrize("condition",["heldout","train","randomized_heldout",""])
def test_condition_must_be_named_train_only(condition) -> None:
    with pytest.raises(ValueError,match="TRAIN"):
        evaluate_preregistered_track(
            track_id="synthetic",policy_id="A",condition=condition,
            successes=60,episodes=60,
        )


def test_partial_eval_budget_does_not_get_a_lucky_early_pass() -> None:
    with pytest.raises(ValueError,match="prespecified evaluation budget"):
        evaluate_preregistered_track(
            track_id="train-1",policy_id="model-A",condition="nominal",
            successes=10,episodes=10,
        )


def test_missing_identifiers_are_rejected() -> None:
    with pytest.raises(ValueError,match="track_id"):
        evaluate_preregistered_track(
            track_id="",policy_id="model-A",condition="nominal",
            successes=60,episodes=60,
        )
