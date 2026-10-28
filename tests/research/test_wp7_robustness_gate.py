"""Robustness numerical gate is not scientific freeze, even when green."""
from dataclasses import replace
import pytest

from src.research.wp7_robustness_gate import (
    RobustnessCell,one_sided_cp_lower,evaluate_train_robustness,
)

KEYS=(
    ("ppo","train-alpha","nominal"),
    ("ppo","train-alpha","perturbed"),
)
def row(key,successes=60,episodes=60,seeds=(1,2,3)):
    return RobustnessCell(*key,seeds,successes,episodes,"synthetic-test-evidence-only")


def test_exact_all_success_cp_lower_bound():
    assert one_sided_cp_lower(60,60)==pytest.approx(.05**(1./60),rel=1e-12)
    assert one_sided_cp_lower(0,60)==0.


def test_partial_failures_achieve_lower_bound_less_than_perfect():
    assert one_sided_cp_lower(58,60)<one_sided_cp_lower(60,60)
    assert one_sided_cp_lower(30,60)<.9


def test_perfect_synthetic_cells_may_pass_numeric_but_never_scientific_gate():
    rows=tuple(row(key) for key in KEYS)
    verdict=evaluate_train_robustness(rows,required_strata=KEYS)
    assert verdict.meets_preregistered_numeric_criterion
    assert not verdict.scientifically_accepted
    assert verdict.reasons==()
    assert all(bound>=.9 for _,bound in verdict.lower_bounds)


def test_missing_stratum_and_failed_cp_bound_are_reported():
    rows=(row(KEYS[0],successes=40),)
    verdict=evaluate_train_robustness(rows,required_strata=KEYS)
    assert not verdict.meets_preregistered_numeric_criterion
    assert len(verdict.reasons)==2
    assert not verdict.scientifically_accepted


def test_unpaired_seeds_cannot_pass_across_cells():
    rows=(row(KEYS[0]),row(KEYS[1],seeds=(4,5,6)))
    verdict=evaluate_train_robustness(rows,required_strata=KEYS)
    assert not verdict.meets_preregistered_numeric_criterion
    assert any("paired seed sets" in reason for reason in verdict.reasons)


def test_insufficient_episodes_or_seeds_fail_even_if_no_failures():
    rows=(row(KEYS[0],episodes=59,successes=59,seeds=(1,2)),row(KEYS[1]))
    verdict=evaluate_train_robustness(rows,required_strata=KEYS)
    assert not verdict.meets_preregistered_numeric_criterion
    assert any("insufficient episodes" in reason for reason in verdict.reasons)
    assert any("insufficient paired seeds" in reason for reason in verdict.reasons)


def test_owner_criteria_cannot_be_lowered():
    with pytest.raises(ValueError,match="cannot be weakened"):
        evaluate_train_robustness(
            (row(KEYS[0]),),required_strata=(KEYS[0],),
            minimum_episodes=10,
        )
    with pytest.raises(ValueError,match="cannot be weakened"):
        evaluate_train_robustness(
            (row(KEYS[0]),),required_strata=(KEYS[0],),
            required_lower_bound=.8,
        )


def test_duplicate_strata_and_missing_provenance_rejected():
    with pytest.raises(ValueError,match="duplicate"):
        evaluate_train_robustness(
            (row(KEYS[0]),row(KEYS[0])),required_strata=(KEYS[0],),
        )
    with pytest.raises(ValueError,match="evidence provenance"):
        row(KEYS[0]).__class__("ppo","train-alpha","nominal",(1,2,3),60,60,"")
