"""Source-pinned TRAIN-only numerical fixture and proposed dev metric policy."""
from pathlib import Path
import json
import pytest

from scripts.wp7_train5_v3_numeric_fixture_audit import (
    SPEC, FixtureContractError, load_spec, validate_leaf,
)

ROOT = Path(__file__).resolve().parents[2]


def test_exact_immutable_fixture_roster_and_no_training_authorization():
    spec = load_spec()
    assert spec["schema"] == "f1rl_train5_v3_nontraining_fixture_and_proposed_development_criteria_v1"
    assert spec["source_anchor"] == "8ba236f4ba1f568ad5d6e3f44e9561f47756da74"
    assert len(spec["track_ids"]) == 5
    assert spec["expected_fixture_reset_count"] == 30
    assert spec["fixture_reset_seeds"] == [20261107, 20261108, 20261109]
    assert spec["reset_indices"] == [0, 1]
    assert spec["reset_max_draw_attempts"] == 32
    assert spec["scripted_control_profile"] == {
        "steering": 0, "throttle": 0.5, "brake": 0, "steps_per_reset": 12,
    }
    assert spec["scope"] == "SIMULATOR_NUMERICAL_TRAIN_ONLY_NOT_PHYSICAL_EMPIRICAL_CALIBRATION"
    assert spec["scientific_admission_approved"] is False
    assert spec["empirical_physical_calibration_approved"] is False
    assert spec["permissions"] == {
        "one_shot_engineering_fixture": True, "ppo_smoke_rerun": False,
        "ppo_training": False, "heldout": False, "scientific_release": False,
    }


def test_development_quality_metrics_are_preregistered_proposals_not_tuned_results():
    spec = load_spec()["scientific_quality_criteria_proposal"]
    assert spec["decision"] == "PROPOSED_NOT_APPROVED"
    assert spec["preexisting_dev_design"]["total"] == 3600
    assert spec["preexisting_dev_design"]["cells"] == 60
    assert spec["preexisting_dev_design"]["episodes_per_cell"] == 60
    assert spec["thresholds_are_exploratory_not_owner_signed"] is True
    t = spec["preproposal_thresholds"]
    assert t["uncertified_crash_or_invalid_physics_count_max"] == 0
    assert t["nominal_finish_rate_min"] == .9
    assert t["per_track_finish_rate_min"] == .8
    assert t["hard_route_violation_rate_max"] == .02
    assert t["certified_physical_crash_rate_max"] == .01
    assert "final model at 200000" in spec["checkpoint_rule"]
    assert "HELD-OUT" in spec["comparison_rule"]


def good_leaf():
    return {
        "energy_residual_j": 0.001, "energy_tolerance_j": 0.01,
        "dissipation_j": 0.05, "linear_impulse_residual_ns": 1e-12,
        "angular_impulse_residual_nms": 1e-12,
    }


def test_strict_numerical_limits_accept_conserving_fixture():
    limits=load_spec()["predeclared_numerical_acceptance"]
    assert validate_leaf(good_leaf(), limits) == pytest.approx(.1)


@pytest.mark.parametrize("field,value", (
    ("energy_tolerance_j", 0.),
    ("energy_residual_j", .1),
    ("dissipation_j", -.01),
    ("linear_impulse_residual_ns", 1e-4),
    ("angular_impulse_residual_nms", 1e-4),
    ("energy_residual_j", float("nan")),
    ("dissipation_j", float("inf")),
))
def test_numerical_audit_fails_closed_for_invalid_energy_or_impulse(field,value):
    limits=load_spec()["predeclared_numerical_acceptance"]
    leaf=good_leaf()
    leaf[field]=value
    with pytest.raises(FixtureContractError):
        validate_leaf(leaf, limits)


def test_unapproved_training_is_independently_denied_after_seed_registry_close():
    from f1rl.research.wp7_train5_v2_protocol import require_training_admission
    with pytest.raises(PermissionError, match="CLOSED"):
        require_training_admission(
            owner_signature=True, global_seed_registry_closed=True,
            physical_validation=True,
        )
