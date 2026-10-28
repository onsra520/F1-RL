"""G1 frozen reference tests with independent closed-form mathematical oracles.

All data are synthetic reference cases, NOT empirical vehicle measurements.
"""
from math import cos, sin
import pytest

from src.validation.wp7_reference_benchmarks import (
    WP7_ANALYTIC_REFERENCES, AnalyticReferenceObservation,
    verify_analytic_references,
)
from src.simulation.ballistic_flight import _flight_pose
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleState
from src.simulation.torque_free_rotation import (
    AngularRigidBodyState, TorqueFreeRotationPolicy,
    advance_torque_free_rotation,
)
from src.simulation.grounded_yaw_dynamics import (
    GroundedYawPolicy, advance_grounded_yaw,
)
from src.simulation.tire_calibration_contract import (
    TireHubGeometry, sample_spherical_tire_contact,
)
from src.simulation.track.patch import PlanarSurfacePatch

TRACK = PlanarSurfacePatch(
    "plane", Vec3(0., 0., 0.), Vec3(0., 0., 1.),
    Vec3(0., 1., 0.), 10., 20.,
)
HUB = TireHubGeometry((
    Vec3(-.4, 0., .8), Vec3(.4, 0., .8),
    Vec3(-.4, 0., -.8), Vec3(.4, 0., -.8),
), .3, .2)


def closed_form_reference_samples():
    g = 9.81
    dt = .2
    start = VehicleState(
        Vec3(0., 5., 0.), Vec3(1., 1., 2.),
        Quaternion.identity(), Vec3(0., 0., 0.), False,
    )
    flight = _flight_pose(start, dt, g)

    spin = .7
    angular = TorqueFreeRotationPolicy(1800., 1600., 600., integration_steps=96)
    rotated, _ = advance_torque_free_rotation(
        AngularRigidBodyState(Quaternion.identity(), Vec3(0., spin, 0.)),
        angular, dt,
    )

    sample = sample_spherical_tire_contact(
        Vec3(0., .3, 0.), Quaternion.identity(), 0, TRACK, HUB,
    )
    yaw = GroundedYawPolicy(1200., 1600., 1.25, 1.25, 18000., 18000., .8)
    grounded = VehicleState(
        Vec3(0., .3, 0.), Vec3(0., 0., 5.),
        Quaternion.identity(), Vec3(0., 0., 0.), True,
    )
    _, audit = advance_grounded_yaw(
        grounded, grounded, 0., yaw, dt=.05, gravity_mps2=g,
    )

    rows = (
        ("gravity_height", flight.position.y, 5. + 1. * dt - .5 * g * dt**2, "m"),
        ("gravity_vertical_speed", flight.velocity.y, 1. - g * dt, "m/s"),
        ("principal_yaw_quaternion_w", rotated.orientation.w, cos(spin * dt / 2.), "1"),
        ("principal_yaw_quaternion_y", rotated.orientation.y, sin(spin * dt / 2.), "1"),
        ("planar_tire_hub_gap", sample.signed_surface_gap_m, .3 - .3, "m"),
        ("zero_steer_zero_yaw_moment", audit.yaw_moment_nm, 0., "N*m"),
    )
    return tuple(
        AnalyticReferenceObservation(name, observed, expected, unit, "closed-form-oracle-v1")
        for name, observed, expected, unit in rows
    )


def test_complete_frozen_analytic_reference_matrix():
    samples = closed_form_reference_samples()
    assert len(samples) == len(WP7_ANALYTIC_REFERENCES)
    verdict = verify_analytic_references(samples)
    assert verdict.math_contract_passed
    assert not verdict.experimental_physics_validated
    assert verdict.max_normalized_residual <= 1.


def test_missing_case_must_fail_even_if_all_other_cases_pass():
    verdict = verify_analytic_references(closed_form_reference_samples()[:-1])
    assert not verdict.math_contract_passed
    assert any("missing mandatory reference" in r for r in verdict.reasons)


def test_wrong_result_unit_and_duplicate_are_detected():
    original = closed_form_reference_samples()
    tampered = AnalyticReferenceObservation(
        original[0].case_id, original[0].observed + .001,
        original[0].expected, original[0].unit, original[0].calculation_id,
    )
    assert not verify_analytic_references((tampered, *original[1:])).math_contract_passed
    mismatch = AnalyticReferenceObservation(
        original[0].case_id, original[0].observed, original[0].expected,
        "km", original[0].calculation_id,
    )
    assert any("unit mismatch" in r
               for r in verify_analytic_references((mismatch, *original[1:])).reasons)
    assert any("duplicate" in r
               for r in verify_analytic_references((*original, original[0])).reasons)


def test_new_unregistered_case_not_silently_accepted():
    samples = closed_form_reference_samples()
    extra = AnalyticReferenceObservation("new-unreviewed-case", 0., 0., "m", "synthetic")
    verdict = verify_analytic_references((*samples, extra))
    assert not verdict.math_contract_passed
    assert any("unregistered" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("field", ["case_id", "unit", "calculation_id"])
def test_invalid_reference_provenance_fails(field):
    fields = {
        "case_id": "gravity_height", "observed": 1.,
        "expected": 1., "unit": "m", "calculation_id": "reference-v1",
    }
    fields[field] = ""
    with pytest.raises(ValueError):
        AnalyticReferenceObservation(**fields)
