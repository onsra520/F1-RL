"""G2/G5 traceability and local restore tests; fixtures are NOT lab evidence."""
from dataclasses import replace
from hashlib import sha256

import pytest

from src.validation.wp7_calibration_readiness import (
    REQUIRED_UNITS, EvidencePayload, evaluate_calibration_readiness,
    compare_restored_evidence,
)
from src.simulation.tire_calibration_contract import (
    MeasuredPhysicsParameter, PhysicsCalibrationRegister, REQUIRED_WP7_MEASUREMENTS,
)


def fixture_bundle():
    measurements = []
    blobs = []
    for name in REQUIRED_WP7_MEASUREMENTS:
        source = f"synthetic-fixture:{name}".encode("ascii")
        evidence_id = "unit-test-" + name
        value = .7 if name in ("tire_friction_mu", "tire_restitution") else 1.
        measurements.append(MeasuredPhysicsParameter(
            name, value, REQUIRED_UNITS[name], .01, evidence_id,
            "synthetic structural test only - not experimentally calibrated",
        ))
        blobs.append(EvidencePayload(
            evidence_id, source, sha256(source).hexdigest(),
        ))
    verifier = b"synthetic-reviewer-placeholder"
    blobs.append(EvidencePayload(
        "unit-test-reviewer", verifier, sha256(verifier).hexdigest(),
    ))
    return (
        PhysicsCalibrationRegister(tuple(measurements), "unit-test-reviewer"),
        tuple(blobs),
    )


def test_all_structural_fields_and_bytes_can_pass_but_not_scientific_gate():
    register, evidence = fixture_bundle()
    verdict = evaluate_calibration_readiness(register, evidence)
    assert verdict.records_structurally_ready
    assert verdict.checked_evidence_count == len(REQUIRED_WP7_MEASUREMENTS) + 1
    assert not verdict.independently_physically_validated


def test_missing_measurements_and_bytes_are_reported_separately():
    register, evidence = fixture_bundle()
    dropped = PhysicsCalibrationRegister(
        register.measurements[:-1], register.verifier_evidence_id,
    )
    report = evaluate_calibration_readiness(dropped, evidence[:-2] + evidence[-1:])
    assert not report.records_structurally_ready
    assert any("missing measurement:" in reason for reason in report.reasons)


def test_wrong_units_cannot_be_a_valid_physical_register():
    register, evidence = fixture_bundle()
    bad = replace(register.measurements[0], unit="lb")
    changed = PhysicsCalibrationRegister(
        (bad, *register.measurements[1:]),
        register.verifier_evidence_id,
    )
    report = evaluate_calibration_readiness(changed, evidence)
    assert not report.records_structurally_ready
    assert any("unexpected unit" in reason for reason in report.reasons)


def test_tampered_payload_is_not_silently_accepted():
    register, evidence = fixture_bundle()
    bad = replace(evidence[0], payload=b"tampered synthetic values")
    report = evaluate_calibration_readiness(register, (bad, *evidence[1:]))
    assert not report.records_structurally_ready
    assert any("SHA256 mismatch" in reason for reason in report.reasons)


def test_missing_review_payload_blocks_even_when_measurements_complete():
    register, evidence = fixture_bundle()
    report = evaluate_calibration_readiness(register, evidence[:-1])
    assert not report.records_structurally_ready
    assert "independent reviewer evidence bytes missing" in report.reasons


def test_verified_local_restore_does_not_prove_private_drive_backup():
    _, evidence = fixture_bundle()
    verdict = compare_restored_evidence(evidence, tuple(reversed(evidence)))
    assert verdict.bytewise_restore_verified
    assert not verdict.private_google_drive_backup_verified


def test_restore_mismatch_and_missing_items_are_rejected():
    _, evidence = fixture_bundle()
    altered = replace(evidence[1], payload=b"corrupted restore")
    report = compare_restored_evidence(
        evidence, (evidence[0], altered, *evidence[2:]),
    )
    assert not report.bytewise_restore_verified
    assert any("digest mismatch" in reason for reason in report.reasons)
    missing = compare_restored_evidence(evidence, evidence[1:])
    assert not missing.bytewise_restore_verified
    assert any("archive content list differs" in reason for reason in missing.reasons)


def test_duplicate_evidence_rows_rejected():
    register, evidence = fixture_bundle()
    report = evaluate_calibration_readiness(register, (*evidence, evidence[0]))
    assert not report.records_structurally_ready
    assert any("duplicate evidence" in r for r in report.reasons)


def test_invalid_evidence_declarations_fail_closed():
    with pytest.raises(ValueError, match="lowercase SHA256"):
        EvidencePayload("test", b"bytes", "NOT-A-SHA")
    with pytest.raises(ValueError, match="evidence bytes"):
        EvidencePayload("test", b"", "0" * 64)
