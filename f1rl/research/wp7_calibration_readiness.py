"""Calibration readiness is an evidence-integrity check, not calibration.

A measured number's name/unit/method/uncertainty and a matching SHA256 of
actual submitted evidence bytes are necessary, but never sufficient, for
scientific validity. A separate independent lab review and signed physical
validation protocol remain mandatory before the WP7 scientific freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from ..sandbox.tire_calibration_contract import (
    PhysicsCalibrationRegister, REQUIRED_WP7_MEASUREMENTS,
)

REQUIRED_UNITS = {
    "mass_kg": "kg",
    "pitch_inertia_kg_m2": "kg*m^2",
    "yaw_inertia_kg_m2": "kg*m^2",
    "roll_inertia_kg_m2": "kg*m^2",
    "wheel_radius_m": "m",
    "wheel_width_m": "m",
    "spring_stiffness_npm": "N/m",
    "damper_nspm": "N*s/m",
    "front_cornering_stiffness_nprad": "N/rad",
    "rear_cornering_stiffness_nprad": "N/rad",
    "tire_friction_mu": "1",
    "tire_restitution": "1",
}


@dataclass(frozen=True, slots=True)
class EvidencePayload:
    evidence_id: str
    payload: bytes
    declared_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("evidence ID required")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("evidence bytes required")
        if (not isinstance(self.declared_sha256, str)
                or len(self.declared_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.declared_sha256)):
            raise ValueError("lowercase SHA256 required")

    @property
    def matches_digest(self) -> bool:
        return sha256(self.payload).hexdigest() == self.declared_sha256


@dataclass(frozen=True, slots=True)
class CalibrationReadiness:
    records_structurally_ready: bool
    reasons: tuple[str, ...]
    checked_evidence_count: int

    @property
    def independently_physically_validated(self) -> bool:
        # No algorithm can infer independent experimental validity solely
        # from descriptive metadata and matching hashes.
        return False


def evaluate_calibration_readiness(
    register: PhysicsCalibrationRegister,
    evidence: tuple[EvidencePayload, ...],
) -> CalibrationReadiness:
    if not isinstance(register, PhysicsCalibrationRegister):
        raise ValueError("calibration register is required")
    if not isinstance(evidence, tuple) or any(
        not isinstance(item, EvidencePayload) for item in evidence
    ):
        raise ValueError("evidence must be an immutable tuple of payloads")
    reasons: list[str] = []
    for name in register.missing(REQUIRED_WP7_MEASUREMENTS):
        reasons.append(f"missing measurement: {name}")
    by_name = {m.name: m for m in register.measurements}
    for name, unit in REQUIRED_UNITS.items():
        sample = by_name.get(name)
        if sample is None:
            continue
        if sample.unit != unit:
            reasons.append(f"unexpected unit for {name}")
        if not isfinite(sample.absolute_uncertainty) or sample.absolute_uncertainty < 0.:
            reasons.append(f"invalid uncertainty for {name}")
    evidence_ids = [item.evidence_id for item in evidence]
    if len(set(evidence_ids)) != len(evidence_ids):
        reasons.append("duplicate evidence payload IDs")
    payloads = {item.evidence_id: item for item in evidence}
    for sample in register.measurements:
        blob = payloads.get(sample.evidence_id)
        if blob is None:
            reasons.append(f"measurement evidence bytes missing: {sample.name}")
        elif not blob.matches_digest:
            reasons.append(f"measurement SHA256 mismatch: {sample.name}")
    reviewer = register.verifier_evidence_id
    if not reviewer or reviewer in {m.evidence_id for m in register.measurements}:
        reasons.append("independent reviewer evidence must be distinct from measurement")
    elif reviewer not in payloads:
        reasons.append("independent reviewer evidence bytes missing")
    elif not payloads[reviewer].matches_digest:
        reasons.append("independent reviewer evidence SHA256 mismatch")
    return CalibrationReadiness(
        not reasons, tuple(reasons),
        sum(item.matches_digest for item in evidence),
    )


@dataclass(frozen=True, slots=True)
class EvidenceRestoreAudit:
    bytewise_restore_verified: bool
    reasons: tuple[str, ...]

    @property
    def private_google_drive_backup_verified(self) -> bool:
        # Local/connector supplied byte equality is not independent proof
        # that the actual source was private Google Drive.
        return False


def compare_restored_evidence(
    original: tuple[EvidencePayload, ...],
    restored: tuple[EvidencePayload, ...],
) -> EvidenceRestoreAudit:
    if not isinstance(original, tuple) or not isinstance(restored, tuple):
        raise ValueError("two immutable payload sets required")
    if any(not isinstance(item, EvidencePayload) for item in (*original, *restored)):
        raise ValueError("evidence entries required")
    reasons = []
    names0 = [x.evidence_id for x in original]
    names1 = [x.evidence_id for x in restored]
    if len(set(names0)) != len(names0) or len(set(names1)) != len(names1):
        reasons.append("duplicate archive entries")
    a = {x.evidence_id: x for x in original}
    b = {x.evidence_id: x for x in restored}
    if set(a) != set(b) or not a:
        reasons.append("archive content list differs or is empty")
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if not x.matches_digest or not y.matches_digest:
            reasons.append(f"evidence digest mismatch: {key}")
        if x.declared_sha256 != y.declared_sha256 or x.payload != y.payload:
            reasons.append(f"restored bytes differ: {key}")
    return EvidenceRestoreAudit(not reasons, tuple(reasons))
