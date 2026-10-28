"""WP7 reference gate: check a frozen analytical case matrix, not lab physics.

These cases are independently calculable closed-form baselines. Passing the
matrix confirms a narrow *software/mathematical* verification contract, not
empirical vehicle validity, approved contact restitution, or scientific freeze.
Measured reference data MUST be separately preregistered and reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ReferenceDefinition:
    case_id: str
    unit: str
    absolute_tolerance: float


# Numerical tolerances are solely for closed-form synthetic invariants.
# They are not tolerances for experimental physical validation.
WP7_ANALYTIC_REFERENCES = (
    ReferenceDefinition("gravity_height", "m", 1e-8),
    ReferenceDefinition("gravity_vertical_speed", "m/s", 1e-8),
    ReferenceDefinition("principal_yaw_quaternion_w", "1", 1e-8),
    ReferenceDefinition("principal_yaw_quaternion_y", "1", 1e-8),
    ReferenceDefinition("planar_tire_hub_gap", "m", 1e-9),
    ReferenceDefinition("zero_steer_zero_yaw_moment", "N*m", 1e-9),
)


@dataclass(frozen=True, slots=True)
class AnalyticReferenceObservation:
    case_id: str
    observed: float
    expected: float
    unit: str
    calculation_id: str

    def __post_init__(self) -> None:
        if (not isinstance(self.case_id, str) or not self.case_id.strip()
                or not isinstance(self.unit, str) or not self.unit.strip()
                or not isinstance(self.calculation_id, str)
                or not self.calculation_id.strip()):
            raise ValueError("analytic observations require traceable identifiers")
        for value in (self.observed, self.expected):
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError("analytic quantities must be finite")


@dataclass(frozen=True, slots=True)
class AnalyticReferenceVerdict:
    math_contract_passed: bool
    reasons: tuple[str, ...]
    max_normalized_residual: float

    @property
    def experimental_physics_validated(self) -> bool:
        return False


def verify_analytic_references(
    samples: tuple[AnalyticReferenceObservation, ...],
) -> AnalyticReferenceVerdict:
    if not isinstance(samples, tuple) or any(
        not isinstance(s, AnalyticReferenceObservation) for s in samples
    ):
        raise ValueError("reference observations must be an immutable tuple")
    expected = {d.case_id: d for d in WP7_ANALYTIC_REFERENCES}
    seen: set[str] = set()
    reasons: list[str] = []
    max_normalized = 0.
    for sample in samples:
        if sample.case_id in seen:
            reasons.append(f"duplicate reference case: {sample.case_id}")
            continue
        seen.add(sample.case_id)
        definition = expected.get(sample.case_id)
        if definition is None:
            reasons.append(f"unregistered reference case: {sample.case_id}")
            continue
        if sample.unit != definition.unit:
            reasons.append(f"reference unit mismatch: {sample.case_id}")
            continue
        normalized = abs(sample.observed - sample.expected) / definition.absolute_tolerance
        max_normalized = max(max_normalized, normalized)
        if normalized > 1.:
            reasons.append(f"analytical residual above threshold: {sample.case_id}")
    for required in expected:
        if required not in seen:
            reasons.append(f"missing mandatory reference: {required}")
    return AnalyticReferenceVerdict(not reasons, tuple(reasons), max_normalized)
