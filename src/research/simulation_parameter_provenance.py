"""Canonical simulator-parameter lineage for the simulation-only scope."""
from __future__ import annotations

from hashlib import sha256
from math import isfinite
from pathlib import Path

from ..sandbox.tire_calibration_contract import REQUIRED_WP7_MEASUREMENTS
from .wp7_calibration_readiness import REQUIRED_UNITS
from .wp7_protocol import load_registration, prepare_scenarios


PROFILE = "spatial_rotating_owner_v3"
REGISTRATION_PATH = "configs/research/wp7-rotating-owner-protocol-v3.json"
_LEGAL_DOMAINS = {
    "mass_kg": (0.0, False, None, False, "independent_suspension_dynamics.py:64-68"),
    "pitch_inertia_kg_m2": (0.0, False, None, False, "independent_suspension_dynamics.py:64-68"),
    "yaw_inertia_kg_m2": (0.0, False, None, False, "spatial_contact_dynamics.py:38-42"),
    "roll_inertia_kg_m2": (0.0, False, None, False, "independent_suspension_dynamics.py:64-68"),
    "wheel_radius_m": (0.0, False, 1.0, True, "tire_calibration_contract.py:29-34"),
    "wheel_width_m": (0.0, False, 1.0, True, "tire_calibration_contract.py:29-34"),
    "spring_stiffness_npm": (0.0, False, None, False, "independent_suspension_dynamics.py:64-68"),
    "damper_nspm": (0.0, True, None, False, "independent_suspension_dynamics.py:69-72"),
    "front_cornering_stiffness_nprad": (0.0, False, None, False, "grounded_yaw_dynamics.py:40-51"),
    "rear_cornering_stiffness_nprad": (0.0, False, None, False, "grounded_yaw_dynamics.py:40-51"),
    "tire_friction_mu": (0.0, True, None, False, "coupled_longitudinal_dynamics.py:39-48"),
    "tire_restitution": (0.0, True, 1.0, True, "contact_manifold.py:47-56"),
}


def build_parameter_provenance(root: str | Path) -> dict:
    """Bind configuration values to the actual WP7 runtime object graph.

    Missing runtime parameters are marked not_consumed; no physical values are
    filled in from measurement placeholders or unrelated models.
    """
    root = Path(root).resolve(strict=True)
    registration_path = root / REGISTRATION_PATH
    registration_bytes = registration_path.read_bytes()
    registration = load_registration(root, profile=PROFILE)
    scenarios = prepare_scenarios(root, registration, seed_index=0)
    if tuple(s.track_id for s in scenarios.scenarios) != (
            "straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"):
        raise ValueError("simulation parameter profile is not the registered TRAIN roster")
    physics = scenarios.scenarios[0].factory.physics_config
    if any(scenario.factory.physics_config.sha256 != physics.sha256
           for scenario in scenarios.scenarios):
        raise ValueError("selected TRAIN tracks do not share one registered runtime physics configuration")
    suspension = physics.independent_suspension_policy
    spatial = physics.spatial_contact_policy
    configured_suspension = registration["engineering_suspension"]
    configured_spatial = registration["engineering_spatial_contact"]

    rows = {
        "mass_kg": ("kg", "engineering_suspension.mass_kg", suspension.mass_kg,
                    configured_suspension["mass_kg"], "runtime_exact"),
        "pitch_inertia_kg_m2": ("kg*m^2", "engineering_suspension.pitch_inertia_kg_m2",
                    suspension.pitch_inertia_kg_m2, configured_suspension["pitch_inertia_kg_m2"], "runtime_exact"),
        "yaw_inertia_kg_m2": ("kg*m^2", "engineering_spatial_contact.yaw_inertia_kg_m2",
                    spatial.yaw_inertia_kg_m2, configured_spatial["yaw_inertia_kg_m2"], "runtime_exact"),
        "roll_inertia_kg_m2": ("kg*m^2", "engineering_suspension.roll_inertia_kg_m2",
                    suspension.roll_inertia_kg_m2, configured_suspension["roll_inertia_kg_m2"], "runtime_exact"),
        "wheel_radius_m": ("m", None, None, None, "not_consumed"),
        "wheel_width_m": ("m", None, None, None, "not_consumed"),
        "spring_stiffness_npm": ("N/m", "engineering_suspension.spring_stiffness_npm",
                    suspension.spring_stiffness_npm, configured_suspension["spring_stiffness_npm"], "runtime_exact"),
        "damper_nspm": ("N*s/m", "engineering_suspension.damper_nspm",
                    suspension.damper_nspm, configured_suspension["damper_nspm"], "runtime_exact"),
        "front_cornering_stiffness_nprad": ("N/rad", None, None, None, "not_consumed"),
        "rear_cornering_stiffness_nprad": ("N/rad", None, None, None, "not_consumed"),
        "tire_friction_mu": ("1", "engineering_spatial_contact.traction.traction_mu",
                    spatial.traction.traction_mu,
                    configured_spatial["traction"]["traction_mu"], "runtime_analogue"),
        "tire_restitution": ("1", None, None, None, "not_consumed"),
    }
    if set(rows) != set(REQUIRED_WP7_MEASUREMENTS):
        raise AssertionError("canonical parameter set differs from WP7 register")
    parameters = []
    for name in REQUIRED_WP7_MEASUREMENTS:
        unit, source_key, runtime_value, config_value, status = rows[name]
        if unit != REQUIRED_UNITS[name]:
            raise AssertionError(f"registered unit mismatch for {name}")
        if status in ("runtime_exact", "runtime_analogue") and runtime_value != config_value:
            raise ValueError(f"source/runtime parameter mismatch for {name}")
        minimum, minimum_inclusive, maximum, maximum_inclusive, bounds_source = _LEGAL_DOMAINS[name]
        for value in (config_value, runtime_value):
            if value is None:
                continue
            if (type(value) not in (int, float) or not _within_domain(
                    value, minimum, minimum_inclusive, maximum, maximum_inclusive)):
                raise ValueError(f"parameter outside source legal domain: {name}")
        parameters.append({
            "name": name, "unit": unit,
            "source": None if source_key is None else f"{REGISTRATION_PATH}#/" + source_key.replace(".", "/"),
            "configured_value": config_value,
            "runtime_value": runtime_value,
            "runtime_status": status,
            "legal_domain_source": f"f1rl/sandbox/{bounds_source}",
            "legal_bounds": {
                "unit": unit, "minimum": minimum, "minimum_inclusive": minimum_inclusive,
                "maximum": maximum, "maximum_inclusive": maximum_inclusive,
                "source": f"f1rl/sandbox/{bounds_source}",
            },
        })
    return {
        "schema_version": "f1rl_simulation_parameter_provenance_v1",
        "status": "source_runtime_identity_checked_simulation_only",
        "profile": PROFILE,
        "configuration_sha256": sha256(registration_bytes).hexdigest(),
        "runtime_physics_sha256": physics.sha256,
        "track_ids": [scenario.track_id for scenario in scenarios.scenarios],
        "parameters": parameters,
        "externally_measured": False,
        "independently_physically_validated": False,
        "limitations": [
            "tire friction source is the spatial traction coefficient and is not a measured tire calibration",
            "wheel hub radius/width, axle cornering stiffness and restitution are absent from this compliant profile",
            "parameter identity does not establish real-world physical accuracy",
        ],
    }


def _within_domain(value: float, minimum: float, minimum_inclusive: bool,
                   maximum: float | None, maximum_inclusive: bool) -> bool:
    if not isfinite(value):
        return False
    if value < minimum or (value == minimum and not minimum_inclusive):
        return False
    if maximum is not None and (value > maximum or (value == maximum and not maximum_inclusive)):
        return False
    return True
