"""Canonical simulator-parameter lineage for the simulation-only scope."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from ..sandbox.tire_calibration_contract import REQUIRED_WP7_MEASUREMENTS
from .wp7_calibration_readiness import REQUIRED_UNITS
from .wp7_protocol import load_registration, prepare_scenarios


PROFILE = "spatial_rotating_owner_v3"
REGISTRATION_PATH = "configs/research/wp7-rotating-owner-protocol-v3.json"


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
        if status == "runtime_exact" and runtime_value != config_value:
            raise ValueError(f"source/runtime parameter mismatch for {name}")
        parameters.append({
            "name": name, "unit": unit,
            "source": None if source_key is None else f"{REGISTRATION_PATH}#/" + source_key.replace(".", "/"),
            "configured_value": config_value,
            "runtime_value": runtime_value,
            "runtime_status": status,
            "legal_domain_source": _domain_source(name),
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


def _domain_source(name: str) -> str:
    if name in ("mass_kg", "pitch_inertia_kg_m2", "roll_inertia_kg_m2", "spring_stiffness_npm", "damper_nspm"):
        return "f1rl/sandbox/independent_suspension_dynamics.py:IndependentSuspensionDynamicsPolicy.__post_init__"
    if name == "yaw_inertia_kg_m2":
        return "f1rl/sandbox/spatial_contact_dynamics.py:SpatialContactPolicy.__post_init__"
    if name == "tire_friction_mu":
        return "f1rl/sandbox/coupled_longitudinal_dynamics.py:LongitudinalContactPolicy.__post_init__"
    if name in ("wheel_radius_m", "wheel_width_m"):
        return "f1rl/sandbox/tire_calibration_contract.py:TireHubGeometry.__post_init__ (not consumed by selected runtime profile)"
    if name in ("front_cornering_stiffness_nprad", "rear_cornering_stiffness_nprad"):
        return "f1rl/sandbox/grounded_yaw_dynamics.py:GroundedYawPolicy.__post_init__ (policy absent from selected runtime profile)"
    if name == "tire_restitution":
        return "f1rl/sandbox/contact_manifold.py:ContactManifoldPolicy.__post_init__ (policy absent from selected runtime profile)"
    raise ValueError(f"unregistered WP7 parameter: {name}")
