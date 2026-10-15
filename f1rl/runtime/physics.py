"""Versioned research physics registration; never scientific training admission.

Longitudinal motion remains legacy-prescribed. Contacts are suspension
reference points, not tire hubs/patches. No physical constants are inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isclose

from ..sandbox.ballistic_flight import BallisticFlightPolicy, FlightState
from ..sandbox.body_collision import VehicleBodyGeometry
from ..sandbox.grounded_yaw_dynamics import GroundedYawPolicy
from ..sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
)
from ..sandbox.partial_wheel_release import AutomaticTakeoffPolicy
from ..sandbox.vehicle import VehicleParameters
from ..sandbox.world import WorldState


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    mode: str
    body_geometry: VehicleBodyGeometry | None = None
    independent_suspension_policy: IndependentSuspensionDynamicsPolicy | None = None
    ballistic_flight_policy: BallisticFlightPolicy | None = None
    automatic_takeoff_policy: AutomaticTakeoffPolicy | None = None
    grounded_yaw_policy: GroundedYawPolicy | None = None
    contact_geometry: str = "suspension_reference_points_v1"

    def __post_init__(self) -> None:
        if self.mode not in (
            "legacy_v1", "wp7_suspension_v1", "wp7_ballistic_v1", "wp7_grounded_yaw_v1",
        ):
            raise ValueError("unregistered physics mode")
        if self.contact_geometry != "suspension_reference_points_v1":
            raise ValueError("tire-hub/contact-patch dynamics are not integrated")
        policies = (
            (self.body_geometry, VehicleBodyGeometry),
            (self.independent_suspension_policy, IndependentSuspensionDynamicsPolicy),
            (self.ballistic_flight_policy, BallisticFlightPolicy),
            (self.automatic_takeoff_policy, AutomaticTakeoffPolicy),
            (self.grounded_yaw_policy, GroundedYawPolicy),
        )
        if any(value is not None and not isinstance(value, kind) for value, kind in policies):
            raise ValueError("physics policies must have their registered types")
        if self.mode == "legacy_v1":
            if any(value is not None for value, _ in policies):
                raise ValueError("legacy physics cannot contain research policies")
            return
        body = self.body_geometry
        suspension = self.independent_suspension_policy
        if body is None or suspension is None:
            raise ValueError("WP7 requires explicit body geometry and independent suspension")
        if (body.length_m, body.width_m, body.height_m) != (4., 1.8, 1.2):
            raise ValueError("WP7 owner body must remain 4.0 x 1.8 x 1.2 m")
        flight = self.ballistic_flight_policy
        yaw = self.grounded_yaw_policy
        if (flight is not None) != (self.mode == "wp7_ballistic_v1"):
            raise ValueError("ballistic mode requires its explicit flight policy exclusively")
        if (yaw is not None) != (self.mode == "wp7_grounded_yaw_v1"):
            raise ValueError("grounded yaw mode requires its explicit yaw policy exclusively")
        if self.automatic_takeoff_policy is not None and flight is None:
            raise ValueError("automatic takeoff requires ballistic mode")
        if flight is not None:
            if flight.contact != suspension.contact:
                raise ValueError("flight and suspension contact contracts must match")
            angular = flight.torque_free
            if angular is not None:
                if (angular.pitch_inertia_kg_m2 != suspension.pitch_inertia_kg_m2
                        or angular.roll_inertia_kg_m2 != suspension.roll_inertia_kg_m2
                        or flight.manifold is None
                        or angular.yaw_inertia_kg_m2 != flight.manifold.yaw_inertia_kg_m2):
                    raise ValueError("torque-free and manifold inertia registration must match suspension")

    def validate_state(self, initial: WorldState, parameters: VehicleParameters) -> None:
        if self.mode == "legacy_v1":
            if initial.suspension is not None or initial.flight is not None:
                raise ValueError("persistent research state requires an explicit WP7 physics mode")
            return
        if not isinstance(initial.suspension, SuspensionDynamicsState):
            raise ValueError("WP7 requires persistent suspension state")
        if not isclose(initial.vehicle.position.y, initial.suspension.height_m, abs_tol=1e-8):
            raise ValueError("initial suspension height disagrees with vehicle")
        if self.ballistic_flight_policy is None:
            if initial.flight is not None or not initial.vehicle.grounded:
                raise ValueError("airborne state requires registered ballistic mode")
        else:
            if not isinstance(initial.flight, FlightState):
                raise ValueError("ballistic mode requires persistent flight state")
            if initial.vehicle.grounded != (initial.flight.phase != "airborne"):
                raise ValueError("initial flight phase disagrees with grounded flag")
        yaw = self.grounded_yaw_policy
        if yaw is not None:
            if not isclose(yaw.mass_kg, self.independent_suspension_policy.mass_kg, abs_tol=1e-8):
                raise ValueError("grounded yaw and suspension mass must match")
            if not isclose(yaw.front_axle_distance_m + yaw.rear_axle_distance_m,
                           parameters.wheelbase, abs_tol=1e-8):
                raise ValueError("grounded yaw axle geometry must match vehicle wheelbase")

    def world_options(self) -> dict[str, object]:
        names = (
            "body_geometry", "independent_suspension_policy", "ballistic_flight_policy",
            "automatic_takeoff_policy", "grounded_yaw_policy",
        )
        return {name: getattr(self, name) for name in names if getattr(self, name) is not None}

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "validation_scope": "software_research_only",
            "longitudinal_model": "legacy_prescribed_v1",
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return sha256(json.dumps(
            self.document(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
