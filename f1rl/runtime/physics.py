"""Versioned research physics registration; never scientific training admission.

Longitudinal motion remains legacy-prescribed. Contacts are suspension
reference points, not tire hubs/patches. No physical constants are inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isclose, atan2

from ..sandbox.ballistic_flight import BallisticFlightPolicy, FlightState, FlightTransition
from ..sandbox.center_trajectory import CenterSegment
from ..sandbox.body_collision import VehicleBodyGeometry
from ..sandbox.grounded_yaw_dynamics import GroundedYawPolicy
from ..sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState, _horizontal_yaw,
    angular_velocity_for_suspension,
    suspension_orientation,
)
from ..sandbox.geometry.vector import Vec3
from ..sandbox.partial_wheel_release import AutomaticTakeoffPolicy
from ..sandbox.interactions.events import ObstacleHit
from ..sandbox.vehicle import VehicleParameters
from ..sandbox.world import WorldState, WorldStepResult
from ..sandbox.track.model import TrackWorld


class PhysicsTransitionError(RuntimeError):
    """Rejected research tick; no physical state or race result is published."""


@dataclass(frozen=True, slots=True)
class PhysicsTransitionAudit:
    version: str
    physical_mode: str
    initial_phase: str
    final_phase: str
    start_time_s: float
    end_time_s: float
    solvers: tuple[str, ...]
    events: tuple[FlightTransition, ...]


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
            "wp7_lifecycle_v1",
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
        if (flight is not None) != (self.mode in ("wp7_ballistic_v1", "wp7_lifecycle_v1")):
            raise ValueError("ballistic mode requires its explicit flight policy exclusively")
        if (yaw is not None) != (self.mode == "wp7_grounded_yaw_v1"):
            raise ValueError("grounded yaw mode requires its explicit yaw policy exclusively")
        if self.automatic_takeoff_policy is not None and flight is None:
            raise ValueError("automatic takeoff requires ballistic mode")
        if self.mode == "wp7_lifecycle_v1":
            if (suspension.angular_rate_mapping != "yaw_rx_rz_v1"
                    or flight.manifold is None or flight.torque_free is not None
                    or flight.angular_impulse is not None or self.automatic_takeoff_policy is None
                    or flight.max_patch_tilt_rad != 0.
                    or self.automatic_takeoff_policy.max_patch_tilt_rad != 0.
                    or flight.manifold.max_ground_yaw_radps > 1e-8):
                raise ValueError("lifecycle requires exact rate mapping, flat manifold/release and zero-yaw recovery; variable-omega TOI unsupported")
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

    def validate_state(self, initial: WorldState, parameters: VehicleParameters, *, _prescribed_yaw_rate=0.) -> None:
        if self.mode == "legacy_v1":
            if initial.suspension is not None or initial.flight is not None:
                raise ValueError("persistent research state requires an explicit WP7 physics mode")
            return
        if not isinstance(initial.suspension, SuspensionDynamicsState):
            raise ValueError("WP7 requires persistent suspension state")
        if initial.vehicle.grounded and not isclose(
            initial.vehicle.position.y, initial.suspension.height_m, rel_tol=0., abs_tol=1e-8,
        ):
            raise ValueError("initial suspension height disagrees with vehicle")
        if self.ballistic_flight_policy is None:
            if initial.flight is not None or not initial.vehicle.grounded:
                raise ValueError("airborne state requires registered ballistic mode")
        else:
            if not isinstance(initial.flight, FlightState):
                raise ValueError("ballistic mode requires persistent flight state")
            if initial.vehicle.grounded != (initial.flight.phase != "airborne"):
                raise ValueError("initial flight phase disagrees with grounded flag")
        explicit_launch = initial.flight is not None and initial.flight.phase == "takeoff"
        if initial.vehicle.grounded and not explicit_launch:
            state = initial.suspension
            if self.mode == "wp7_lifecycle_v1":
                expected_q = suspension_orientation(_horizontal_yaw(initial.vehicle.orientation), state)
                q = initial.vehicle.orientation
                if abs(abs(q.w*expected_q.w + q.x*expected_q.x + q.y*expected_q.y + q.z*expected_q.z)-1.) > 1e-10:
                    raise ValueError("grounded chassis and suspension attitude disagree")
            expected = angular_velocity_for_suspension(
                _horizontal_yaw(initial.vehicle.orientation), state, _prescribed_yaw_rate,
                self.independent_suspension_policy.angular_rate_mapping,
            )
            actual = initial.vehicle.angular_velocity
            pairs = ((initial.vehicle.velocity.y, state.heave_velocity_mps),
                     (actual.x, expected.x), (actual.z, expected.z))
            if self.mode == "wp7_lifecycle_v1":
                pairs += ((actual.y, expected.y),)
            if any(not isclose(a, b, rel_tol=0., abs_tol=1e-8) for a, b in pairs):
                raise ValueError("grounded chassis and suspension rates disagree")
        yaw = self.grounded_yaw_policy
        if yaw is not None:
            if not isclose(yaw.mass_kg, self.independent_suspension_policy.mass_kg, abs_tol=1e-8):
                raise ValueError("grounded yaw and suspension mass must match")
            if not isclose(yaw.front_axle_distance_m + yaw.rear_axle_distance_m,
                           parameters.wheelbase, abs_tol=1e-8):
                raise ValueError("grounded yaw axle geometry must match vehicle wheelbase")

    def validate_track(self, track: TrackWorld) -> None:
        if self.mode == "wp7_lifecycle_v1" and (
            not track.surfaces or any(p.normal != Vec3(0., 1., 0.) for p in track.surfaces)
            or len({p.origin.y for p in track.surfaces}) != 1
        ):
            raise ValueError("lifecycle v1 requires coplanar horizontal road patches")

    def validate_transition(
        self, result: WorldStepResult, *, previous: WorldState,
        parameters: VehicleParameters, dt: float, start_time_s: float, end_time_s: float,
    ) -> PhysicsTransitionAudit | None:
        if self.mode == "legacy_v1":
            return None
        if self.mode != "legacy_v1" and any(isinstance(e, ObstacleHit) for e in result.events):
            # Legacy rollback zeroes the chassis velocity while retaining the
            # previous suspension rates; it is not a registered impulse/recovery.
            raise PhysicsTransitionError("WP7 collision recovery is unregistered; step rejected")
        events = tuple(e for e in result.events if isinstance(e, FlightTransition))
        if self.mode == "wp7_lifecycle_v1":
            segments = result.center_segments
            if (not isinstance(segments,tuple) or not segments
                    or not all(isinstance(s,CenterSegment) for s in segments)
                    or abs(segments[0].start_time_s) > 1e-9
                    or abs(segments[-1].end_time_s-dt) > 1e-8
                    or (segments[0].start-previous.vehicle.position).magnitude > 1e-7
                    or (segments[-1].end-result.state.vehicle.position).magnitude > 1e-7
                    or any(abs(a.end_time_s-b.start_time_s) > 1e-9
                           or (a.end-b.start).magnitude > 1e-7 for a,b in zip(segments,segments[1:]))):
                raise PhysicsTransitionError("lifecycle center trajectory is incomplete or discontinuous")
            yaw_rate = 0.
            if previous.vehicle.grounded and result.state.vehicle.grounded and not events:
                first = _horizontal_yaw(previous.vehicle.orientation).rotate(Vec3(0.,0.,1.))
                last = _horizontal_yaw(result.state.vehicle.orientation).rotate(Vec3(0.,0.,1.))
                yaw_rate = atan2(first.cross(last).y, first.dot(last))/dt
            self.validate_state(result.state, parameters, _prescribed_yaw_rate=yaw_rate)
            if any(e.time_s > dt+1e-8 for e in events) or any(
                b.time_s < a.time_s for a, b in zip(events, events[1:])
            ):
                raise PhysicsTransitionError("unordered/out-of-tick flight events")
            flight = result.flight_telemetry
            phase = "airborne" if previous.flight.phase == "airborne" else "grounded"
            for event in events:
                if event.kind == "takeoff":
                    if phase != "grounded":
                        raise PhysicsTransitionError("takeoff requires grounded predecessor")
                    phase = "airborne"
                else:
                    if phase != "airborne":
                        raise PhysicsTransitionError("impact requires airborne predecessor")
                    if event.kind == "landing":
                        phase = "grounded"
            if (phase == "grounded") != result.state.vehicle.grounded:
                raise PhysicsTransitionError("phase change lacks ordered transition evidence")
            if any(e.kind == "landing" for e in events):
                if flight is None or flight.manifold_audit is None or len(flight.manifold_audit.contacts) != 4:
                    raise PhysicsTransitionError("grounded recovery requires four simultaneous audited wheel contacts")
            if flight is not None:
                for audit in flight.manifold_history:
                    if audit.post_kinetic_energy_j > audit.pre_kinetic_energy_j + self.ballistic_flight_policy.manifold.energy_tolerance_j:
                        raise PhysicsTransitionError("impact energy gain violates lifecycle contract")
        solvers = []
        if result.suspension_telemetry is not None:
            solvers.append("independent_suspension_v1")
        if result.partial_support_telemetry is not None:
            solvers.append("partial_wheel_support_v1")
        if result.flight_telemetry is not None:
            solvers.append("torque_free_rk4_v1" if self.ballistic_flight_policy.torque_free else "ballistic_constant_omega_v1")
            if result.flight_telemetry.manifold_audit is not None:
                solvers.append("contact_manifold_pgs_v1")
                if self.mode == "wp7_lifecycle_v1":
                    solvers.append("wheel_toi_monotone_v1")
        if result.grounded_yaw_telemetry is not None:
            solvers.append("grounded_dynamic_yaw_v1")
        return PhysicsTransitionAudit(
            "wp7_transition_audit_v1", self.mode,
            previous.flight.phase if previous.flight is not None else "grounded",
            result.state.flight.phase if result.state.flight is not None else "grounded",
            start_time_s, end_time_s, tuple(solvers), events,
        )

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
