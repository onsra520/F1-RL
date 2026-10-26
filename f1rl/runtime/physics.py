"""Versioned research physics registration; never scientific training admission.

Longitudinal motion is versioned: legacy Euler, force-frozen distance bicycle,
or coupled world-Z/heave/pitch contact forces with numerical work balance.
Contacts are suspension reference points, not tire hubs/patches. No physical
constants are inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isclose, isfinite, atan2, sin, tan

from ..sandbox.ballistic_flight import BallisticFlightPolicy, FlightState, FlightTransition
from ..sandbox.center_trajectory import CenterSegment
from ..sandbox.coupled_longitudinal_dynamics import LongitudinalContactPolicy, _longitudinal_frame
from ..sandbox.dynamics.integrator import _distance_ground_frame
from ..sandbox.body_collision import VehicleBodyGeometry
from ..sandbox.grounded_yaw_dynamics import GroundedYawPolicy
from ..sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState, _horizontal_yaw,
    angular_velocity_for_suspension,
    suspension_orientation,
)
from ..sandbox.geometry.vector import Vec3
from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.partial_wheel_release import AutomaticTakeoffPolicy
from ..sandbox.interactions.events import ObstacleHit
from ..sandbox.vehicle import VehicleParameters
from ..sandbox.spatial_contact_dynamics import SpatialContactPolicy, _suspension_state
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
    ground_motion_version: str = 'legacy_euler_v1'
    spawn_reference: str = 'legacy_vehicle_position_v1'
    longitudinal_contact_policy: LongitudinalContactPolicy | None = None
    spatial_contact_policy: SpatialContactPolicy | None = None

    def __post_init__(self) -> None:
        if (self.spawn_reference not in ('legacy_vehicle_position_v1','authored_horizontal_road_v1')
                or self.spawn_reference!='legacy_vehicle_position_v1' and self.mode=='legacy_v1'):
            raise ValueError('unregistered research spawn reference')
        if (self.ground_motion_version not in ('legacy_euler_v1','arc_distance_v1','coupled_longitudinal_v1','spatial_contact_v1','spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4')
                or self.ground_motion_version=='arc_distance_v1' and self.mode not in ('wp7_suspension_v1','wp7_lifecycle_v1')):
            raise ValueError('unregistered research ground motion version/mode')
        coupled = self.ground_motion_version == 'coupled_longitudinal_v1'
        compliant = self.ground_motion_version in ('spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4')
        spatial = self.ground_motion_version in ('spatial_contact_v1','spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4')
        if (spatial != (self.spatial_contact_policy is not None)
                or spatial and (self.mode!=('wp7_lifecycle_v1' if compliant else 'wp7_suspension_v1')
                    or not isinstance(self.spatial_contact_policy,SpatialContactPolicy)
                    or not isinstance(self.independent_suspension_policy,IndependentSuspensionDynamicsPolicy)
                    or compliant and self.independent_suspension_policy.force_model!='road_relative_unilateral_v3'
                    or self.independent_suspension_policy.force_model not in ('road_relative_vertical_v2','road_relative_unilateral_v3')
                    or self.independent_suspension_policy.angular_rate_mapping!='yaw_rx_rz_v1')):
            raise ValueError('spatial ground version requires its exclusive supported contact policy')
        if (coupled != (self.longitudinal_contact_policy is not None)
                or coupled and (self.mode != 'wp7_suspension_v1'
                    or not isinstance(self.longitudinal_contact_policy,LongitudinalContactPolicy)
                    or not isinstance(self.independent_suspension_policy,IndependentSuspensionDynamicsPolicy)
                    or self.independent_suspension_policy.force_model != 'road_relative_vertical_v2')):
            raise ValueError('coupled ground version requires its exclusive grounded contact policy')
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
        if suspension.force_model=='road_relative_unilateral_v3' and not spatial:
            raise ValueError('unilateral force model requires spatial contact integration')
        if suspension.force_model != 'legacy_vertical_v1' and self.mode != 'wp7_suspension_v1' and not compliant:
            raise ValueError('road-relative force model requires grounded suspension-only mode')
        if self.ground_motion_version=='arc_distance_v1' and suspension.angular_rate_mapping!='yaw_rx_rz_v1':
            raise ValueError('distance ground version requires exact suspension angular-rate mapping')
        if (body.length_m, body.width_m, body.height_m) != (4., 1.8, 1.2):
            raise ValueError("WP7 owner body must remain 4.0 x 1.8 x 1.2 m")
        flight = self.ballistic_flight_policy
        yaw = self.grounded_yaw_policy
        if (flight is not None) != (self.mode in ("wp7_ballistic_v1", "wp7_lifecycle_v1") and not compliant):
            raise ValueError("ballistic mode requires its explicit flight policy exclusively")
        if (yaw is not None) != (self.mode == "wp7_grounded_yaw_v1"):
            raise ValueError("grounded yaw mode requires its explicit yaw policy exclusively")
        if self.automatic_takeoff_policy is not None and flight is None:
            raise ValueError("automatic takeoff requires ballistic mode")
        if self.mode == "wp7_lifecycle_v1" and not compliant:
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
        if self.ground_motion_version=='arc_distance_v1' and initial.vehicle.grounded:
            _distance_ground_frame(initial.vehicle,parameters)
        if self.ground_motion_version=='coupled_longitudinal_v1':
            _longitudinal_frame(initial.vehicle,initial.suspension)
        if self.ground_motion_version in ('spatial_contact_v1','spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4'):
            derived = _suspension_state(initial.vehicle)
            if any(abs(getattr(derived,key)-getattr(initial.suspension,key))>1e-8
                   for key in derived.__dataclass_fields__):
                raise ValueError('spatial chassis and persistent suspension state disagree')
            _prescribed_yaw_rate = initial.vehicle.angular_velocity.y + sin(derived.pitch_x_rad)*derived.roll_rate_radps
        if initial.vehicle.grounded and not isclose(
            initial.vehicle.position.y, initial.suspension.height_m, rel_tol=0., abs_tol=1e-8,
        ):
            raise ValueError("initial suspension height disagrees with vehicle")
        if self.ground_motion_version in ('spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4'):
            if (not isinstance(initial.flight,FlightState) or initial.flight.phase=='takeoff'
                    or initial.vehicle.grounded!=(initial.flight.phase=='grounded')):
                raise ValueError('compliant lifecycle requires consistent persistent phase')
        elif self.ballistic_flight_policy is None:
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
            if (self.mode == "wp7_lifecycle_v1"
                    or self.independent_suspension_policy.force_model != 'legacy_vertical_v1'):
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
            if (self.mode == "wp7_lifecycle_v1" or self.ground_motion_version=='arc_distance_v1'
                    or self.independent_suspension_policy.force_model != 'legacy_vertical_v1'):
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

    def validate_navigation_spawn(self, authored: Vec3, position: Vec3,
                                  parameters: VehicleParameters, track: TrackWorld) -> None:
        if self.spawn_reference=='legacy_vehicle_position_v1':
            if authored!=position:
                raise ValueError('navigation start must match initial vehicle position')
            return
        contact = track.surface_at(authored,0.)
        if contact is None or contact.normal!=Vec3(0.,1.,0.):
            raise ValueError('authored spawn requires horizontal road contact')
        expected = authored+Vec3(0.,parameters.ride_height,0.)
        if (position-expected).magnitude>1e-9:
            raise ValueError('research chassis spawn must match authored road plus ride height')

    def validate_track(self, track: TrackWorld) -> None:
        if self.ground_motion_version=='coupled_longitudinal_v1' and (
            not track.surfaces or any(abs(p.normal.x)>1e-10 for p in track.surfaces)
        ):
            raise ValueError('coupled longitudinal contact requires world-Z road slopes')
        if self.mode == "wp7_lifecycle_v1" and self.ground_motion_version not in ('spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4') and (
            not track.surfaces or any(p.normal != Vec3(0., 1., 0.) for p in track.surfaces)
            or len({p.origin.y for p in track.surfaces}) != 1
        ):
            raise ValueError("lifecycle v1 requires coplanar horizontal road patches")

    def validate_transition(
        self, result: WorldStepResult, *, previous: WorldState,
        parameters: VehicleParameters, dt: float, start_time_s: float, end_time_s: float,
    ) -> PhysicsTransitionAudit | None:
        if result.roll_envelope_stop is not None:
            from .roll_envelope_validation import verify_roll_envelope_stop
            verify_roll_envelope_stop(result,previous,self,dt=dt)
            self.validate_state(result.state,parameters)
            return PhysicsTransitionAudit(
                'wp7_transition_audit_v1',self.mode,
                previous.flight.phase if previous.flight is not None else 'grounded',
                'grounded',start_time_s,end_time_s,
                ('spatial_road_edge_partial_lifecycle_rk4_v4',
                 'certified_roll_envelope_absorbing_v1_development'),())
        if result.crash is not None:
            if result.road_crash_source is not None:
                # Independently replay finite-road first contact and accepted
                # rigid impulse. Distinct, development-only source/version.
                from .road_crash_validation import verify_development_road_crash
                verify_development_road_crash(result,previous,self,parameters=parameters,dt=dt)
                self.validate_state(result.state,parameters)
                return PhysicsTransitionAudit(
                    "wp7_transition_audit_v1",self.mode,
                    previous.flight.phase if previous.flight is not None else "grounded",
                    "grounded",start_time_s,end_time_s,
                    ("spatial_road_edge_partial_lifecycle_rk4_v4",
                     "grounded_wheel_hard_stop_absorbing_v4_development"
                     if result.crash.contact.kind=="wheel_hard_stop" else
                     "grounded_wheel_edge_hard_stop_absorbing_v4_development"
                     if result.crash.contact.kind=="wheel_edge_hard_stop" else
                     "grounded_road_obb_impulse_absorbing_v4_development"),())
            # Fully revalidate the pre-impact RK4 tick against its unchanged
            # momenta and work audit, then separately audit the accepted
            # instantaneous impulse and absorbing center/event prefix.
            if (self.ground_motion_version!='spatial_road_edge_partial_lifecycle_v4'
                or result.precrash_result is None
                or result.precrash_result.crash is not None
                or result.precrash_result.precrash_result is not None
                or result.events or result.flight_telemetry is not None):
                raise PhysicsTransitionError('unregistered or self-referential crash transition')
            original=self.validate_transition(result.precrash_result,
                previous=previous,parameters=parameters,dt=dt,
                start_time_s=start_time_s,end_time_s=end_time_s)
            from ..sandbox.accepted_contact_transition import (
                _apply_mass_impulse, absorb_crash_center_prefix)
            from ..sandbox.rigid_body_impact_response import RigidBodyImpactMass
            crash=result.crash
            c=crash.contact
            if (c.kind,crash.version) not in (
                    ("obstacle_face","wp3_grounded_applied_crash_v4_dev"),
                    ("obstacle_edge","wp3_grounded_applied_edge_crash_v4_dev")):
                raise PhysicsTransitionError("unregistered obstacle contact type/version")
            solver_name=("grounded_face_impulse_absorbing_v4_development"
                if c.kind=="obstacle_face" else
                "grounded_edge_impulse_absorbing_v4_development")
            base=result.precrash_result
            mass=RigidBodyImpactMass(self.independent_suspension_policy.mass_kg,
                Vec3(self.independent_suspension_policy.pitch_inertia_kg_m2,
                     self.spatial_contact_policy.yaw_inertia_kg_m2,
                     self.independent_suspension_policy.roll_inertia_kg_m2))
            try:
                audited=_apply_mass_impulse(c.preimpact,mass,
                    c.impulse_n_s,c.angular_impulse_nms,
                    c.kinetic_before_j,c.kinetic_after_j,c.energy_lost_j,
                    first_clear=c.first_clear_s,witness=c.witnessed_contact_s,
                    kind=c.kind)
                expected=absorb_crash_center_prefix(base.center_segments,
                    crash.tick_contact_time_s,fixed_dt_s=dt,accepted=c)
            except ValueError as error:
                raise PhysicsTransitionError('crash state/impulse/prefix not audited') from error
            index=crash.segment_index
            if (audited!=c or not 0<=index<len(base.center_segments)
                or not c.preimpact.grounded or not c.postimpact.grounded
                or result.state.vehicle!=c.postimpact
                or result.state.interactions!=previous.interactions
                or result.state.score!=previous.score
                or result.state.track!=previous.track
                or result.center_segments!=expected or crash.prefix!=expected
                or not base.center_segments[index].start_time_s<=crash.last_proven_clear_s
                         < crash.tick_contact_time_s<=base.center_segments[index].end_time_s
                or (base.center_segments[index].position_at(crash.tick_contact_time_s)
                    -c.preimpact.position).magnitude>1e-6
                or original is None):
                raise PhysicsTransitionError('unaccepted crash contact momentum or event-time source')
            self.validate_state(result.state,parameters)
            return replace(original,solvers=(*original.solvers,
                solver_name))
        if self.mode == "legacy_v1":
            return None
        if self.mode != "legacy_v1" and any(isinstance(e, ObstacleHit) for e in result.events):
            # Legacy rollback zeroes the chassis velocity while retaining the
            # previous suspension rates; it is not a registered impulse/recovery.
            raise PhysicsTransitionError("WP7 collision recovery is unregistered; step rejected")
        events = tuple(e for e in result.events if isinstance(e, FlightTransition))
        compliant = self.ground_motion_version in ('spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4')
        if self.ground_motion_version in ('coupled_longitudinal_v1','spatial_contact_v1','spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4'):
            spatial = self.ground_motion_version in ('spatial_contact_v1','spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4')
            audit = result.spatial_ground_telemetry if spatial else result.coupled_ground_telemetry
            segments = result.center_segments
            expected_version = 'spatial_road_edge_partial_lifecycle_rk4_v4' if self.ground_motion_version=='spatial_road_edge_partial_lifecycle_v4' else 'spatial_rotating_compliant_lifecycle_rk4_v3' if self.ground_motion_version=='spatial_rotating_compliant_lifecycle_v3' else 'spatial_compliant_atlas_lifecycle_rk4_v2' if self.ground_motion_version=='spatial_compliant_atlas_lifecycle_v2' else 'spatial_compliant_lifecycle_rk4_v1' if compliant else 'spatial_contact_world_momentum_rk4_v1' if spatial else 'coupled_longitudinal_rk4_work_v1'
            if audit is not None and any(
                any(type(v) not in (int,float) or not isfinite(v) for v in
                    (leaf.initial_energy_j,leaf.final_energy_j,leaf.actuator_work_j,leaf.dissipation_j,
                     leaf.energy_residual_j,leaf.energy_tolerance_j,leaf.linear_impulse_residual_ns,
                     leaf.angular_impulse_residual_nms))
                or leaf.energy_tolerance_j<=0. or leaf.dissipation_j<0.
                or min(leaf.linear_impulse_residual_ns,leaf.angular_impulse_residual_nms)<0.
                for leaf in audit.substeps
            ):
                raise PhysicsTransitionError(f'{"spatial" if spatial else "coupled"} force/time audit requires finite work evidence')
            if (audit is None or audit.version!=expected_version
                    or not audit.substeps or len(segments)!=len(audit.substeps)
                    or (events and not compliant) or result.flight_telemetry is not None
                    or abs(segments[0].start_time_s)>1e-9 or abs(segments[-1].end_time_s-dt)>1e-8
                    or (segments[0].start-previous.vehicle.position).magnitude>1e-7
                    or (segments[-1].end-result.state.vehicle.position).magnitude>1e-7
                    or any(abs(a.end_time_s-b.start_time_s)>1e-9 or (a.end-b.start).magnitude>1e-7
                           for a,b in zip(segments,segments[1:]))
                    or any(s.start_time_s!=a.start_time_s or s.end_time_s!=a.end_time_s
                           or not abs(a.energy_residual_j)<=a.energy_tolerance_j
                           or not a.linear_impulse_residual_ns<=1e-7
                           or not a.angular_impulse_residual_nms<=1e-7
                           for s,a in zip(segments,audit.substeps))):
                raise PhysicsTransitionError('coupled force/time audit missing or inconsistent')
            if spatial:
                poses = result.suspension_telemetry.poses if result.suspension_telemetry is not None else ()
                if (len(poses)!=len(audit.substeps)+1 or poses[0]!=previous.vehicle
                        or poses[-1]!=result.state.vehicle or result.coupled_ground_telemetry is not None):
                    raise PhysicsTransitionError('spatial impulse trajectory missing or inconsistent')
                suspension = self.independent_suspension_policy
                def momentum(pose):
                    q = pose.orientation
                    omega = Quaternion(q.w,-q.x,-q.y,-q.z).rotate(pose.angular_velocity)
                    return q.rotate(Vec3(suspension.pitch_inertia_kg_m2*omega.x,
                        self.spatial_contact_policy.yaw_inertia_kg_m2*omega.y,
                        suspension.roll_inertia_kg_m2*omega.z))
                for first,last,leaf,segment in zip(poses,poses[1:],audit.substeps,segments):
                    if compliant:
                        air=not first.grounded and not last.grounded
                        h=segment.end_time_s-segment.start_time_s
                        if ((segment.initial_velocity is not None)!=air or air and (
                                segment.gravity_mps2!=parameters.gravity
                                or (segment.initial_velocity-first.velocity).magnitude>1e-9
                                or (last.velocity-first.velocity-Vec3(0.,-parameters.gravity*h,0.)).magnitude>1e-9
                                or self.ground_motion_version not in ('spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4') and max(first.angular_velocity.magnitude,last.angular_velocity.magnitude)>1e-9
                                or self.ground_motion_version in ('spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4') and (leaf.torque_impulse_nms.magnitude>1e-9
                                    or leaf.actuator_work_j!=0. or leaf.dissipation_j!=0.))):
                            raise PhysicsTransitionError('compliant ballistic law differs from accepted states')
                    linear = ((last.velocity-first.velocity)*suspension.mass_kg-leaf.force_impulse_ns).magnitude
                    angular = (momentum(last)-momentum(first)-leaf.torque_impulse_nms).magnitude
                    balance = leaf.final_energy_j-leaf.initial_energy_j-leaf.actuator_work_j+leaf.dissipation_j
                    if (not linear<=1e-7 or not angular<=1e-7
                            or not isclose(linear,leaf.linear_impulse_residual_ns,abs_tol=1e-8)
                            or not isclose(angular,leaf.angular_impulse_residual_nms,abs_tol=1e-8)
                            or not isclose(balance,leaf.energy_residual_j,abs_tol=1e-7)):
                        raise PhysicsTransitionError('spatial impulse/work evidence disagrees with accepted poses')
                totals = {
                    name:sum(getattr(leaf,name) for leaf in audit.substeps)
                    for name in ('actuator_work_j','dissipation_j','energy_residual_j')
                }
                totals.update(max_linear_impulse_residual_ns=max(a.linear_impulse_residual_ns for a in audit.substeps),
                              max_angular_impulse_residual_nms=max(a.angular_impulse_residual_nms for a in audit.substeps))
                if any(type(getattr(audit,name)) not in (int,float) or not isfinite(getattr(audit,name))
                       or not isclose(getattr(audit,name),value,rel_tol=1e-10,abs_tol=1e-8)
                       for name,value in totals.items()):
                    raise PhysicsTransitionError('spatial work summary differs from accepted leaves')
        if self.independent_suspension_policy.force_model != 'legacy_vertical_v1':
            self.validate_state(result.state, parameters)
            first = _horizontal_yaw(previous.vehicle.orientation).rotate(Vec3(0.,0.,1.))
            last = _horizontal_yaw(result.state.vehicle.orientation).rotate(Vec3(0.,0.,1.))
            if self.ground_motion_version not in ('spatial_contact_v1','spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4') and (first-last).magnitude > 1e-8:
                raise PhysicsTransitionError('road-relative reduced driving cannot change yaw')
        distance_yaw_rate = None
        if (self.ground_motion_version=='arc_distance_v1' and previous.vehicle.grounded
                and result.state.vehicle.grounded and not events):
            first = _horizontal_yaw(previous.vehicle.orientation).rotate(Vec3(0.,0.,1.))
            last = _horizontal_yaw(result.state.vehicle.orientation).rotate(Vec3(0.,0.,1.))
            yaw_change = atan2(first.cross(last).y,first.dot(last))
            delta = result.state.vehicle.position-previous.vehicle.position
            horizontal_delta = Vec3(delta.x,0.,delta.z)
            chord = horizontal_delta.magnitude
            midpoint_forward = (first+last).normalized()
            along = horizontal_delta.dot(midpoint_forward)
            if along < -1e-8 or (horizontal_delta-midpoint_forward*along).magnitude>1e-8:
                raise PhysicsTransitionError('accepted displacement does not follow a forward arc')
            if chord<=1e-12 and abs(yaw_change)>1e-8:
                raise PhysicsTransitionError('zero-distance steering cannot rotate the chassis')
            curvature = 2*sin(yaw_change/2)/chord if chord>1e-12 else 0.
            if abs(curvature)>tan(parameters.max_steering_angle)/parameters.wheelbase+1e-8:
                raise PhysicsTransitionError('accepted ground path exceeds steering curvature')
            distance_yaw_rate = curvature*result.state.vehicle.velocity.dot(last)
            self.validate_state(result.state,parameters,_prescribed_yaw_rate=distance_yaw_rate)
        if self.mode == "wp7_lifecycle_v1" and not compliant:
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
                yaw_change = atan2(first.cross(last).y, first.dot(last))
                yaw_rate = yaw_change/dt
                if distance_yaw_rate is not None:
                    yaw_rate = distance_yaw_rate
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
        if compliant:
            audit=result.spatial_ground_telemetry
            poses=result.suspension_telemetry.poses
            changes=[(leaf,'takeoff' if not last.grounded else 'landing')
                     for first,last,leaf in zip(poses,poses[1:],audit.substeps) if first.grounded!=last.grounded]
            brackets=getattr(audit,'phase_brackets',None)
            if not isinstance(brackets,tuple) or len(brackets)!=len(changes) or len(events)!=len(changes):
                raise PhysicsTransitionError('compliant phase evidence missing')
            for bracket,(leaf,kind),event in zip(brackets,changes,events):
                if (bracket.kind!=kind or event.kind!=kind or event.time_s!=bracket.upper_time_s
                        or bracket.lower_time_s!=leaf.start_time_s or bracket.upper_time_s!=leaf.end_time_s
                        or not 0.<bracket.upper_time_s-bracket.lower_time_s<=1e-8):
                    raise PhysicsTransitionError('compliant phase time bracket disagrees with trajectory')
        solvers = []
        if self.ground_motion_version=='arc_distance_v1' and (
            result.partial_support_telemetry is not None
            or result.suspension_telemetry is not None and result.flight_telemetry is None
        ):
            solvers.append('arc_distance_ground_v1')
        if result.spatial_ground_telemetry is not None:
            solvers.append(result.spatial_ground_telemetry.version)
        elif result.coupled_ground_telemetry is not None:
            solvers.append('coupled_longitudinal_rk4_work_v1')
        elif result.suspension_telemetry is not None:
            solvers.append("road_relative_vertical_suspension_v2" if
                self.independent_suspension_policy.force_model == 'road_relative_vertical_v2'
                else "independent_suspension_v1")
        if result.partial_support_telemetry is not None:
            solvers.append("partial_wheel_support_v1")
        if result.flight_telemetry is not None:
            solvers.append("torque_free_rk4_v1" if self.ballistic_flight_policy.torque_free else "ballistic_constant_omega_v1")
            if result.flight_telemetry.manifold_audit is not None:
                solvers.append("contact_manifold_pgs_v1")
                if self.mode == "wp7_lifecycle_v1" and not compliant:
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
            "longitudinal_contact_policy", "spatial_contact_policy",
        )
        options = {name: getattr(self, name) for name in names if getattr(self, name) is not None}
        if self.ground_motion_version!='legacy_euler_v1':
            options['ground_motion_version'] = self.ground_motion_version
        return options

    def document(self) -> dict[str, object]:
        configuration = asdict(self)
        coupled = self.longitudinal_contact_policy is not None
        spatial = self.spatial_contact_policy is not None
        if not spatial:
            configuration.pop('spatial_contact_policy')
        if not coupled:
            configuration.pop('longitudinal_contact_policy')
        suspension = configuration['independent_suspension_policy']
        road_relative = suspension is not None and suspension['force_model'] != 'legacy_vertical_v1'
        if suspension is not None and not road_relative:
            suspension.pop('force_model')
        if self.ground_motion_version=='legacy_euler_v1':
            configuration.pop('ground_motion_version')
        if self.spawn_reference=='legacy_vehicle_position_v1':
            configuration.pop('spawn_reference')
        return {
            "schema_version": 9 if self.ground_motion_version=='spatial_road_edge_partial_lifecycle_v4' else 8 if self.ground_motion_version=='spatial_rotating_compliant_lifecycle_v3' else 7 if self.ground_motion_version=='spatial_compliant_atlas_lifecycle_v2' else 6 if self.ground_motion_version in ('spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4') else 5 if spatial else 4 if coupled else 3 if road_relative else 1 if self.ground_motion_version=='legacy_euler_v1' and self.spawn_reference=='legacy_vehicle_position_v1' else 2,
            "validation_scope": "software_research_only",
            "longitudinal_model": 'spatial_road_edge_partial_lifecycle_rk4_v4' if self.ground_motion_version=='spatial_road_edge_partial_lifecycle_v4' else 'spatial_rotating_compliant_lifecycle_rk4_v3' if self.ground_motion_version=='spatial_rotating_compliant_lifecycle_v3' else 'spatial_compliant_atlas_lifecycle_rk4_v2' if self.ground_motion_version=='spatial_compliant_atlas_lifecycle_v2' else 'spatial_compliant_lifecycle_rk4_v1' if self.ground_motion_version in ('spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3','spatial_road_edge_partial_lifecycle_v4') else 'spatial_contact_world_momentum_rk4_v1' if spatial else 'coupled_contact_force_rk4_v1' if coupled else "legacy_prescribed_v1" if self.ground_motion_version=='legacy_euler_v1' else 'force_frozen_distance_bicycle_v1',
            **configuration,
            **({'body_clearance_version': 'bounded_patch_planes_v2'} if road_relative else {}),
        }

    @property
    def sha256(self) -> str:
        return sha256(json.dumps(
            self.document(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
