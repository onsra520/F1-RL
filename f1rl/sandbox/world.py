from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

from .body_collision import VehicleBodyGeometry, sweep_body_against_obstacle
from .center_trajectory import CenterSegment, linear_center_segments
from .geometry.vector import Vec3
from .grounded_yaw_dynamics import (
    GroundedYawPolicy,GroundedYawAudit,advance_grounded_yaw,
)
from .dynamics.integrator import integrate, integrate_ground_distance
from .dynamics.modifiers import MotionModifiers
from .dynamics.vehicle_dynamics import compute_motion
from .interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from .road_footprint import swept_body_road_union
from .partial_wheel_release import (
    AutomaticTakeoffPolicy, GroundSupportStep, integrate_partial_support, partial_ground_contact,
)
from .ballistic_flight import (
    BallisticFlightPolicy, FlightPhysicsError, FlightState,
    FlightStep, FlightTransition, authorize_takeoff, advance_ballistic_flight,
)
from .independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy,
    SuspensionDynamicsResult,
    SuspensionDynamicsState,
    integrate_independent_suspension,
)
from .ramp_support import PlanarRampPolicy, certify_single_planar_ramp
from .multi_surface_support import MultiSurfaceSuspensionPolicy, certify_multisurface_support
from .interactions.resolver import resolve_entry_events, resolve_motion_modifiers, resolve_obstacle_block
from .interactions.state import InteractionState
from .rules.scoring import ScoreState, apply_score_events
from .track.model import TrackWorld
from .track.query import TrackQuery
from .track.state import ScorePickupCollected, TrackState, collect_score_pickups
from .vehicle import VehicleControl, VehicleParameters, VehicleState

WorldEvent: TypeAlias = (
    ObstacleHit | CheckpointEntered | FinishLineEntered |
    ScorePickupCollected | FlightTransition
)


@dataclass(frozen=True, slots=True)
class WorldState:
    vehicle: VehicleState
    track: TrackState
    interactions: InteractionState
    score: ScoreState
    suspension: SuspensionDynamicsState | None = None
    flight: FlightState | None = None


@dataclass(frozen=True, slots=True)
class WorldStepResult:
    state: WorldState
    query: TrackQuery
    modifiers: MotionModifiers
    events: tuple[WorldEvent, ...]
    suspension_telemetry: SuspensionDynamicsResult | None = None
    flight_telemetry: FlightStep | None = None
    partial_support_telemetry: GroundSupportStep | None = None
    grounded_yaw_telemetry: GroundedYawAudit | None = None
    center_segments: tuple[CenterSegment, ...] = ()


def step_world(
    track: TrackWorld,
    state: WorldState,
    control: VehicleControl,
    parameters: VehicleParameters,
    dt: float,
    max_surface_distance: float,
    *,
    body_geometry: VehicleBodyGeometry | None = None,
    ramp_support_policy: PlanarRampPolicy | None = None,
    multi_surface_policy: MultiSurfaceSuspensionPolicy | None = None,
    independent_suspension_policy: IndependentSuspensionDynamicsPolicy | None = None,
    ballistic_flight_policy: BallisticFlightPolicy | None = None,
    automatic_takeoff_policy: AutomaticTakeoffPolicy | None = None,
    grounded_yaw_policy: GroundedYawPolicy | None = None,
    ground_motion_version: str = 'legacy_euler_v1',
) -> WorldStepResult:
    if ground_motion_version not in ('legacy_euler_v1','arc_distance_v1'):
        raise ValueError('unregistered ground motion version')
    if ground_motion_version=='arc_distance_v1' and grounded_yaw_policy is not None:
        raise ValueError('distance bicycle integration cannot replace dynamic tire yaw')
    if ramp_support_policy is not None and (body_geometry is None or not isinstance(ramp_support_policy, PlanarRampPolicy)):
        raise ValueError('ramp_support_policy requires full-body geometry and PlanarRampPolicy')
    if multi_surface_policy is not None:
        if body_geometry is None or not isinstance(multi_surface_policy, MultiSurfaceSuspensionPolicy):
            raise ValueError("multi_surface_policy requires full-body geometry and MultiSurfaceSuspensionPolicy")
        if ramp_support_policy is not None:
            raise ValueError("single-patch and multi-surface policies are mutually exclusive")
    if independent_suspension_policy is not None:
        if (not isinstance(independent_suspension_policy, IndependentSuspensionDynamicsPolicy)
                or not isinstance(body_geometry, VehicleBodyGeometry)
                or not isinstance(state.suspension, SuspensionDynamicsState)):
            raise ValueError("independent_suspension_policy requires explicit geometry and persistent suspension state")
        if multi_surface_policy is not None or ramp_support_policy is not None:
            raise ValueError("independent suspension and other road policies are mutually exclusive")
    if grounded_yaw_policy is not None:
        if (not isinstance(grounded_yaw_policy,GroundedYawPolicy)
                or independent_suspension_policy is None
                or automatic_takeoff_policy is not None):
            raise ValueError("grounded_yaw_policy requires independent suspension and no automatic takeoff")
        if abs(grounded_yaw_policy.mass_kg-independent_suspension_policy.mass_kg)>1e-8:
            raise ValueError("grounded yaw and suspension mass must match")
        if (abs(grounded_yaw_policy.front_axle_distance_m
                +grounded_yaw_policy.rear_axle_distance_m-parameters.wheelbase)>1e-8):
            raise ValueError("grounded yaw axle geometry must match vehicle wheelbase")
        if (ballistic_flight_policy is not None
                and ballistic_flight_policy.manifold is not None
                and abs(grounded_yaw_policy.yaw_inertia_kg_m2
                        -ballistic_flight_policy.manifold.yaw_inertia_kg_m2)>1e-8):
            raise ValueError("grounded yaw inertia must match registered flight manifold")
    if ballistic_flight_policy is not None:
        if (not isinstance(ballistic_flight_policy, BallisticFlightPolicy)
                or not isinstance(body_geometry, VehicleBodyGeometry)
                or not isinstance(state.flight, FlightState)
                or independent_suspension_policy is None
                or ballistic_flight_policy.contact != independent_suspension_policy.contact):
            raise ValueError(
                "ballistic_flight_policy requires explicit body, matching independent "
                "suspension policy, and persistent flight phase"
            )
        if state.flight.phase == "grounded" and not state.vehicle.grounded:
            raise FlightPhysicsError("grounded flight state disagrees with vehicle")
        if state.flight.phase == "airborne" and state.vehicle.grounded:
            raise FlightPhysicsError("airborne flight state disagrees with vehicle")
    if automatic_takeoff_policy is not None:
        if (not isinstance(automatic_takeoff_policy, AutomaticTakeoffPolicy)
                or ballistic_flight_policy is None
                or independent_suspension_policy is None
                or state.flight.phase not in ("grounded", "takeoff", "airborne")):
            raise ValueError(
                "automatic_takeoff_policy requires grounded flight state, matching "
                "ballistic policy and independent suspension"
            )
    current_query = track.query_point(state.vehicle.position, max_surface_distance, state.track)

    flight_telemetry: FlightStep | None = None
    partial_support_telemetry: GroundSupportStep | None = None
    auto_released = False
    flight_events: tuple[FlightTransition, ...] = ()
    center_segments: tuple[CenterSegment, ...] = ()
    suspension_telemetry: SuspensionDynamicsResult | None = None
    suspension_override: SuspensionDynamicsState | None = None
    grounded_yaw_telemetry: GroundedYawAudit | None = None
    flight_active = (ballistic_flight_policy is not None
                     and state.flight.phase in ("takeoff", "airborne"))
    if flight_active:
        modifiers = MotionModifiers()  # air cannot receive ground boost or drag
        start_air = state.vehicle
        if state.flight.phase == "takeoff":
            start_air = authorize_takeoff(
                state.vehicle, ballistic_flight_policy,
                body_geometry, track.surfaces,
            )
            flight_events = (FlightTransition("takeoff", 0.),)
        flight_telemetry = advance_ballistic_flight(
            start_air, ballistic_flight_policy,
            body_geometry, track.surfaces,
            tuple(item.volume for item in track.obstacles),
            dt=dt, gravity_mps2=parameters.gravity,
            landing_dynamics=independent_suspension_policy,
        )
        candidate = flight_telemetry.vehicle
        center_segments = flight_telemetry.center_segments
        flight_events += flight_telemetry.transitions
        swept_poses = (
            (state.vehicle, *flight_telemetry.poses[1:])
            if state.flight.phase == "takeoff"
            else flight_telemetry.poses
        )
        if candidate.grounded:
            # Exact event-time touchdown was independently certified; consume
            # the rest of dt using grounded suspension, never discard elapsed time.
            suspension_override = (
                flight_telemetry.landing_suspension
                or SuspensionDynamicsState(candidate.position.y)
            )
            remainder = flight_telemetry.remaining_dt_s
            if remainder > 0.:
                end_target = VehicleState(
                    candidate.position + Vec3(
                        candidate.velocity.x*remainder, 0.,
                        candidate.velocity.z*remainder),
                    candidate.velocity, candidate.orientation,
                    candidate.angular_velocity, True,
                )
                suspension_telemetry = integrate_independent_suspension(
                    candidate, end_target, suspension_override,
                    independent_suspension_policy, body_geometry, track.surfaces,
                    dt=remainder, gravity_mps2=parameters.gravity,
                )
                candidate = suspension_telemetry.vehicle
                center_segments += linear_center_segments(suspension_telemetry.poses,dt-remainder,dt)
                swept_poses = (*swept_poses, *suspension_telemetry.poses[1:])
    else:
        modifiers = resolve_motion_modifiers(current_query)
        motion_contact = current_query.contact
        if ground_motion_version=='arc_distance_v1' and motion_contact is None:
            if automatic_takeoff_policy is None or independent_suspension_policy is None:
                raise ValueError('distance ground motion requires verified ground support')
            motion_contact = partial_ground_contact(state.vehicle,independent_suspension_policy.contact,
                                                   track.surfaces,automatic_takeoff_policy)
        motion = compute_motion(state.vehicle, control, parameters, motion_contact, modifiers)
        path_at_time = None
        if ground_motion_version=='arc_distance_v1':
            def path_at_time(time_s):
                return integrate_ground_distance(state.vehicle,motion,control,parameters,motion_contact,time_s)
            candidate = path_at_time(dt)
        else:
            candidate = integrate(state.vehicle, motion, parameters, current_query.contact, dt)
        if grounded_yaw_policy is not None:
            candidate,grounded_yaw_telemetry=advance_grounded_yaw(
                state.vehicle,candidate,
                control.steering*parameters.max_steering_angle,
                grounded_yaw_policy,dt=dt,gravity_mps2=parameters.gravity,
            )
        swept_poses = (state.vehicle, candidate)
        if automatic_takeoff_policy is not None:
            partial_support_telemetry = integrate_partial_support(
                state.vehicle, candidate, state.suspension,
                independent_suspension_policy, body_geometry, track.surfaces,
                automatic_takeoff_policy, dt=dt, gravity_mps2=parameters.gravity,
                path_at_time=path_at_time,
            )
            candidate = partial_support_telemetry.vehicle
            partial_times = tuple(f.time_s for f in partial_support_telemetry.frames) + (
                partial_support_telemetry.takeoff_at_s if partial_support_telemetry.takeoff_at_s is not None else dt,)
            center_segments = tuple(CenterSegment(a,b,p.position,q.position)
                for a,b,p,q in zip(partial_times,partial_times[1:],partial_support_telemetry.poses,
                                   partial_support_telemetry.poses[1:]))
            swept_poses = partial_support_telemetry.poses
            if partial_support_telemetry.takeoff_at_s is not None:
                auto_released = True
                departure = partial_support_telemetry.takeoff_at_s
                flight_events = (FlightTransition("takeoff", departure),)
                remaining = partial_support_telemetry.remaining_dt_s
                if remaining > 0.:
                    flight_telemetry = advance_ballistic_flight(
                        candidate, ballistic_flight_policy,
                        body_geometry, track.surfaces,
                        tuple(item.volume for item in track.obstacles),
                        dt=remaining, gravity_mps2=parameters.gravity,
                        landing_dynamics=independent_suspension_policy,
                    )
                    if flight_telemetry.vehicle.grounded:
                        raise FlightPhysicsError(
                            "same-tick automatic recontact lacks a registered impulse solver"
                        )
                    candidate = flight_telemetry.vehicle
                    center_segments += tuple(replace(s,start_time_s=s.start_time_s+departure,
                        end_time_s=s.end_time_s+departure) for s in flight_telemetry.center_segments)
                    flight_events += tuple(
                        FlightTransition(event.kind, departure + event.time_s, event.patch_id)
                        for event in flight_telemetry.transitions
                    )
                    swept_poses = (*swept_poses, *flight_telemetry.poses[1:])
        elif independent_suspension_policy is not None:
            # Legacy XZ/yaw stay prescribed; default 4-wheel solver unchanged.
            suspension_telemetry = integrate_independent_suspension(
                state.vehicle, candidate, state.suspension,
                independent_suspension_policy, body_geometry, track.surfaces,
                dt=dt, gravity_mps2=parameters.gravity,
                path_at_time=path_at_time,
            )
            candidate = suspension_telemetry.vehicle
            center_segments = linear_center_segments(suspension_telemetry.poses,0.,dt)
            swept_poses = suspension_telemetry.poses
    candidate_query = track.query_point(candidate.position, max_surface_distance, state.track)
    if body_geometry is not None:
        if not isinstance(body_geometry, VehicleBodyGeometry):
            raise ValueError("body_geometry must be VehicleBodyGeometry")
        # Prospective full-body engine: continuous translation + rotation.
        # Unresolved conservative intervals must block, not tunnel.
        body_hits = []
        for obstacle in track.obstacles:
            for segment_start, segment_end in zip(swept_poses, swept_poses[1:]):
                verdict = sweep_body_against_obstacle(
                    segment_start, segment_end, body_geometry, obstacle.volume
                )
                if verdict.uncertain:
                    # Unknown is not a verified collision; invalidate rather
                    # than fabricate ObstacleHit or a successful evaluation.
                    raise RuntimeError(
                        f"body sweep unresolved for obstacle {obstacle.id!r}; "
                        "candidate rollout is invalid, not a verified collision"
                    )
                if verdict.contact:
                    body_hits.append(obstacle)
                    break
        if (flight_active or auto_released) and body_hits:
            raise FlightPhysicsError(
                "flight/substep body hit obstacle: no scored impact response registered"
            )
        candidate_query = replace(candidate_query, obstacles=tuple(body_hits))

    accepted, obstacle_events = resolve_obstacle_block(state.vehicle, candidate, candidate_query)
    if body_geometry is not None:
        if flight_active or auto_released:
            # Airborne path is permitted only by explicit flight contract;
            # any touchdown and grounded remainder have separate certificates.
            road_verdict = None
        elif independent_suspension_policy is not None:
            # The integrator already conservatively certified every
            # individual substep; never switch to the legacy centre policy.
            road_verdict = None
        elif multi_surface_policy is not None:
            road_verdict = certify_multisurface_support(
                state.vehicle, accepted, body_geometry,
                multi_surface_policy, track.surfaces,
            )
        elif ramp_support_policy is None:
            road_verdict = swept_body_road_union(
                state.vehicle, accepted, body_geometry, track.surfaces,
                ride_height_m=parameters.ride_height,
            )
        elif len(track.surfaces) == 1:
            road_verdict = certify_single_planar_ramp(
                state.vehicle, accepted, body_geometry,
                ramp_support_policy.wheels, track.surfaces[0],
                plane_tolerance_m=ramp_support_policy.plane_tolerance_m,
                max_subdivisions=ramp_support_policy.max_subdivisions,
            )
        else:
            # Crossing surface seams, slope transitions and multiple decks
            # have no approved tire-support rule in this research version.
            raise RuntimeError(
                "prospective full-body ramp support unsupported: "
                "requires exactly one planar patch; rollout invalid"
            )
        if road_verdict is not None and not road_verdict.valid:
            # This is not an ObstacleHit and must not be counted as a
            # successful/valid simulation transition. A later registered
            # road-violation episode rule can replace this hard gate.
            raise RuntimeError(
                f"prospective full-body road validity {road_verdict.status}: "
                f"{road_verdict.reason}; rollout invalid"
            )
    accepted_query = (
        track.query_point(accepted.position, max_surface_distance, state.track)
        if obstacle_events
        else candidate_query
    )
    if flight_active or auto_released:
        # Legacy point-volume triggers are not a valid airborne/landing race
        # rule. Do not generate checkpoints/finish/pickups for these transitions.
        interaction_state, checkpoint_events, finish_events = state.interactions, (), ()
        track_state, pickup_events = state.track, ()
    else:
        interaction_state, checkpoint_events, finish_events = resolve_entry_events(
            accepted_query,
            state.interactions,
        )
        track_state, pickup_events = collect_score_pickups(
            accepted_query.score_pickups,
            state.track,
        )
    score_state = apply_score_events(state.score, pickup_events)

    final_query = track.query_point(
        accepted.position,
        max_surface_distance,
        track_state,
    )
    next_state = WorldState(
        vehicle=accepted,
        track=track_state,
        interactions=interaction_state,
        score=score_state,
        suspension=(
            state.suspension if obstacle_events else
            suspension_telemetry.dynamics if suspension_telemetry is not None else
            suspension_override if suspension_override is not None else
            partial_support_telemetry.dynamics if partial_support_telemetry is not None else
            state.suspension
        ),
        flight=(
            flight_telemetry.flight if flight_telemetry is not None
            else FlightState("airborne") if auto_released
            else state.flight
        ),
    )
    events: tuple[WorldEvent, ...] = (
        *flight_events,
        *obstacle_events,
        *checkpoint_events,
        *finish_events,
        *pickup_events,
    )
    return WorldStepResult(
        next_state, final_query, modifiers, events,
        suspension_telemetry if not obstacle_events else None,
        flight_telemetry,
        partial_support_telemetry,
        grounded_yaw_telemetry,
        center_segments if not obstacle_events else (),
    )
