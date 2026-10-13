from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

from .body_collision import VehicleBodyGeometry, sweep_body_against_obstacle
from .geometry.vector import Vec3
from .dynamics.integrator import integrate
from .dynamics.modifiers import MotionModifiers
from .dynamics.vehicle_dynamics import compute_motion
from .interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from .road_footprint import swept_body_road_union
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
) -> WorldStepResult:
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
    current_query = track.query_point(state.vehicle.position, max_surface_distance, state.track)

    flight_telemetry: FlightStep | None = None
    flight_events: tuple[FlightTransition, ...] = ()
    suspension_telemetry: SuspensionDynamicsResult | None = None
    suspension_override: SuspensionDynamicsState | None = None
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
        )
        candidate = flight_telemetry.vehicle
        flight_events += flight_telemetry.transitions
        swept_poses = (
            (state.vehicle, *flight_telemetry.poses[1:])
            if state.flight.phase == "takeoff"
            else flight_telemetry.poses
        )
        if candidate.grounded:
            # Exact event-time touchdown was independently certified; consume
            # the rest of dt using grounded suspension, never discard elapsed time.
            suspension_override = SuspensionDynamicsState(candidate.position.y)
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
                swept_poses = (*swept_poses, *suspension_telemetry.poses[1:])
    else:
        modifiers = resolve_motion_modifiers(current_query)
        motion = compute_motion(state.vehicle, control, parameters, current_query.contact, modifiers)
        candidate = integrate(state.vehicle, motion, parameters, current_query.contact, dt)
        swept_poses = (state.vehicle, candidate)
        if independent_suspension_policy is not None:
            # Prescribed XZ/yaw are still from the legacy bicycle model; the
            # grounded suspension solver owns height, pitch and roll.
            suspension_telemetry = integrate_independent_suspension(
                state.vehicle, candidate, state.suspension,
                independent_suspension_policy, body_geometry, track.surfaces,
                dt=dt, gravity_mps2=parameters.gravity,
            )
            candidate = suspension_telemetry.vehicle
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
        if flight_active and body_hits:
            raise FlightPhysicsError(
                "flight/substep body hit obstacle: no scored impact response registered"
            )
        candidate_query = replace(candidate_query, obstacles=tuple(body_hits))

    accepted, obstacle_events = resolve_obstacle_block(state.vehicle, candidate, candidate_query)
    if body_geometry is not None:
        if flight_active:
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
    if flight_active:
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
            state.suspension
        ),
        flight=(
            flight_telemetry.flight if flight_telemetry is not None
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
    )
