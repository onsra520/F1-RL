from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

from .body_collision import VehicleBodyGeometry, sweep_body_against_obstacle
from .dynamics.integrator import integrate
from .dynamics.modifiers import MotionModifiers
from .dynamics.vehicle_dynamics import compute_motion
from .interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from .road_footprint import swept_body_road_union
from .ramp_support import PlanarRampPolicy, certify_single_planar_ramp
from .interactions.resolver import resolve_entry_events, resolve_motion_modifiers, resolve_obstacle_block
from .interactions.state import InteractionState
from .rules.scoring import ScoreState, apply_score_events
from .track.model import TrackWorld
from .track.query import TrackQuery
from .track.state import ScorePickupCollected, TrackState, collect_score_pickups
from .vehicle import VehicleControl, VehicleParameters, VehicleState

WorldEvent: TypeAlias = ObstacleHit | CheckpointEntered | FinishLineEntered | ScorePickupCollected


@dataclass(frozen=True, slots=True)
class WorldState:
    vehicle: VehicleState
    track: TrackState
    interactions: InteractionState
    score: ScoreState


@dataclass(frozen=True, slots=True)
class WorldStepResult:
    state: WorldState
    query: TrackQuery
    modifiers: MotionModifiers
    events: tuple[WorldEvent, ...]


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
) -> WorldStepResult:
    if ramp_support_policy is not None and (body_geometry is None or not isinstance(ramp_support_policy, PlanarRampPolicy)):
        raise ValueError('ramp_support_policy requires full-body geometry and PlanarRampPolicy')
    current_query = track.query_point(state.vehicle.position, max_surface_distance, state.track)
    modifiers = resolve_motion_modifiers(current_query)
    motion = compute_motion(state.vehicle, control, parameters, current_query.contact, modifiers)
    candidate = integrate(state.vehicle, motion, parameters, current_query.contact, dt)
    candidate_query = track.query_point(candidate.position, max_surface_distance, state.track)
    if body_geometry is not None:
        if not isinstance(body_geometry, VehicleBodyGeometry):
            raise ValueError("body_geometry must be VehicleBodyGeometry")
        # Prospective full-body engine: continuous translation + rotation.
        # Unresolved conservative intervals must block, not tunnel.
        body_hits = []
        for obstacle in track.obstacles:
            verdict = sweep_body_against_obstacle(
                state.vehicle, candidate, body_geometry, obstacle.volume
            )
            if verdict.uncertain:
                # Unknown is not evidence of a real collision. Fail closed,
                # but NEVER forge an ObstacleHit or a valid evaluation score.
                raise RuntimeError(
                    f"body sweep unresolved for obstacle {obstacle.id!r}; "
                    "candidate rollout is invalid, not a verified collision"
                )
            if verdict.contact:
                body_hits.append(obstacle)
        candidate_query = replace(candidate_query, obstacles=tuple(body_hits))

    accepted, obstacle_events = resolve_obstacle_block(state.vehicle, candidate, candidate_query)
    if body_geometry is not None:
        if ramp_support_policy is None:
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
        if not road_verdict.valid:
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
    )
    events: tuple[WorldEvent, ...] = (
        *obstacle_events,
        *checkpoint_events,
        *finish_events,
        *pickup_events,
    )
    return WorldStepResult(next_state, final_query, modifiers, events)
