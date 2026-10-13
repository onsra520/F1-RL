from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

from .body_collision import VehicleBodyGeometry, sweep_body_against_obstacle
from .dynamics.integrator import integrate
from .dynamics.modifiers import MotionModifiers
from .dynamics.vehicle_dynamics import compute_motion
from .interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
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
) -> WorldStepResult:
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
        body_hits = tuple(
            obstacle for obstacle in track.obstacles
            if sweep_body_against_obstacle(
                state.vehicle, candidate, body_geometry, obstacle.volume
            ).blocked
        )
        candidate_query = replace(candidate_query, obstacles=body_hits)

    accepted, obstacle_events = resolve_obstacle_block(state.vehicle, candidate, candidate_query)
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
