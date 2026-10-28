from __future__ import annotations

from math import isfinite

from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.state import InteractionState
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.state import TrackState
from src.simulation.vehicle import VehicleState
from src.simulation.world import WorldState

from .model import TrackAsset


def build_initial_world_state(
    asset: TrackAsset,
    *,
    max_surface_distance: float,
) -> WorldState:
    if not isinstance(asset, TrackAsset):
        raise ValueError("asset must be TrackAsset")
    if isinstance(max_surface_distance, bool) or not isinstance(max_surface_distance, (int, float)):
        raise ValueError("max_surface_distance must be a finite non-negative number")
    distance = float(max_surface_distance)
    if not isfinite(distance) or distance < 0.0:
        raise ValueError("max_surface_distance must be a finite non-negative number")

    grounded = asset.world.surface_at(asset.spawn.position, distance) is not None
    zero = Vec3(0.0, 0.0, 0.0)
    return WorldState(
        vehicle=VehicleState(
            position=asset.spawn.position,
            velocity=zero,
            orientation=asset.spawn.orientation,
            angular_velocity=zero,
            grounded=grounded,
        ),
        track=TrackState(frozenset()),
        interactions=InteractionState(frozenset(), frozenset()),
        score=ScoreState(0),
    )
