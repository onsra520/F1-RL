from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..sandbox.track.model import TrackWorld
from ..sandbox.track.query import TrackQuery
from ..sandbox.vehicle import VehicleParameters
from ..sandbox.world import WorldEvent, WorldState


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    fixed_dt: float
    max_surface_distance: float

    def __post_init__(self) -> None:
        fixed_dt = _finite_number(self.fixed_dt, "fixed_dt")
        max_surface_distance = _finite_number(self.max_surface_distance, "max_surface_distance")
        if fixed_dt <= 0.0:
            raise ValueError("fixed_dt must be strictly positive")
        if max_surface_distance < 0.0:
            raise ValueError("max_surface_distance must be non-negative")


@dataclass(frozen=True, slots=True)
class SimulationState:
    tick: int
    elapsed_time: float
    world: WorldState

    def __post_init__(self) -> None:
        if isinstance(self.tick, bool) or not isinstance(self.tick, int) or self.tick < 0:
            raise ValueError("tick must be a non-negative integer")
        elapsed_time = _finite_number(self.elapsed_time, "elapsed_time")
        if elapsed_time < 0.0:
            raise ValueError("elapsed_time must be non-negative")


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    tick: int
    elapsed_time: float
    world: WorldState
    query: TrackQuery
    events: tuple[WorldEvent, ...]


class Simulation:
    __slots__ = (
        "_track",
        "_initial_state",
        "_parameters",
        "_config",
        "_state",
        "_snapshot",
    )

    def __init__(
        self,
        track: TrackWorld,
        initial_state: WorldState,
        parameters: VehicleParameters,
        config: SimulationConfig,
    ) -> None:
        self._track = track
        self._initial_state = initial_state
        self._parameters = parameters
        self._config = config
        self._state = SimulationState(0, 0.0, initial_state)
        self._snapshot = self._snapshot_for(initial_state)

    @property
    def snapshot(self) -> SimulationSnapshot:
        return self._snapshot

    def reset(self) -> SimulationSnapshot:
        self._state = SimulationState(0, 0.0, self._initial_state)
        self._snapshot = self._snapshot_for(self._initial_state)
        return self._snapshot

    def _snapshot_for(self, world: WorldState) -> SimulationSnapshot:
        query = self._track.query_point(
            world.vehicle.position,
            self._config.max_surface_distance,
            world.track,
        )
        return SimulationSnapshot(0, 0.0, world, query, ())
