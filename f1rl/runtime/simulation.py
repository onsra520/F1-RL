from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..sandbox.body_collision import VehicleBodyGeometry
from ..sandbox.ramp_support import PlanarRampPolicy
from ..sandbox.track.model import TrackWorld
from ..sandbox.track.query import TrackQuery
from ..sandbox.vehicle import VehicleControl, VehicleParameters
from ..sandbox.world import WorldEvent, WorldState, step_world


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
        "_body_geometry",
        "_ramp_support_policy",
        "_state",
        "_snapshot",
    )

    def __init__(
        self,
        track: TrackWorld,
        initial_state: WorldState,
        parameters: VehicleParameters,
        config: SimulationConfig,
        *,
        body_geometry: VehicleBodyGeometry | None = None,
        ramp_support_policy: PlanarRampPolicy | None = None,
    ) -> None:
        self._track = track
        self._initial_state = initial_state
        self._parameters = parameters
        self._config = config
        if body_geometry is not None and not isinstance(body_geometry, VehicleBodyGeometry):
            raise ValueError('body_geometry must be VehicleBodyGeometry or None')
        if ramp_support_policy is not None and (
            body_geometry is None or not isinstance(ramp_support_policy, PlanarRampPolicy)
        ):
            raise ValueError('ramp_support_policy requires full-body geometry and PlanarRampPolicy')
        self._body_geometry = body_geometry
        self._ramp_support_policy = ramp_support_policy
        self._state = SimulationState(0, 0.0, initial_state)
        self._snapshot = self._snapshot_for(initial_state)

    @property
    def snapshot(self) -> SimulationSnapshot:
        return self._snapshot

    def step(self, control: VehicleControl) -> SimulationSnapshot:
        current = self._state
        result = step_world(
            self._track,
            current.world,
            control,
            self._parameters,
            self._config.fixed_dt,
            self._config.max_surface_distance,
            **({"body_geometry": self._body_geometry} if self._body_geometry is not None else {}),
            **({"ramp_support_policy": self._ramp_support_policy} if self._ramp_support_policy is not None else {}),
        )
        next_tick = current.tick + 1
        elapsed_time = next_tick * self._config.fixed_dt
        next_state = SimulationState(next_tick, elapsed_time, result.state)
        next_snapshot = SimulationSnapshot(
            next_tick,
            elapsed_time,
            result.state,
            result.query,
            result.events,
        )
        self._state = next_state
        self._snapshot = next_snapshot
        return next_snapshot

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
