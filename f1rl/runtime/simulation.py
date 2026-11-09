from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .physics import PhysicsConfig, PhysicsTransitionAudit
from ..sandbox.ballistic_flight import FlightStep
from ..sandbox.center_trajectory import CenterSegment
from ..sandbox.coupled_longitudinal_dynamics import CoupledGroundAudit
from ..sandbox.spatial_contact_dynamics import (
    SpatialContactGeometry, SpatialGroundAudit, prepare_spatial_contact_geometry,
)
from ..sandbox.grounded_yaw_dynamics import GroundedYawAudit
from ..sandbox.independent_suspension_dynamics import SuspensionDynamicsResult
from ..sandbox.partial_wheel_release import GroundSupportStep
from ..sandbox.body_collision import VehicleBodyGeometry
from ..sandbox.ramp_support import PlanarRampPolicy
from ..sandbox.track.model import TrackWorld
from ..sandbox.track.query import TrackQuery
from ..sandbox.vehicle import VehicleControl, VehicleParameters
from ..sandbox.world import WorldEvent, WorldState, step_world
from ..sandbox.physical_crash import PhysicalCrash
from ..sandbox.roll_envelope_terminal import RollEnvelopeStop


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
    suspension_telemetry: SuspensionDynamicsResult | None = None
    flight_telemetry: FlightStep | None = None
    partial_support_telemetry: GroundSupportStep | None = None
    grounded_yaw_telemetry: GroundedYawAudit | None = None
    physics_transition: PhysicsTransitionAudit | None = None
    center_segments: tuple[CenterSegment, ...] = ()
    coupled_ground_telemetry: CoupledGroundAudit | None = None
    spatial_ground_telemetry: SpatialGroundAudit | None = None
    physical_crash: PhysicalCrash | None = None
    roll_envelope_stop: RollEnvelopeStop | None = None


class Simulation:
    __slots__ = (
        "_track",
        "_initial_state",
        "_parameters",
        "_config",
        "_body_geometry",
        "_ramp_support_policy",
        "_physics_config",
        "_spatial_geometry_context",
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
        physics_config: PhysicsConfig | None = None,
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
        if physics_config is not None:
            if not isinstance(physics_config, PhysicsConfig):
                raise ValueError("physics_config must be PhysicsConfig or None")
            if body_geometry is not None or ramp_support_policy is not None:
                raise ValueError("physics_config cannot be mixed with legacy body/ramp options")
        self._physics_config = physics_config
        self._spatial_geometry_context: SpatialContactGeometry | None = None
        (physics_config or PhysicsConfig("legacy_v1")).validate_state(initial_state, parameters)
        if physics_config is not None:
            physics_config.validate_track(track)
        self._state = SimulationState(0, 0.0, initial_state)
        self._snapshot = self._snapshot_for(initial_state)

    @property
    def snapshot(self) -> SimulationSnapshot:
        return self._snapshot

    def step(self, control: VehicleControl) -> SimulationSnapshot:
        # An accepted crash is an absorbing end-of-episode state. Only reset()
        # may start a fresh trajectory; there is no post-impact future tick.
        if self._snapshot.physical_crash is not None:
            raise RuntimeError('absorbing physical crash terminal: reset required')
        if self._snapshot.roll_envelope_stop is not None:
            raise RuntimeError('absorbing certified roll-envelope terminal: reset required')
        current = self._state
        geometry_context = self._spatial_geometry_context_for_step()
        result = step_world(
            self._track,
            current.world,
            control,
            self._parameters,
            self._config.fixed_dt,
            self._config.max_surface_distance,
            **({"body_geometry": self._body_geometry} if self._body_geometry is not None else {}),
            **({"ramp_support_policy": self._ramp_support_policy} if self._ramp_support_policy is not None else {}),
            **(self._physics_config.world_options() if self._physics_config is not None else {}),
            **({"spatial_geometry_context": geometry_context}
               if geometry_context is not None else {}),
        )
        next_tick = current.tick + 1
        elapsed_time = next_tick * self._config.fixed_dt
        transition = None
        if self._physics_config is not None:
            transition = self._physics_config.validate_transition(
                result, previous=current.world, parameters=self._parameters,
                dt=self._config.fixed_dt, start_time_s=current.elapsed_time,
                end_time_s=elapsed_time,
            )
        next_state = SimulationState(next_tick, elapsed_time, result.state)
        next_snapshot = SimulationSnapshot(
            next_tick,
            elapsed_time,
            result.state,
            result.query,
            result.events,
            result.suspension_telemetry,
            result.flight_telemetry,
            result.partial_support_telemetry,
            result.grounded_yaw_telemetry,
            transition,
            result.center_segments,
            result.coupled_ground_telemetry,
            result.spatial_ground_telemetry,
            result.crash,
            result.roll_envelope_stop,
        )
        self._state = next_state
        self._snapshot = next_snapshot
        return next_snapshot

    def reset(self, *, initial_state: WorldState | None = None) -> SimulationSnapshot:
        initial = self._initial_state if initial_state is None else initial_state
        if initial_state is not None:
            if self._physics_config is None or self._physics_config.mode == "legacy_v1":
                raise ValueError("alternate reset state requires registered research physics")
            if not isinstance(initial, WorldState):
                raise ValueError("reset initial_state must be WorldState")
            self._physics_config.validate_state(initial, self._parameters)
        snapshot = self._snapshot_for(initial)
        self._state = SimulationState(0, 0.0, initial)
        self._snapshot = snapshot
        return self._snapshot

    def _spatial_geometry_context_for_step(self) -> SpatialContactGeometry | None:
        physics = self._physics_config
        if (physics is None
                or physics.ground_motion_version != 'spatial_road_edge_partial_lifecycle_v4'):
            return None
        suspension = physics.independent_suspension_policy
        body = physics.body_geometry
        if suspension is None or body is None:
            raise ValueError('Physics v4 geometry context requires registered suspension and body')
        seam_tolerance = suspension.contact.seam_height_tolerance_m
        context = self._spatial_geometry_context
        if context is None or not context.matches(self._track.surfaces,seam_tolerance,body):
            context = prepare_spatial_contact_geometry(self._track.surfaces,seam_tolerance,body)
            self._spatial_geometry_context = context
        return context

    def _snapshot_for(self, world: WorldState) -> SimulationSnapshot:
        query = self._track.query_point(
            world.vehicle.position,
            self._config.max_surface_distance,
            world.track,
        )
        return SimulationSnapshot(0, 0.0, world, query, ())
