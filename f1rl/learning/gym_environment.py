from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..runtime.simulation import Simulation, SimulationConfig, SimulationSnapshot
from ..sandbox.body_collision import VehicleBodyGeometry
from ..sandbox.ramp_support import PlanarRampPolicy
from ..sandbox.geometry.vector import Vec3
from ..sandbox.interactions.events import ObstacleHit
from ..sandbox.track.model import TrackWorld
from ..sandbox.track.query import TrackQuery
from ..sandbox.vehicle import VehicleParameters, VehicleState
from ..track_assets.model import AuthoredNavigation
from ..sandbox.world import WorldState
from .action import PolicyAction, map_policy_action
from .episode import (
    EpisodeConfig,
    EpisodeStatus,
    RaceProgressResult,
    RaceProgressState,
    RaceRoute,
    advance_race_progress,
    episode_status,
)
from .observation import NavigationTarget, ObservationEncoder
from .offtrack import OffTrackConfig, OffTrackState
from .route_navigation import AuthoredRoute
from .rewards import RewardComposer, RewardResult, RewardTransition, RouteProgressReward, OffTrackPenalty


def _vec3_tuple(value: Vec3) -> tuple[float, float, float]:
    return (float(value.x), float(value.y), float(value.z))


def _applied_modifier_ids(query: TrackQuery) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if query.contact is None:
        return (), ()
    return (
        tuple(item.id for item in sorted(query.boost_pads, key=lambda item: item.id)),
        tuple(item.id for item in sorted(query.slow_zones, key=lambda item: item.id)),
    )


def _obstacle_hit_ids(snapshot: SimulationSnapshot) -> tuple[str, ...]:
    return tuple(
        event.obstacle_id
        for event in snapshot.events
        if isinstance(event, ObstacleHit)
    )


def _vehicle_telemetry(
    vehicle: VehicleState,
    *,
    obstacle_hit_ids: tuple[str, ...] = (),
    applied_boost_pad_ids: tuple[str, ...] = (),
    applied_slow_zone_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "vehicle_position": _vec3_tuple(vehicle.position),
        "vehicle_velocity": _vec3_tuple(vehicle.velocity),
        "vehicle_speed": float(vehicle.speed),
        "grounded": vehicle.grounded,
        "obstacle_hit_ids": obstacle_hit_ids,
        "applied_boost_pad_ids": applied_boost_pad_ids,
        "applied_slow_zone_ids": applied_slow_zone_ids,
    }


class F1RLEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        track: TrackWorld,
        initial_state: WorldState,
        vehicle_parameters: VehicleParameters,
        simulation_config: SimulationConfig,
        observation_encoder: ObservationEncoder,
        reward_composer: RewardComposer,
        episode_config: EpisodeConfig,
        navigation: AuthoredNavigation | None = None,
        offtrack_config: OffTrackConfig | None = None,
        body_geometry: VehicleBodyGeometry | None = None,
        ramp_support_policy: PlanarRampPolicy | None = None,
    ) -> None:
        super().__init__()
        values = (
            (track, TrackWorld, "track"),
            (initial_state, WorldState, "initial_state"),
            (vehicle_parameters, VehicleParameters, "vehicle_parameters"),
            (simulation_config, SimulationConfig, "simulation_config"),
            (observation_encoder, ObservationEncoder, "observation_encoder"),
            (reward_composer, RewardComposer, "reward_composer"),
            (episode_config, EpisodeConfig, "episode_config"),
        )
        for value, expected_type, name in values:
            if not isinstance(value, expected_type):
                raise ValueError(f"{name} must be {expected_type.__name__}")

        self._track = track
        self._simulation_config = simulation_config
        self._observation_encoder = observation_encoder
        self._reward_composer = reward_composer
        self._episode_config = episode_config
        self._route = RaceRoute.from_track(track)
        self._authored_route: AuthoredRoute | None = None
        if navigation is not None:
            if not isinstance(navigation, AuthoredNavigation):
                raise ValueError("navigation must be AuthoredNavigation or None")
            if navigation.points[0] != initial_state.vehicle.position:
                raise ValueError("navigation start must match initial vehicle position")
            if navigation.points[-1] != self._route.finish.target_position:
                raise ValueError("navigation finish must match track finish")
            if len(navigation.checkpoint_point_indices) != len(self._route.checkpoints):
                raise ValueError("navigation checkpoint count must match track")
            if any(
                navigation.points[index] != checkpoint.target_position
                for index, checkpoint in zip(navigation.checkpoint_point_indices, self._route.checkpoints, strict=True)
            ):
                raise ValueError("navigation checkpoints must match track order")
            if not set(navigation.surface_ids).issubset({surface.id for surface in track.surfaces}):
                raise ValueError("navigation references unknown track surface")
            self._authored_route = AuthoredRoute(navigation)
        if self._authored_route is None and any(isinstance(term, RouteProgressReward) for term in reward_composer.terms):
            raise ValueError("route_progress reward requires authored navigation")
        if offtrack_config is not None and not isinstance(offtrack_config, OffTrackConfig):
            raise ValueError("offtrack_config must be OffTrackConfig or None")
        if offtrack_config is not None and self._authored_route is None:
            raise ValueError("offtrack_config requires authored navigation")
        if offtrack_config is None and any(isinstance(term, OffTrackPenalty) for term in reward_composer.terms):
            raise ValueError("offtrack penalty requires offtrack_config")
        self._offtrack_config = offtrack_config
        self._offtrack_state = OffTrackState()
        self._max_route_distance = min((surface.width for surface in track.surfaces), default=0.0) / 2.0
        self._route_highwater = 0.0
        self._simulation = Simulation(
            track, initial_state, vehicle_parameters, simulation_config,
            body_geometry=body_geometry,
            ramp_support_policy=ramp_support_policy,
        )
        self._progress: RaceProgressState | None = None
        self._needs_reset = True

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        sensor_count = len(observation_encoder.config.sensor_suite.rangefinders)
        observation_low = np.array(
            [0.0] * sensor_count
            + [-1.0] * 6
            + [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        observation_high = np.ones(observation_encoder.size, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=observation_low,
            high=observation_high,
            shape=(observation_encoder.size,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if options:
            raise ValueError("non-empty reset options are not supported")
        super().reset(seed=seed)

        snapshot = self._simulation.reset()
        progress = RaceProgressState(0, False)
        target_position = self._route_target(snapshot.world.vehicle, progress)
        try:
            observation = self._encode_observation(snapshot.world.vehicle, target_position)
        except Exception:
            self._progress = None
            self._needs_reset = True
            raise

        self._progress = progress
        self._offtrack_state = OffTrackState()
        self._route_highwater = 0.0
        self._needs_reset = False
        return observation, self._reset_info()

    def _route_target(self, vehicle: VehicleState, progress: RaceProgressState) -> Vec3 | None:
        if not isinstance(vehicle, VehicleState):
            raise ValueError("vehicle must be VehicleState")
        if self._authored_route is None:
            return self._route.target_for(progress)
        if progress.finished:
            return None
        return self._authored_route.project(
            vehicle.position, next_checkpoint_index=progress.next_checkpoint_index
        ).endpoint

    def _encode_observation(
        self,
        vehicle: VehicleState,
        target_position: Vec3 | None,
    ) -> np.ndarray:
        target = NavigationTarget(target_position) if target_position is not None else None
        try:
            raw = self._observation_encoder.encode(self._track, vehicle, target)
            observation = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("encoded observation must be numeric") from error

        if observation.shape != self.observation_space.shape:
            raise ValueError("encoded observation has unexpected shape")
        if not np.isfinite(observation).all():
            raise ValueError("encoded observation must be finite")
        if not self.observation_space.contains(observation):
            raise ValueError("encoded observation is outside observation_space")
        return observation

    def _reset_info(self) -> dict[str, object]:
        snapshot = self._simulation.snapshot
        return {
            "tick": 0,
            "elapsed_time": 0.0,
            "next_checkpoint_index": 0,
            "finished": False,
            "completed_checkpoint_id": None,
            "finished_now": False,
            "game_score": snapshot.world.score.points,
            "reward_components": {},
            "telemetry": _vehicle_telemetry(snapshot.world.vehicle),
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        progress_before = self._progress
        if self._needs_reset or progress_before is None:
            raise RuntimeError("reset() must be called before step()")

        policy_action = self._policy_action(action)
        previous_vehicle = self._simulation.snapshot.world.vehicle
        target_before = self._route_target(previous_vehicle, progress_before)
        control = map_policy_action(policy_action)
        query_before = self._simulation.snapshot.query

        snapshot_after = self._simulation.step(control)
        try:
            progress_result = advance_race_progress(
                self._route,
                progress_before,
                snapshot_after.events,
            )
            next_highwater = self._route_highwater
            route_delta = None
            if self._authored_route is not None:
                projection = self._authored_route.project(
                    snapshot_after.world.vehicle.position,
                    next_checkpoint_index=progress_result.state.next_checkpoint_index,
                )
                next_highwater = self._authored_route.advance_highwater(
                    self._route_highwater,
                    projection,
                    max_path_distance=self._max_route_distance,
                )
                route_delta = next_highwater - self._route_highwater
            next_offtrack = self._offtrack_state
            lateral_distance = None
            if self._offtrack_config is not None:
                assert self._authored_route is not None
                lateral_distance = self._authored_route.horizontal_distance(
                    snapshot_after.world.vehicle.position
                )
                next_offtrack = self._offtrack_state.advance(
                    lateral_distance, self._offtrack_config
                )
            reward_result = self._reward_composer.evaluate(
                RewardTransition(
                    previous_vehicle=previous_vehicle,
                    next_vehicle=snapshot_after.world.vehicle,
                    target_position=target_before,
                    events=snapshot_after.events,
                    dt=self._simulation_config.fixed_dt,
                    route_progress_delta_meters=route_delta,
                    off_track=next_offtrack.outside if self._offtrack_config is not None else None,
                )
            )
            status = episode_status(
                progress_result.state,
                snapshot_after.tick,
                self._episode_config,
            )
            if next_offtrack.truncated and not progress_result.state.finished:
                status = EpisodeStatus(False, True)
            observation = self._encode_observation(
                snapshot_after.world.vehicle,
                self._route_target(snapshot_after.world.vehicle, progress_result.state),
            )
            info = self._step_info(
                snapshot_after,
                progress_result,
                reward_result,
                query_before,
            )
            if self._authored_route is not None:
                info["route_highwater_meters"] = next_highwater
            if self._offtrack_config is not None:
                info["offtrack_lateral_meters"] = lateral_distance
                info["offtrack_consecutive_steps"] = next_offtrack.consecutive_steps
                info["truncation_reason"] = (
                    "offtrack" if next_offtrack.truncated and not progress_result.state.finished else None
                )
        except Exception:
            self._needs_reset = True
            raise

        self._progress = progress_result.state
        self._offtrack_state = next_offtrack
        self._route_highwater = next_highwater
        self._needs_reset = status.terminated or status.truncated
        return (
            observation,
            float(reward_result.total),
            status.terminated,
            status.truncated,
            info,
        )

    def _policy_action(self, action: np.ndarray) -> PolicyAction:
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be a numpy.ndarray")
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        if action.dtype.kind not in "iuf":
            raise ValueError("action must contain real numeric values")
        if not np.isfinite(action).all():
            raise ValueError("action values must be finite")
        return PolicyAction(float(action[0]), float(action[1]))

    def _step_info(
        self,
        snapshot: SimulationSnapshot,
        progress_result: RaceProgressResult,
        reward_result: RewardResult,
        query_before: TrackQuery,
    ) -> dict[str, object]:
        boost_ids, slow_ids = _applied_modifier_ids(query_before)
        return {
            "tick": snapshot.tick,
            "elapsed_time": snapshot.elapsed_time,
            "next_checkpoint_index": progress_result.state.next_checkpoint_index,
            "finished": progress_result.state.finished,
            "completed_checkpoint_id": progress_result.completed_checkpoint_id,
            "finished_now": progress_result.finished_now,
            "game_score": snapshot.world.score.points,
            "reward_components": {
                contribution.name: float(contribution.value)
                for contribution in reward_result.contributions
            },
            "telemetry": _vehicle_telemetry(
                snapshot.world.vehicle,
                obstacle_hit_ids=_obstacle_hit_ids(snapshot),
                applied_boost_pad_ids=boost_ids,
                applied_slow_zone_ids=slow_ids,
            ),
        }
