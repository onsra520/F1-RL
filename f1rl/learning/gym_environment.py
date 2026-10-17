from __future__ import annotations

from typing import Any
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np

from ..runtime.simulation import Simulation, SimulationConfig, SimulationSnapshot
from ..runtime.physics import PhysicsConfig
from ..sandbox.body_collision import VehicleBodyGeometry
from ..sandbox.ramp_support import PlanarRampPolicy
from ..sandbox.geometry.vector import Vec3
from ..sandbox.interactions.events import ObstacleHit, CheckpointEntered, FinishLineEntered
from ..sandbox.ballistic_flight import FlightTransition
from ..sandbox.track.model import TrackWorld
from ..sandbox.track.query import TrackQuery
from ..sandbox.vehicle import VehicleParameters, VehicleState
from ..track_assets.model import AuthoredNavigation
from ..sandbox.world import WorldState
from .action import ActionContract, BrakingPolicyAction, PolicyAction, map_policy_action
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
from .wp7_race import WP7RaceConfig, RaceContractError, advance_directed_race
from .train_reset import TrainResetConfig, sample_train_spawn
from .rewards import (RewardComposer, RewardResult, RewardTransition, RewardContribution,
    RouteProgressReward, OffTrackPenalty, TargetProgressReward, TimePenalty, CollisionPenalty)


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
        physics_config: PhysicsConfig | None = None,
        race_config: WP7RaceConfig | None = None,
        reset_config: TrainResetConfig | None = None,
        reset_track_id: str | None = None,
        reset_rank: int = 0,
        reset_sequence: int = 0,
        action_contract: ActionContract | None = None,
    ) -> None:
        super().__init__()
        if action_contract is not None and not isinstance(action_contract,ActionContract):
            raise ValueError("action_contract must be ActionContract or None")
        self._action_contract = action_contract or ActionContract()
        if physics_config is not None and not isinstance(physics_config,PhysicsConfig):
            raise ValueError('physics_config must be PhysicsConfig or None')
        research = body_geometry is not None or (physics_config is not None and physics_config.mode!='legacy_v1')
        if (research and self._action_contract.version=='steering-throttle-brake-v2'
                and (physics_config is None or physics_config.ground_motion_version not in ('arc_distance_v1','coupled_longitudinal_v1','spatial_contact_v1','spatial_compliant_lifecycle_v1','spatial_compliant_atlas_lifecycle_v2','spatial_rotating_compliant_lifecycle_v3'))):
            raise ValueError('research braking requires distance ground integration')
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
        if (isinstance(race_config,WP7RaceConfig) and race_config.owner_terminal
                and any(type(t) not in (RouteProgressReward,OffTrackPenalty,TargetProgressReward,TimePenalty,CollisionPenalty)
                        for t in reward_composer.terms)):
            raise ValueError('owner outcome requires registered reward terms with valid-prefix semantics')
        self._episode_config = episode_config
        self._route = RaceRoute.from_track(track)
        self._authored_route: AuthoredRoute | None = None
        if navigation is not None:
            if not isinstance(navigation, AuthoredNavigation):
                raise ValueError("navigation must be AuthoredNavigation or None")
            (physics_config or PhysicsConfig('legacy_v1')).validate_navigation_spawn(
                navigation.points[0],initial_state.vehicle.position,vehicle_parameters,track)
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
        if race_config is not None:
            if not isinstance(race_config, WP7RaceConfig):
                raise ValueError("race_config must be WP7RaceConfig or None")
            if (physics_config is None or not race_config.supports_physics_mode(physics_config.mode)
                    or self._authored_route is None):
                raise ValueError("WP7 race requires registered grounded physics and authored navigation")
            if offtrack_config is not None:
                raise ValueError("WP7 race cannot mix historical C2 offtrack semantics")
        self._race_config = race_config
        if reset_config is not None:
            if not isinstance(reset_config, TrainResetConfig) or race_config is None:
                raise ValueError("TRAIN reset requires explicit config and WP7 race contract")
            if not reset_config.supports_physics_mode(physics_config.mode):
                raise ValueError("TRAIN reset requires grounded suspension mode")
            reset_config.validate_track(track, reset_track_id)
        elif reset_track_id is not None:
            raise ValueError("reset track identity requires TRAIN reset config")
        if any(type(i) is not int or i < 0 for i in (reset_rank, reset_sequence)):
            raise ValueError("reset rank/sequence must be nonnegative integers")
        self._reset_config = reset_config
        self._reset_track_id = reset_track_id
        self._reset_rank = reset_rank
        self._reset_sequence = reset_sequence
        self._nominal_initial_state = initial_state
        self._reset_metadata = None
        self._offtrack_state = OffTrackState()
        self._max_route_distance = min((surface.width for surface in track.surfaces), default=0.0) / 2.0
        self._route_highwater = 0.0
        self._simulation = Simulation(
            track, initial_state, vehicle_parameters, simulation_config,
            body_geometry=body_geometry,
            ramp_support_policy=ramp_support_policy,
            physics_config=physics_config,
        )
        self._physics_config = physics_config
        self._requires_scientific_admission = body_geometry is not None or (
            physics_config is not None and physics_config.mode != "legacy_v1"
        )
        self._progress: RaceProgressState | None = None
        self._needs_reset = True

        self._last_research_outcome: dict[str, object] | None = None

        self.action_space = gym.spaces.Box(
            low=np.array(self._action_contract.low, dtype=np.float32),
            high=np.array(self._action_contract.high, dtype=np.float32),
            shape=(len(self._action_contract.low),),
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

    @property
    def requires_scientific_admission(self) -> bool:
        return self._requires_scientific_admission

    @property
    def last_research_outcome(self) -> dict[str, object] | None:
        """Rejected research steps raise; their classification remains inspectable."""
        return None if self._last_research_outcome is None else dict(self._last_research_outcome)

    def _record_research_outcome(self, kind: str, valid: bool, reason: str) -> None:
        if self.requires_scientific_admission:
            self._last_research_outcome = {
                "version": "wp7_episode_outcome_v1", "kind": kind,
                "valid_sample": valid, "reason": reason,
            }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if options:
            raise ValueError("non-empty reset options are not supported")
        super().reset(seed=seed)

        self._last_research_outcome = None

        self._reset_metadata = None
        if self._reset_config is not None:
            self._needs_reset = True
            initial, metadata = sample_train_spawn(
                self._reset_config, self._reset_track_id, self._track,
                self._nominal_initial_state, self._physics_config, self._authored_route,
                rank=self._reset_rank, reset_index=self._reset_sequence,
            )
            snapshot = self._simulation.reset(initial_state=initial)
        else:
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
        if self._reset_config is not None:
            self._reset_sequence += 1
            self._reset_metadata = metadata
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
            **self._physics_info(snapshot),
            **self._race_info(),
            **self._train_reset_info(),
        }

    def _train_reset_info(self) -> dict[str, object]:
        if self._reset_metadata is None:
            return {}
        from copy import deepcopy
        return {"train_reset": deepcopy(self._reset_metadata)}

    def _race_info(self) -> dict[str, object]:
        if self._race_config is None:
            return {}
        return {"race": self._race_config.provenance(self._track, self._authored_route)}

    def _physics_info(self, snapshot: SimulationSnapshot) -> dict[str, object]:
        if self._physics_config is None:
            return {}
        telemetry = {
            name: asdict(value) if value is not None else None
            for name, value in (
                ("suspension", snapshot.suspension_telemetry),
                ("flight", snapshot.flight_telemetry),
                ("partial_support", snapshot.partial_support_telemetry),
                ("grounded_yaw", snapshot.grounded_yaw_telemetry),
                ("transition", snapshot.physics_transition),
            )
        }
        if snapshot.coupled_ground_telemetry is not None:
            telemetry['coupled_ground'] = asdict(snapshot.coupled_ground_telemetry)
        if snapshot.spatial_ground_telemetry is not None:
            telemetry['spatial_ground'] = asdict(snapshot.spatial_ground_telemetry)
        return {"physics": {
            "configuration": self._physics_config.document(),
            "sha256": self._physics_config.sha256,
            **telemetry,
        }}

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

        try:
            snapshot_after = self._simulation.step(control)
        except Exception as exc:
            if self._physics_config is not None and self._physics_config.mode != "legacy_v1":
                self._needs_reset = True
                self._record_research_outcome("invalid_physical_transition", False, str(exc))
            raise
        try:
            race_step = None
            if self._race_config is not None:
                race_step = advance_directed_race(
                    self._track, self._route, self._authored_route, progress_before, snapshot_after,
                    self._race_config,
                )
                progress_result = race_step.progress
            else:
                progress_result = advance_race_progress(
                    self._route, progress_before, snapshot_after.events,
                )
            reward_duration=self._simulation_config.fixed_dt
            reward_vehicle=snapshot_after.world.vehicle
            reward_events=snapshot_after.events
            if race_step is not None and self._race_config.owner_terminal:
                if race_step.terminal_time_s is not None:
                    reward_duration=race_step.terminal_time_s
                    segment=next(s for s in snapshot_after.center_segments
                                 if s.start_time_s<=reward_duration<=s.end_time_s)
                    reward_vehicle=replace(reward_vehicle,position=segment.position_at(reward_duration))
                physical=[]
                for event in snapshot_after.events:
                    if isinstance(event,(CheckpointEntered,FinishLineEntered)):
                        continue
                    if isinstance(event,FlightTransition):
                        if event.time_s<=reward_duration:
                            physical.append(event)
                    elif reward_duration<self._simulation_config.fixed_dt:
                        raise RaceContractError('terminal prefix contains untimed physical or pickup event')
                    else:
                        physical.append(event)
                reward_events=tuple(physical)+race_step.valid_gate_events
            next_highwater = self._route_highwater
            route_delta = None
            if self._authored_route is not None:
                projection = self._authored_route.project(
                    reward_vehicle.position,
                    next_checkpoint_index=progress_result.state.next_checkpoint_index,
                )
                next_highwater = self._authored_route.advance_highwater(
                    self._route_highwater,
                    projection,
                    max_path_distance=self._max_route_distance,
                )
                if reward_duration==0.:
                    next_highwater=self._route_highwater
                route_delta = next_highwater - self._route_highwater
            next_offtrack = self._offtrack_state
            lateral_distance = None
            if self._offtrack_config is not None:
                assert self._authored_route is not None
                lateral_distance = self._authored_route.horizontal_distance(
                    reward_vehicle.position
                )
                next_offtrack = self._offtrack_state.advance(
                    lateral_distance, self._offtrack_config
                )
            reward_result = (RewardResult(0.,tuple(RewardContribution(t.name,0.) for t in self._reward_composer.terms))
                             if reward_duration==0. else self._reward_composer.evaluate(
                RewardTransition(
                    previous_vehicle=previous_vehicle,
                    next_vehicle=reward_vehicle,
                    target_position=target_before,
                    events=reward_events,
                    dt=reward_duration,
                    route_progress_delta_meters=route_delta,
                    off_track=next_offtrack.outside if self._offtrack_config is not None else None,
                )
            ))
            status = episode_status(
                progress_result.state,
                snapshot_after.tick,
                self._episode_config,
            )
            if next_offtrack.truncated and not progress_result.state.finished:
                status = EpisodeStatus(False, True)
            if race_step is not None and race_step.terminal_failure:
                status=EpisodeStatus(True,False)
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
            if race_step is not None:
                info.update(self._race_info())
                info["race"].update({
                    "completed_checkpoint_ids": race_step.completed_checkpoint_ids,
                    "directed_crossings": race_step.directed_crossings,
                    "max_sampled_deviation_m": race_step.max_sampled_deviation_m,
                    "soft_zone_encountered": race_step.soft_zone_encountered,
                })
                if self._race_config.owner_terminal:
                    info['race'].update(terminal_failure=race_step.terminal_failure,
                        terminal_time_s=race_step.terminal_time_s,reward_valid_prefix_s=reward_duration,
                        reward_prefix_position=_vec3_tuple(reward_vehicle.position),
                        physical_tick_end_time_s=self._simulation_config.fixed_dt)
                if self._race_config.version in ('wp7_directed_lifecycle_research_v1','wp7_directed_grounded_owner_v2','wp7_directed_lifecycle_owner_v2'):
                    info["race"]["tick_local_crossing_times_s"] = race_step.crossing_times_s
                    info["race"]["crossing_time_origin_s"] = snapshot_after.physics_transition.start_time_s
            if self._authored_route is not None:
                info["route_highwater_meters"] = next_highwater
            if self._offtrack_config is not None:
                info["offtrack_lateral_meters"] = lateral_distance
                info["offtrack_consecutive_steps"] = next_offtrack.consecutive_steps
                info["truncation_reason"] = (
                    "offtrack" if next_offtrack.truncated and not progress_result.state.finished else None
                )
        except Exception as exc:
            self._needs_reset = True
            self._record_research_outcome(
                "uncertified_route_state" if isinstance(exc, RaceContractError) else "invalid_environment_transition",
                False, str(exc),
            )
            raise

        self._progress = progress_result.state
        self._offtrack_state = next_offtrack
        self._route_highwater = next_highwater
        self._needs_reset = status.terminated or status.truncated
        self._record_research_outcome(
            "valid_terminal_failure" if race_step is not None and race_step.terminal_failure else
            "successful_finish" if progress_result.state.finished else
            "valid_truncation" if status.truncated else "active", True,
            "hard_route_violation" if race_step is not None and race_step.terminal_failure else
            "finish" if progress_result.state.finished else
            (info.get("truncation_reason") or "step_limit") if status.truncated else "accepted_step",
        )
        if self.last_research_outcome is not None:
            info["research_outcome"] = self.last_research_outcome
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
        shape = (len(self._action_contract.low),)
        if action.shape != shape:
            raise ValueError(f"action must have shape {shape}")
        if action.dtype.kind not in "iuf":
            raise ValueError("action must contain real numeric values")
        if not np.isfinite(action).all():
            raise ValueError("action values must be finite")
        if shape == (3,):
            return BrakingPolicyAction(*(float(v) for v in action))
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
            **self._physics_info(snapshot),
            **self._train_reset_info(),
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
