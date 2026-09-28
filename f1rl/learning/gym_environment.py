from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..runtime.simulation import Simulation, SimulationConfig, SimulationSnapshot
from ..sandbox.geometry.vector import Vec3
from ..sandbox.track.model import TrackWorld
from ..sandbox.vehicle import VehicleParameters, VehicleState
from ..sandbox.world import WorldState
from .action import PolicyAction, map_policy_action
from .episode import (
    EpisodeConfig,
    RaceProgressResult,
    RaceProgressState,
    RaceRoute,
    advance_race_progress,
    episode_status,
)
from .observation import NavigationTarget, ObservationEncoder
from .rewards import RewardComposer, RewardResult, RewardTransition


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
        self._simulation = Simulation(track, initial_state, vehicle_parameters, simulation_config)
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
        target_position = self._route.target_for(progress)
        try:
            observation = self._encode_observation(snapshot.world.vehicle, target_position)
        except Exception:
            self._progress = None
            self._needs_reset = True
            raise

        self._progress = progress
        self._needs_reset = False
        return observation, self._reset_info()

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
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        progress_before = self._progress
        if self._needs_reset or progress_before is None:
            raise RuntimeError("reset() must be called before step()")

        policy_action = self._policy_action(action)
        target_before = self._route.target_for(progress_before)
        previous_vehicle = self._simulation.snapshot.world.vehicle
        control = map_policy_action(policy_action)

        snapshot_after = self._simulation.step(control)
        try:
            reward_result = self._reward_composer.evaluate(
                RewardTransition(
                    previous_vehicle=previous_vehicle,
                    next_vehicle=snapshot_after.world.vehicle,
                    target_position=target_before,
                    events=snapshot_after.events,
                    dt=self._simulation_config.fixed_dt,
                )
            )
            progress_result = advance_race_progress(
                self._route,
                progress_before,
                snapshot_after.events,
            )
            status = episode_status(
                progress_result.state,
                snapshot_after.tick,
                self._episode_config,
            )
            observation = self._encode_observation(
                snapshot_after.world.vehicle,
                progress_result.target_position,
            )
            info = self._step_info(snapshot_after, progress_result, reward_result)
        except Exception:
            self._needs_reset = True
            raise

        self._progress = progress_result.state
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
    ) -> dict[str, object]:
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
        }
