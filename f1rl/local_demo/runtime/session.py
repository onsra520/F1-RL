from __future__ import annotations

import json
import threading
import time
from hashlib import sha256
from queue import Empty, Full, Queue
from uuid import uuid4

import numpy as np

from f1rl.local_demo.runtime.auto_drive import AutoDriveController
from f1rl.local_demo.runtime.adapter import RuntimeEnvironmentAdapter
from f1rl.local_demo.runtime.catalog import MAX_RUNTIME_STEPS, RuntimeCatalog
from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.local_demo.runtime.directed_gate_guidance import directed_gate_guidance
from f1rl.local_demo.runtime.exploratory_models import EXPLORATORY_IDS
from f1rl.local_demo.runtime.development_collision_terminal import RuntimeCollisionEvidence, inspect_collision_error
from f1rl.local_demo.runtime.frames import (
    EpisodeResult,
    ReplayDocument,
    ReplayProvenance,
    RuntimeAction,
    RuntimeFrame,
    build_runtime_frame,
)
from f1rl.local_demo.runtime.hybrid_drive import HybridDrive
from f1rl.local_demo.runtime.v4_dev_profile import bind_custom_auto_v4

MAX_ACTIVE_SESSIONS = 2
MAX_RETAINED_SESSIONS = 8
SESSION_TTL_SECONDS = 900.0


class ManualActionController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._action = np.zeros(3, dtype=np.float32)

    def set_action(self, steering: float, throttle: float, brake: float) -> None:
        action = RuntimeAction(steering=steering, throttle=throttle, brake=brake)
        with self._lock:
            self._action = np.asarray([action.steering, action.throttle, action.brake], dtype=np.float32)

    def stop(self) -> None:
        with self._lock:
            self._action = np.zeros(3, dtype=np.float32)

    def predict(self, _observation, *, deterministic: bool = True):
        del deterministic
        with self._lock:
            return self._action.copy()


class RuntimeFault(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class RuntimeSession:
    def __init__(
        self,
        catalog: RuntimeCatalog,
        *,
        session_id: str,
        model_id: str | None,
        track_id: str,
        seed: int,
        mode: str = "manual",
        custom_factory=None,
        custom_track_sha256: str | None = None,
        custom_ppo_compatible: bool | None = None,
        auto_drive_config: AutoDriveConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.session_id = session_id
        if mode not in ("ppo", "manual", "auto", "hybrid"):
            raise ValueError("unknown development runtime mode")
        if mode in ("ppo", "hybrid") and model_id is None:
            raise ValueError(f"{mode} mode requires a model id")
        if mode == "ppo" and model_id in EXPLORATORY_IDS:
            raise ValueError("exploratory checkpoints require Hybrid development mode")
        if mode == "hybrid" and model_id not in EXPLORATORY_IDS:
            raise ValueError("Hybrid mode requires an exploratory WP7 model")
        if mode in ("manual", "auto") and model_id is not None:
            raise ValueError(f"{mode} mode cannot select a PPO model")
        if mode not in ("auto", "hybrid") and auto_drive_config is not None:
            raise ValueError("Auto Drive configuration is allowed only for Auto Drive or Hybrid mode")
        if mode in ("ppo", "hybrid") and custom_factory is not None and custom_ppo_compatible is not True:
            raise ValueError("custom track is not compatible with the registered PPO observation/action contract")
        self.auto_drive_config = (auto_drive_config or AutoDriveConfig()) if mode in ("auto", "hybrid") else None
        self._hybrid_controller = None
        self.mode = mode
        self.model = catalog.model(model_id) if model_id is not None else None
        self._exploratory_ppo = model_id in EXPLORATORY_IDS
        self._custom_factory = custom_factory
        self._custom_track_sha256 = custom_track_sha256
        self.track = custom_factory if custom_factory is not None else catalog.track_factory(track_id)
        self.track_id = track_id
        self._guidance_navigation = self.track.navigation
        self._effective_auto_config = self.auto_drive_config
        self._uses_directed_gate_guidance = False
        if mode in ("auto", "hybrid") and custom_factory is None and self.track.navigation is not None:
            self._guidance_navigation, self._effective_auto_config, self._uses_directed_gate_guidance = directed_gate_guidance(
                track_id, catalog.track_sha256(track_id), self.track.navigation, self.auto_drive_config,
                self.track.initial_state.vehicle.position, self.track.initial_state.vehicle.orientation,
            )
        self.seed = seed
        self.max_steps = min(self.track.episode_config.max_steps, MAX_RUNTIME_STEPS)
        self._manual = ManualActionController() if mode == "manual" else None
        self.status = "ready"
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._restart = threading.Event()
        self._condition = threading.Condition(self._lock)
        self._paused = False
        self._paced = False
        self._speed_multiplier = 1.0
        self._last_step_started: float | None = None
        self._events: Queue[dict[str, object]] = Queue(maxsize=16)
        self._socket_claimed = False
        self._thread: threading.Thread | None = None
        self._frames: list[RuntimeFrame] = []
        self._steps = 0
        self._result: EpisodeResult | None = None
        self._error: str | None = None
        self._collision_terminal: RuntimeCollisionEvidence | None = None
        self._start_reset_epoch = False
        self._created_at = time.monotonic()
        self._finished_at: float | None = None

    def start(self, *, paced: bool = False) -> None:
        with self._lock:
            if self.status != "ready":
                raise RuntimeFault("invalid_transition", f"session cannot start from {self.status}", status_code=409)
            self.status = "running"
            self._paced = paced
            self._last_step_started = None
            self._thread = threading.Thread(
                target=self._run_episode,
                name=f"f1rl-{self.session_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        with self._condition:
            if self.status in ("running", "paused"):
                self._cancel.set()
                self._paused = False
                self._condition.notify_all()
            elif self.status == "ready":
                self.status = "stopped"
                self._finished_at = time.monotonic()
                self._emit("terminal", {"status": "stopped"})

    def pause(self) -> None:
        with self._condition:
            if self.status != "running":
                raise RuntimeFault("invalid_transition", f"session cannot pause from {self.status}", status_code=409)
            self._paused = True
            self.status = "paused"
            self._emit("status", {"status": self.status})

    def resume(self) -> None:
        with self._condition:
            if self.status != "paused":
                raise RuntimeFault("invalid_transition", f"session cannot resume from {self.status}", status_code=409)
            self._paused = False
            self.status = "running"
            self._condition.notify_all()
            self._emit("status", {"status": self.status})

    def reset(self) -> None:
        # A terminal WebSocket event can arrive just before the producer
        # thread leaves its finally block. Join outside the condition lock.
        if self.status == "failed":
            previous_thread = self._thread
            if previous_thread is not None and previous_thread is not threading.current_thread():
                previous_thread.join(timeout=5.0)
        restart_failed_episode = False
        with self._condition:
            if self.status == "failed":
                if self._thread is not None and self._thread.is_alive():
                    raise RuntimeFault("invalid_transition", "previous episode is still closing", status_code=409)
                self._cancel.clear()
                self._restart.clear()
                self._start_reset_epoch = True
                self.status = "ready"
                restart_failed_episode = True
                self._emit("status", {"status": "resetting"})
            else:
                if self.status not in ("running", "paused"):
                    raise RuntimeFault("invalid_transition", f"session cannot reset from {self.status}", status_code=409)
                self._paused = False
                self._restart.set()
                self.status = "running"
                self._condition.notify_all()
                self._emit("status", {"status": "resetting"})
        if restart_failed_episode:
            self.start(paced=self._paced)

    def set_speed(self, multiplier: float) -> None:
        if isinstance(multiplier, bool) or not isinstance(multiplier, (float, int)) or not 0.25 <= multiplier <= 16.0:
            raise RuntimeFault("invalid_speed", "simulation speed must be within [0.25, 16]", status_code=422)
        with self._condition:
            self._speed_multiplier = float(multiplier)
            self._condition.notify_all()
            self._emit("status", {"status": self.status, "speed_multiplier": self._speed_multiplier})

    def set_action(self, steering: float, throttle: float, brake: float) -> None:
        if self.mode != "manual" or self._manual is None:
            raise RuntimeFault("invalid_controller", "manual actions require a manual runtime session", status_code=409)
        if self.status not in ("running", "paused"):
            raise RuntimeFault("invalid_transition", f"manual action cannot be set from {self.status}", status_code=409)
        try:
            self._manual.set_action(steering, throttle, brake)
        except ValueError as error:
            raise RuntimeFault("invalid_action", "manual action is outside the registered action bounds", status_code=422) from error

    def next_event(self, timeout: float = 0.25) -> dict[str, object] | None:
        try:
            return self._events.get(timeout=timeout)
        except Empty:
            return None

    def claim_socket(self) -> None:
        with self._lock:
            if self._socket_claimed:
                raise RuntimeFault("session_owned", "runtime session already has a WebSocket controller", status_code=409)
            self._socket_claimed = True

    def release_socket(self) -> None:
        with self._lock:
            self._socket_claimed = False

    def wait(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return self.status != "running"
        thread.join(timeout)
        return not thread.is_alive()

    def close(self) -> None:
        self.cancel()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def status_document(self) -> dict[str, object]:
        with self._lock:
            latest = self._frames[-1] if self._frames else None
            return {
                "protocol_version": "f1rl_runtime_v1",
                "session_id": self.session_id,
                "mode": self.mode,
                "model_id": None if self.model is None else self.model.model_id,
                "model_sha256": None if self.model is None else self.model.sha256,
                "track_id": self.track_id,
                "track_sha256": self._custom_track_sha256 or self.catalog.track_sha256(self.track_id),
                "physics": {
                    "mode": self.track.physics_config.mode,
                    "ground_motion_version": self.track.physics_config.ground_motion_version,
                    "sha256": self.track.physics_config.sha256,
                },
                "race_contract": {"version": self.track.race_config.version}
                    if self.track.race_config is not None else None,
                "seed": self.seed,
                "status": self.status,
                "steps": self._steps,
                "max_steps": self.max_steps,
                "step_progress": min(1.0, self._steps / self.max_steps),
                "latest_frame": None if latest is None else latest.model_dump(mode="json"),
                "episode_result": None if self._result is None else self._result.model_dump(mode="json"),
                "error": self._error,
                **({"collision_terminal": self._collision_terminal.document()}
                   if self._collision_terminal is not None else {}),
                "provenance": "hybrid-development" if self.mode == "hybrid"
                else "exploratory-development" if self._exploratory_ppo
                else "TRAIN-development" if self.mode == "ppo" and self._custom_factory is None
                else "custom-development" if self._custom_factory is not None
                else "auto-development" if self.mode == "auto"
                else "manual-development",
                "speed_multiplier": self._speed_multiplier,
                **({"hybrid": self._hybrid_controller.status()} if self._hybrid_controller is not None else {}),
                **({"auto_drive_config": self.auto_drive_config.model_dump(mode="json"),
                    "effective_max_speed_kmh": 3.6 * min(self.track.vehicle_parameters.max_ground_speed,
                                                       self._effective_auto_config.target_max_speed_kmh / 3.6)}
                   if self.auto_drive_config is not None else {}),
                **({"auto_drive_guidance": "directed_gate_crossing",
                    "guidance_max_speed_kmh": self._effective_auto_config.target_max_speed_kmh}
                   if self._uses_directed_gate_guidance else {}),
            }

    def replay(self) -> dict[str, object]:
        with self._lock:
            if self._result is None:
                raise RuntimeFault("replay_not_ready", "episode replay is not available yet", status_code=409)
            if not self._frames:
                raise RuntimeFault("replay_unavailable", "session ended before a simulation frame was captured", status_code=409)
            label = f"{self.model.model_id if self.model is not None else 'Auto Drive' if self.mode == 'auto' else 'Manual'} · {self.track_id} · seed {self.seed}"
            if self.mode == "hybrid":
                label = f"Hybrid PPO+Auto · {self.model.model_id} · {self.track_id} · seed {self.seed}"
            if self._uses_directed_gate_guidance:
                if self.mode == "hybrid":
                    label += f" · gates · target {self._effective_auto_config.target_max_speed_kmh:g} km/h"
                else:
                    label += f" · directed gates · target {self._effective_auto_config.target_max_speed_kmh:g} km/h · requested {self.auto_drive_config.target_max_speed_kmh:g} km/h"
            document = ReplayDocument(
                schema="f1rl_racing_sandbox_replay_v2",
                label=label,
                provenance=ReplayProvenance(
                    dataset_split="TRAIN" if self._custom_factory is None else "DEVELOPMENT",
                    source_sha256=self.catalog.source_sha256,
                    model_id=self.model.model_id if self.model is not None else "auto-drive" if self.mode == "auto" else "manual",
                    model_sha256=None if self.model is None else self.model.sha256,
                    track_id=self.track_id,
                    track_sha256=self._custom_track_sha256 or self.catalog.track_sha256(self.track_id),
                    seed=self.seed,
                ),
                result=self._result,
                frames=list(self._frames),
                auto_drive_config=self._effective_auto_config.model_dump(mode="json") if self._effective_auto_config else None,
                hybrid=self._hybrid_controller.status() if self._hybrid_controller is not None else None,
                collision_terminal=(self._collision_terminal.document()
                                    if self._collision_terminal is not None else None),
            )
            result = document.model_dump(mode="json", by_alias=True)
            if self.auto_drive_config is None:
                result.pop("auto_drive_config", None)
            if self._collision_terminal is None:
                result.pop("collision_terminal", None)
            if self._hybrid_controller is None:
                result.pop("hybrid", None)
            return result

    def _capture_reset(self, env, _observation, info) -> None:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        frame = build_runtime_frame(
            env.runtime_snapshot(), action, info,
            previous_velocity=None,
            collision_count_before=0,
            completed_checkpoint_ids_before=(),
        )
        with self._lock:
            self._frames = [frame]
            self._steps = 0
        if self._manual is not None:
            self._manual.stop()
        self._emit("frame", frame.model_dump(mode="json"))
        self._emit("progress", {"steps": 0, "max_steps": self.max_steps, "step_progress": 0.0})

    def _capture_step(self, env, step, action, transition) -> None:
        info = transition[4]
        with self._lock:
            previous = self._frames[-1]
            frame = build_runtime_frame(
                env.runtime_snapshot(), action, info,
                previous_velocity=previous.velocity,
                collision_count_before=previous.collision_count,
                completed_checkpoint_ids_before=tuple(previous.completed_checkpoint_ids),
            )
            if len(self._frames) >= self.max_steps + 1:
                raise RuntimeFault("frame_limit", "runtime frame limit reached")
            self._frames.append(frame)
            self._steps = step
        document = frame.model_dump(mode="json")
        self._emit("frame", document)
        self._emit("progress", {
            "steps": step,
            "max_steps": self.max_steps,
            "step_progress": min(1.0, step / self.max_steps),
            "route_progress_fraction": frame.route_progress_fraction,
            **({"hybrid": self._hybrid_controller.status()} if self._hybrid_controller is not None else {}),
        })
        if frame.checkpoint_crossings:
            self._emit("checkpoint", {"crossings": frame.checkpoint_crossings, "completed": frame.completed_checkpoint_ids})
        if frame.collision_ids:
            self._emit("collision", {"count": frame.collision_count, "ids": frame.collision_ids})

    def _capture_failure(self, error: Exception, env) -> None:
        if self.mode not in ("auto", "hybrid") or self._custom_factory is None:
            return
        evidence = inspect_collision_error(error, env)
        if evidence is None:
            return
        with self._lock:
            self._collision_terminal = evidence
        # An out-of-band runtime event, NOT a Physics/Gym transition.
        self._emit("collision", {
            "count": len(evidence.obstacle_ids),
            "ids": list(evidence.obstacle_ids),
            "diagnostic_penalty": evidence.diagnostic_penalty,
            "physics_transition_accepted": False,
            "scientific_evidence": False,
        })

    def _before_step(self, _step: int) -> None:
        with self._condition:
            while self._paused and not self._cancel.is_set() and not self._restart.is_set():
                self._condition.wait()
            if self._cancel.is_set():
                raise RuntimeFault("cancelled", "runtime session cancelled")
            if self._restart.is_set():
                raise RuntimeFault("session_reset", "runtime session reset")
            if self._paced:
                dt = float(self.track.simulation_config.fixed_dt)
                while self._last_step_started is not None:
                    target = self._last_step_started + dt / self._speed_multiplier
                    remaining = target - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                    if self._cancel.is_set():
                        raise RuntimeFault("cancelled", "runtime session cancelled")
                    if self._restart.is_set():
                        raise RuntimeFault("session_reset", "runtime session reset")
                    while self._paused and not self._cancel.is_set() and not self._restart.is_set():
                        self._condition.wait()
                self._last_step_started = time.monotonic()

    def _run_episode(self) -> None:
        try:
            restart_epoch = self._start_reset_epoch
            self._start_reset_epoch = False
            while not self._cancel.is_set():
                with self._lock:
                    self._frames = []
                    self._steps = 0
                    self._result = None
                    self._error = None
                    self._collision_terminal = None
                    self._hybrid_controller = None
                    self._last_step_started = None
                env = None
                try:
                    raw_env = (
                        self.catalog.create_environment(self.track_id, seed=self.seed, mode=self.mode)
                        if self._custom_factory is None
                        else self._custom_factory.create()
                    )
                    env = RuntimeEnvironmentAdapter(raw_env)
                    if self.mode == "ppo":
                        assert self.model is not None
                        model = self.catalog.load_model(self.model.model_id, env=env)
                    elif self.mode == "manual":
                        assert self._manual is not None
                        model = self._manual
                    else:
                        if self.track.navigation is None:
                            raise RuntimeFault("missing_navigation", "Auto Drive requires an authored track centerline")
                        model = AutoDriveController(
                            env,
                            self._guidance_navigation,
                            self.track.vehicle_parameters,
                            min(surface.width for surface in self.track.track.surfaces),
                            self._effective_auto_config,
                            obstacles=self.track.track.obstacles,
                            checkpoints=self.track.track.checkpoints,
                            body_geometry=(self.track.body_geometry or self.track.physics_config.body_geometry),
                            route_limit_m=self.track.race_config.hard_corridor_m if self.track.race_config else None,
                            allow_compact_corridor=True,
                            allow_centerline_avoidance=self.mode == "hybrid",
                        )
                        if self.mode == "hybrid":
                            assert self.model is not None
                            ppo = self.catalog.load_model(
                                self.model.model_id,
                                env=env,
                                observation_encoder=self.track.observation_encoder,
                            )
                            model = HybridDrive(
                                ppo,
                                model,
                                0.2 if self._uses_directed_gate_guidance else 0.5,
                            )
                            with self._lock:
                                self._hybrid_controller = model
                except Exception as error:
                    cleanup_error = None
                    if env is not None:
                        try:
                            env.close()
                        except Exception as close_error:
                            cleanup_error = close_error
                    with self._condition:
                        if self._restart.is_set() and not self._cancel.is_set() and cleanup_error is None:
                            self._restart.clear()
                            restart_epoch = True
                            continue
                        message = f"{type(error).__name__}: {error}"
                        if cleanup_error is not None:
                            message += f"; cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                        self._complete({"error": message, "valid_sample": False, "success": False,
                                        "steps": self._steps,
                                        "duration_s": self._frames[-1].time_s if self._frames else 0.0,
                                        "outcome": {}})
                    break
                # The restart request is cleared before this new episode.
                # Carry the epoch marker explicitly so browser clients can
                # discard stale frames and accept the new tick zero.
                self._emit("status", {"status": "running", "reset": restart_epoch})
                restart_epoch = False
                try:
                    observation, info = env.reset(seed=self.seed)
                    self._capture_reset(env, observation, info)
                    last_info = info
                    terminal = False
                    for step in range(1, self.max_steps + 1):
                        self._before_step(step)
                        prediction = model.predict(observation, deterministic=True)
                        action = prediction[0] if isinstance(prediction, tuple) else prediction
                        action = np.asarray(action, dtype=np.float32)
                        try:
                            transition = env.step(action)
                        except Exception as error:
                            self._capture_failure(error, env)
                            raise
                        observation, _reward, terminated, truncated, last_info = transition
                        self._capture_step(env, step, action, transition)
                        if terminated or truncated:
                            terminal = True
                            break
                    with self._condition:
                        if self._restart.is_set() and not self._cancel.is_set():
                            self._restart.clear()
                            restart_epoch = True
                            continue
                    success = last_info.get("finished") is True
                    self._complete({
                        "steps": self._steps,
                        "duration_s": self._frames[-1].time_s if self._frames else 0.0,
                        "success": success,
                        "valid_sample": False,
                        "outcome": {
                            "reason": "finish" if success else "episode_terminal" if terminal else "step_limit",
                        },
                    })
                except RuntimeFault as error:
                    if error.code == "session_reset" and self._restart.is_set() and not self._cancel.is_set():
                        self._restart.clear()
                        restart_epoch = True
                        continue
                    self._complete({"error": str(error), "valid_sample": False, "success": False,
                                    "steps": self._steps,
                                    "duration_s": self._frames[-1].time_s if self._frames else 0.0,
                                    "outcome": {}})
                except Exception as error:
                    if self._restart.is_set() and not self._cancel.is_set():
                        self._restart.clear()
                        restart_epoch = True
                        continue
                    self._complete({"error": f"{type(error).__name__}: {error}", "valid_sample": False,
                                    "success": False, "steps": self._steps,
                                    "duration_s": self._frames[-1].time_s if self._frames else 0.0,
                                    "outcome": {}})
                finally:
                    env.close()
                break
            else:
                with self._condition:
                    self._complete({"steps": self._steps,
                                    "duration_s": self._frames[-1].time_s if self._frames else 0.0})
        except Exception as error:  # noqa: BLE001 — convert simulator/model failures into a terminal runtime result
            self._complete({
                "error": f"{type(error).__name__}: {error}",
                "valid_sample": False,
                "success": False,
                "steps": self._steps,
                "duration_s": self._frames[-1].time_s if self._frames else 0.0,
                "outcome": {},
            })
        finally:
            with self._lock:
                if self.status not in ("running", "paused"):
                    self._finished_at = time.monotonic()

    def _complete(self, row: dict[str, object]) -> None:
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        cancelled = self._cancel.is_set()
        error = row.get("error")
        registered_ppo = False
        witnessed_collision = self._collision_terminal is not None and not cancelled
        status = (
            "cancelled" if cancelled
            else "failed" if (error or witnessed_collision or
                              (registered_ppo and row.get("valid_sample") is not True))
            else "finished"
        )
        trace_sha256 = row.get("trace_sha256")
        if not isinstance(trace_sha256, str) or len(trace_sha256) != 64:
            trace_sha256 = sha256(json.dumps(
                [frame.model_dump(mode="json") for frame in self._frames],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()
        reason = (
            "cancelled" if cancelled
            else "collision" if witnessed_collision
            else str(error) if error
            else str(outcome.get("reason")) if outcome.get("reason")
            else None
        )
        result = EpisodeResult(
            status=status,
            valid_sample=registered_ppo and row.get("valid_sample") is True and not cancelled,
            successful_finish=row.get("success") is True and not cancelled,
            reason=reason,
            steps=int(row.get("steps", self._steps)),
            duration_s=float(row.get("duration_s", 0.0)),
            trace_sha256=trace_sha256,
        )
        with self._lock:
            self._result = result
            self._error = None if error is None or witnessed_collision else str(error)
            self.status = status
            self._finished_at = time.monotonic()
        self._emit("terminal", result.model_dump(mode="json"))

    def _emit(self, event_type: str, data: dict[str, object]) -> None:
        event = {
            "protocol_version": "f1rl_runtime_ws_v1",
            "type": event_type,
            "session_id": self.session_id,
            "data": data,
        }
        try:
            self._events.put_nowait(event)
        except Full:
            # State frames are snapshots; discard stale queued state so the
            # sender always converges on current authoritative state.
            try:
                self._events.get_nowait()
            except Empty:
                pass
            try:
                self._events.put_nowait(event)
            except Full:
                pass


class RuntimeSessionManager:
    def __init__(self, catalog: RuntimeCatalog) -> None:
        self.catalog = catalog
        self._lock = threading.RLock()
        self._sessions: dict[str, RuntimeSession] = {}

    def create(
        self,
        *,
        model_id: str | None,
        track_id: str,
        seed: int,
        mode: str = "manual",
        custom_factory=None,
        custom_track_sha256: str | None = None,
        custom_ppo_compatible: bool | None = None,
        auto_drive_config: AutoDriveConfig | None = None,
        physics_profile: str | None = None,
    ) -> RuntimeSession:
        if mode in ("manual", "auto"):
            if model_id is not None:
                raise ValueError(f"{mode} mode cannot select a PPO model")
        elif mode == "ppo":
            if model_id is None or model_id in EXPLORATORY_IDS:
                raise ValueError("PPO mode requires a registered frozen model")
            self.catalog.model(model_id)
        elif mode == "hybrid":
            if model_id not in EXPLORATORY_IDS:
                raise ValueError("Hybrid mode requires an exploratory WP7 model")
            self.catalog.model(model_id)
        else:
            raise ValueError("unknown development runtime mode")
        if mode not in ("auto", "hybrid") and auto_drive_config is not None:
            raise ValueError("Auto Drive configuration is allowed only for Auto Drive or Hybrid mode")
        if mode in ("ppo", "hybrid") and custom_factory is not None and custom_ppo_compatible is not True:
            raise ValueError("custom track is not compatible with the registered PPO observation/action contract")
        if physics_profile not in (None, "experimental_v4"):
            raise ValueError("unregistered Auto Drive Physics development profile")
        if physics_profile == "experimental_v4":
            if mode != "auto" or custom_factory is None:
                raise ValueError("experimental Physics v4 requires custom Auto Drive")
            custom_factory = bind_custom_auto_v4(custom_factory)
        if custom_factory is None:
            self.catalog.track_factory(track_id)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise RuntimeFault("invalid_seed", "seed must be a uint32")
        with self._lock:
            self._expire_finished()
            active = sum(session.status in ("ready", "running", "paused") for session in self._sessions.values())
            if active >= MAX_ACTIVE_SESSIONS:
                raise RuntimeFault("session_limit", "runtime session limit reached", status_code=429)
            if len(self._sessions) >= MAX_RETAINED_SESSIONS:
                raise RuntimeFault("session_limit", "retained runtime session limit reached", status_code=429)
            session = RuntimeSession(
                self.catalog,
                session_id=str(uuid4()),
                model_id=model_id,
                track_id=track_id,
                seed=seed,
                mode=mode,
                custom_factory=custom_factory,
                custom_track_sha256=custom_track_sha256,
                custom_ppo_compatible=custom_ppo_compatible,
                auto_drive_config=auto_drive_config,
            )
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> RuntimeSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise RuntimeFault("session_not_found", "runtime session was not found", status_code=404) from error

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise RuntimeFault("session_not_found", "runtime session was not found", status_code=404)
        session.close()

    def shutdown(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def _expire_finished(self) -> None:
        now = time.monotonic()
        expired = [
            session_id for session_id, session in self._sessions.items()
            if session._finished_at is not None and now - session._finished_at >= SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
