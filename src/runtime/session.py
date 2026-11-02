from __future__ import annotations

import json
import threading
import time
from hashlib import sha256
from queue import Empty, Full, Queue
from uuid import uuid4

import numpy as np

from src.runtime.auto_drive import AutoDriveController
from src.runtime.catalog import MAX_RUNTIME_STEPS, RuntimeCatalog
from src.runtime.drive_config import AutoDriveConfig
from src.runtime.frames import (
    EpisodeResult,
    ReplayDocument,
    ReplayProvenance,
    RuntimeAction,
    RuntimeFrame,
    build_runtime_frame,
)

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
        mode: str = "ppo",
        custom_factory=None,
        custom_track_sha256: str | None = None,
        auto_drive_config: AutoDriveConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.session_id = session_id
        if mode not in ("ppo", "manual", "auto"):
            raise ValueError("runtime mode must be ppo, manual, or auto")
        if mode == "ppo" and model_id is None:
            raise ValueError("PPO mode requires a registered frozen model")
        if mode in ("manual", "auto") and model_id is not None:
            raise ValueError(f"{mode} mode cannot select a PPO model")
        if mode != "auto" and auto_drive_config is not None:
            raise ValueError("Auto Drive configuration is allowed only for auto mode")
        self.auto_drive_config = (auto_drive_config or AutoDriveConfig()) if mode == "auto" else None
        self.mode = mode
        self.model = catalog.model(model_id) if model_id is not None else None
        self._custom_factory = custom_factory
        self._custom_track_sha256 = custom_track_sha256
        self.track = custom_factory if custom_factory is not None else catalog.track_factory(track_id)
        self.track_id = track_id
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
        with self._condition:
            if self.status not in ("running", "paused"):
                raise RuntimeFault("invalid_transition", f"session cannot reset from {self.status}", status_code=409)
            self._paused = False
            self._restart.set()
            self.status = "running"
            self._condition.notify_all()
            self._emit("status", {"status": "resetting"})

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
                "seed": self.seed,
                "status": self.status,
                "steps": self._steps,
                "max_steps": self.max_steps,
                "step_progress": min(1.0, self._steps / self.max_steps),
                "latest_frame": None if latest is None else latest.model_dump(mode="json"),
                "episode_result": None if self._result is None else self._result.model_dump(mode="json"),
                "error": self._error,
                "provenance": "TRAIN-development" if self.mode == "ppo" and self._custom_factory is None
                else "custom-development" if self._custom_factory is not None
                else "auto-development" if self.mode == "auto"
                else "manual-development",
                "speed_multiplier": self._speed_multiplier,
                **({"auto_drive_config": self.auto_drive_config.model_dump(mode="json"),
                    "effective_max_speed_kmh": 3.6 * min(self.track.vehicle_parameters.max_ground_speed,
                                                       self.auto_drive_config.target_max_speed_kmh / 3.6)}
                   if self.auto_drive_config is not None else {}),
            }

    def replay(self) -> dict[str, object]:
        with self._lock:
            if self._result is None:
                raise RuntimeFault("replay_not_ready", "episode replay is not available yet", status_code=409)
            if not self._frames:
                raise RuntimeFault("replay_unavailable", "session ended before a simulation frame was captured", status_code=409)
            document = ReplayDocument(
                schema="f1rl_racing_sandbox_replay_v2",
                label=f"{self.model.model_id if self.model is not None else 'Auto Drive' if self.mode == 'auto' else 'Manual'} · {self.track_id} · seed {self.seed}",
                provenance=ReplayProvenance(
                    dataset_split="TRAIN" if self.mode == "ppo" and self._custom_factory is None else "DEVELOPMENT",
                    source_sha256=self.catalog.source_sha256,
                    model_id=self.model.model_id if self.model is not None else "auto-drive" if self.mode == "auto" else "manual",
                    model_sha256=None if self.model is None else self.model.sha256,
                    track_id=self.track_id,
                    track_sha256=self._custom_track_sha256 or self.catalog.track_sha256(self.track_id),
                    seed=self.seed,
                ),
                result=self._result,
                frames=list(self._frames),
                auto_drive_config=self.auto_drive_config.model_dump(mode="json") if self.auto_drive_config else None,
            )
            result = document.model_dump(mode="json", by_alias=True)
            if self.auto_drive_config is None:
                result.pop("auto_drive_config", None)
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
        })
        if frame.checkpoint_crossings:
            self._emit("checkpoint", {"crossings": frame.checkpoint_crossings, "completed": frame.completed_checkpoint_ids})
        if frame.collision_ids:
            self._emit("collision", {"count": frame.collision_count, "ids": frame.collision_ids})

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
            from src.evaluation.rollout import evaluate_policy_episode

            while not self._cancel.is_set():
                with self._lock:
                    self._frames = []
                    self._steps = 0
                    self._result = None
                    self._error = None
                    self._last_step_started = None
                env = (
                    self.catalog.create_environment(self.track_id, seed=self.seed)
                    if self._custom_factory is None
                    else self._custom_factory.create()
                )
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
                        self.track.navigation,
                        self.track.vehicle_parameters,
                        min(surface.width for surface in self.track.track.surfaces),
                        self.auto_drive_config,
                        obstacles=self.track.track.obstacles,
                        body_geometry=(self.track.body_geometry or self.track.physics_config.body_geometry),
                    )
                self._emit("status", {"status": "running", "reset": self._restart.is_set()})
                row = evaluate_policy_episode(
                    env,
                    model,
                    policy_name="trained_final" if self.mode == "ppo" else "manual" if self.mode == "manual" else "auto_drive",
                    episode=0,
                    environment_seed=self.seed,
                    on_reset=lambda observation, info, env=env: self._capture_reset(env, observation, info),
                    before_step=self._before_step,
                    on_frame=lambda step, action, transition, env=env: self._capture_step(
                        env, step, action, transition
                    ),
                )
                if self._restart.is_set() and not self._cancel.is_set():
                    self._restart.clear()
                    continue
                self._complete(row)
                break
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
        registered_ppo = self.mode == "ppo" and self._custom_factory is None
        status = (
            "cancelled" if cancelled
            else "failed" if error or (registered_ppo and row.get("valid_sample") is not True)
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
            self._error = None if error is None else str(error)
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
        mode: str = "ppo",
        custom_factory=None,
        custom_track_sha256: str | None = None,
        auto_drive_config: AutoDriveConfig | None = None,
    ) -> RuntimeSession:
        if mode == "ppo":
            if model_id is None:
                raise ValueError("PPO mode requires a registered frozen model")
            self.catalog.model(model_id)
        elif mode in ("manual", "auto"):
            if model_id is not None:
                raise ValueError(f"{mode} mode cannot select a PPO model")
        else:
            raise ValueError("runtime mode must be ppo, manual, or auto")
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
