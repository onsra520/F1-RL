from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class RuntimeAction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    steering: float = Field(ge=-1.0, le=1.0)
    throttle: float = Field(ge=0.0, le=1.0)
    brake: float = Field(ge=0.0, le=1.0)


class RuntimeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    valid_sample: bool
    reason: str
    scientific_evidence: Literal[False] = False


class RuntimeFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["f1rl_runtime_frame_v1"]
    tick: int = Field(ge=0)
    time_s: float = Field(ge=0.0)
    dt_s: float = Field(gt=0.0)
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration_mps2: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    heading_rad: float
    speed_mps: float = Field(ge=0.0)
    grounded: bool
    action: RuntimeAction
    route_progress_m: float | None = Field(default=None, ge=0.0)
    route_progress_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    route_highwater_m: float = Field(ge=0.0)
    next_checkpoint_index: int | None = Field(default=None, ge=0)
    checkpoint_count: int = Field(ge=0)
    completed_checkpoint_ids: list[str]
    checkpoint_crossings: list[str]
    collision_count: int = Field(ge=0)
    collision_ids: list[str]
    physical_event_types: list[str]
    outcome: RuntimeOutcome | None = None


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: Literal["finished", "failed", "cancelled", "stopped"]
    valid_sample: bool
    successful_finish: bool
    reason: str | None
    steps: int = Field(ge=0)
    duration_s: float = Field(ge=0.0)
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReplayProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_split: Literal["TRAIN", "DEVELOPMENT"]
    runtime_mode: Literal["development"] = "development"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    track_id: str
    track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0, le=2**32 - 1)


class ReplayDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["f1rl_racing_sandbox_replay_v2"] = Field(alias="schema")
    label: str = Field(min_length=1, max_length=120)
    provenance: ReplayProvenance
    result: EpisodeResult
    frames: list[RuntimeFrame] = Field(min_length=2, max_length=5001)
    auto_drive_config: dict[str, object] | None = None

    @model_validator(mode="after")
    def ordered_frames(self):
        if any(
            later.time_s <= earlier.time_s
            for earlier, later in zip(self.frames, self.frames[1:])
        ):
            raise ValueError("replay frame times must be strictly increasing")
        return self


def _action_document(action: object) -> RuntimeAction:
    try:
        values = np.asarray(action)
    except (TypeError, ValueError) as error:
        raise ValueError("action must contain two or three finite values") from error
    if values.dtype.kind not in "iuf" or values.ndim != 1 or values.shape not in ((2,), (3,)):
        raise ValueError("action must contain two or three finite values")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("action must contain two or three finite values")
    try:
        return RuntimeAction(
            steering=float(values[0]),
            throttle=float(values[1]),
            brake=float(values[2]) if len(values) == 3 else 0.0,
        )
    except ValidationError as error:
        raise ValueError("action is outside the registered bounds") from error


def build_runtime_frame(
    snapshot: dict[str, object],
    action: object,
    info: dict[str, object],
    *,
    previous_velocity: tuple[float, float, float] | None,
    collision_count_before: int,
    completed_checkpoint_ids_before: tuple[str, ...],
) -> RuntimeFrame:
    if snapshot.get("schema_version") != "f1rl_runtime_snapshot_v1":
        raise ValueError("unsupported Gym runtime snapshot schema")
    if not isinstance(info, dict):
        raise TypeError("environment step info must be a mapping")
    dt = float(snapshot["dt_s"])
    velocity = tuple(float(value) for value in snapshot["velocity"])
    previous = previous_velocity or (0.0, 0.0, 0.0)
    acceleration = tuple((value - prior) / dt for value, prior in zip(velocity, previous, strict=True))
    telemetry = info.get("telemetry") or {}
    race = info.get("race") or {}
    collision_ids = list(telemetry.get("obstacle_hit_ids", ()))
    completed = list(dict.fromkeys((
        *completed_checkpoint_ids_before,
        *race.get("completed_checkpoint_ids", ()),
    )))
    # The snapshot may expose only a generic Python event class
    # (FlightTransition). The authoritative physics transition audit has the
    # actual takeoff/landing kind; forward that kind to Replay unchanged.
    physics = info.get("physics") or {}
    transition = physics.get("transition") if isinstance(physics, dict) else None
    events = transition.get("events", ()) if isinstance(transition, dict) else ()
    physical_event_types = list(snapshot.get("physical_event_types", ()))
    if events:
        if not all(isinstance(event, dict) and isinstance(event.get("kind"), str)
                   and 0 < len(event["kind"]) <= 200 for event in events):
            raise ValueError("malformed certified physics transition events")
        physical_event_types = [event["kind"] for event in events]
    outcome = info.get("research_outcome")
    terminal_outcome = None
    if isinstance(outcome, dict):
        terminal_outcome = RuntimeOutcome(
            kind=str(outcome.get("kind", "unknown")),
            valid_sample=outcome.get("valid_sample") is True,
            reason=str(outcome.get("reason", "")),
            scientific_evidence=False,
        )

    return RuntimeFrame(
        schema_version="f1rl_runtime_frame_v1",
        tick=int(snapshot["tick"]),
        time_s=float(snapshot["time_s"]),
        dt_s=dt,
        position=tuple(float(value) for value in snapshot["position"]),
        velocity=velocity,
        acceleration_mps2=acceleration,
        orientation_wxyz=tuple(float(value) for value in snapshot["orientation_wxyz"]),
        heading_rad=float(snapshot["heading_rad"]),
        speed_mps=float(snapshot["speed_mps"]),
        grounded=snapshot["grounded"] is True,
        action=_action_document(action),
        route_progress_m=snapshot["route_progress_m"],
        route_progress_fraction=snapshot["route_progress_fraction"],
        route_highwater_m=float(snapshot["route_highwater_m"]),
        next_checkpoint_index=snapshot["next_checkpoint_index"],
        checkpoint_count=int(snapshot["checkpoint_count"]),
        completed_checkpoint_ids=completed,
        checkpoint_crossings=list(race.get("directed_crossings", ())),
        collision_count=collision_count_before + len(collision_ids),
        collision_ids=collision_ids,
        physical_event_types=physical_event_types,
        outcome=terminal_outcome,
    )
