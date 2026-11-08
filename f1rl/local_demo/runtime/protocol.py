from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal["f1rl_runtime_ws_v1"]


class StartCommand(_Command):
    type: Literal["start"]


class PauseCommand(_Command):
    type: Literal["pause"]


class ResumeCommand(_Command):
    type: Literal["resume"]


class ResetCommand(_Command):
    type: Literal["reset"]


class StopCommand(_Command):
    type: Literal["stop"]


class SpeedCommand(_Command):
    type: Literal["speed"]
    multiplier: float = Field(ge=0.25, le=16.0, allow_inf_nan=False)


class ActionCommand(_Command):
    type: Literal["action"]
    steering: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    throttle: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    brake: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


WsCommand = Annotated[
    StartCommand | PauseCommand | ResumeCommand | ResetCommand | StopCommand | SpeedCommand | ActionCommand,
    Field(discriminator="type"),
]
WS_COMMAND_ADAPTER = TypeAdapter(WsCommand)


class WsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["f1rl_runtime_ws_v1"] = "f1rl_runtime_ws_v1"
    type: str
    session_id: str
    data: dict[str, object]
