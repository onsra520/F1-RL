from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.responses import JSONResponse

from src.runtime.catalog import MAX_SEED, RuntimeCatalog
from src.runtime.custom_tracks import (
    ConvertedTrack,
    TrackValidationError,
    convert_custom_track,
)
from src.runtime.protocol import (
    WS_COMMAND_ADAPTER,
    ActionCommand,
    PauseCommand,
    ResetCommand,
    ResumeCommand,
    SpeedCommand,
    StartCommand,
    StopCommand,
)
from src.runtime.session import RuntimeFault, RuntimeSessionManager

logger = logging.getLogger(__name__)
ALLOWED_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str | None = Field(default=None, min_length=1, max_length=32)
    track_id: str | None = Field(default=None, min_length=1, max_length=64)
    seed: int = Field(ge=0, le=MAX_SEED)
    mode: Literal["ppo", "manual"] = "ppo"
    custom_track: dict[str, object] | None = None

    @model_validator(mode="after")
    def controller_matches_mode(self):
        if self.mode == "ppo" and self.model_id is None:
            raise ValueError("PPO mode requires model_id")
        if self.mode == "manual" and self.model_id is not None:
            raise ValueError("manual mode cannot select model_id")
        if self.custom_track is None and self.track_id is None:
            raise ValueError("registered sessions require track_id")
        if self.custom_track is not None and self.track_id is not None:
            raise ValueError("custom sessions derive track_id from the validated draft")
        return self


class ValidateCustomTrackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    track: dict[str, object]


def _manager(request: Request) -> RuntimeSessionManager:
    return request.app.state.sessions


def _runtime_fault(error: ValueError) -> RuntimeFault:
    message = str(error)
    if "model" in message:
        return RuntimeFault("invalid_model", message)
    if "track" in message or "TRAIN" in message:
        return RuntimeFault("invalid_track", message)
    return RuntimeFault("invalid_request", message)


def _converted_track_document(converted: ConvertedTrack) -> dict[str, object]:
    return {
        "schema_version": converted.schema_version,
        "development_only": converted.development_only,
        "track_id": converted.track_id,
        "track_sha256": converted.track_sha256,
        "preview": converted.source_document,
        "road_samples": converted.road_samples,
        "auto_bridge_count": converted.auto_bridge_count,
        "ppo_compatible": converted.ppo_compatible,
        "surface_count": len(converted.factory.track.surfaces),
        "checkpoint_count": len(converted.factory.track.checkpoints),
        "obstacle_count": len(converted.factory.track.obstacles),
        "boost_pad_count": len(converted.factory.track.boost_pads),
        "warnings": converted.warnings,
    }


def create_app(
    *,
    repository_root: str | Path | None = None,
    catalog: RuntimeCatalog | None = None,
) -> FastAPI:
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve(strict=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime_catalog = catalog or RuntimeCatalog(root)
        app.state.catalog = runtime_catalog
        app.state.sessions = RuntimeSessionManager(runtime_catalog)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.sessions.shutdown)

    app = FastAPI(
        title="F1-RL Development Runtime",
        version="1.0.0",
        description="Local, development-only read-only PPO inference and Gym simulation.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["content-type"],
        allow_credentials=False,
    )

    @app.exception_handler(RuntimeFault)
    async def runtime_fault_handler(_request: Request, error: RuntimeFault):
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, error: RequestValidationError):
        details = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "reason": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {
                "code": "invalid_request",
                "message": "request does not match the runtime protocol",
                "details": details,
            }},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, error: Exception):
        logger.exception("runtime request failed", exc_info=error)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "runtime request failed"}},
        )

    @app.get("/api/v1/health")
    async def health(request: Request):
        return {
            "status": "ok",
            "runtime_mode": "development",
            "scientific_admission": "disabled",
            "protocol_version": "f1rl_runtime_v1",
            "source_sha256": request.app.state.catalog.source_sha256,
        }

    @app.get("/api/v1/models")
    async def models(request: Request):
        return {
            "schema_version": "f1rl_runtime_models_v1",
            "models": [
                {"model_id": item.model_id, "sha256": item.sha256, "bytes": item.bytes}
                for item in request.app.state.catalog.models
            ],
        }

    @app.get("/api/v1/tracks")
    async def tracks(request: Request):
        runtime_catalog = request.app.state.catalog
        return {
            "schema_version": "f1rl_runtime_tracks_v1",
            "tracks": [
                {
                    "track_id": item.track_id,
                    "split": item.split,
                    "purpose": item.purpose,
                    "world_sha256": runtime_catalog.track_sha256(item.track_id),
                    "preview": item.preview,
                    "road_samples": item.road_samples,
                }
                for item in await asyncio.to_thread(lambda: runtime_catalog.tracks)
            ],
        }

    @app.post("/api/v1/tracks/validate")
    async def validate_custom_track(payload: ValidateCustomTrackRequest, request: Request):
        try:
            converted = await asyncio.to_thread(
                convert_custom_track,
                payload.track,
                request.app.state.catalog,
            )
        except TrackValidationError as error:
            raise RuntimeFault(error.code, str(error), status_code=422) from error
        return _converted_track_document(converted)

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session(payload: CreateSessionRequest, request: Request):
        try:
            converted = None
            if payload.custom_track is not None:
                converted = await asyncio.to_thread(
                    convert_custom_track,
                    payload.custom_track,
                    request.app.state.catalog,
                )
                if payload.mode == "ppo" and not converted.ppo_compatible:
                    raise RuntimeFault("incompatible_model", "custom track observation/action spaces are incompatible with frozen PPO models", status_code=422)
            track_id = converted.track_id if converted is not None else payload.track_id
            assert track_id is not None
            session = await asyncio.to_thread(
                _manager(request).create,
                model_id=payload.model_id,
                track_id=track_id,
                seed=payload.seed,
                mode=payload.mode,
                custom_factory=None if converted is None else converted.factory,
                custom_track_sha256=None if converted is None else converted.track_sha256,
            )
        except RuntimeFault:
            raise
        except TrackValidationError as error:
            raise RuntimeFault(error.code, str(error), status_code=422) from error
        except ValueError as error:
            raise _runtime_fault(error) from error
        status = session.status_document()
        if converted is not None:
            status["track_preview"] = _converted_track_document(converted)
        return status

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        return _manager(request).get(session_id).status_document()

    @app.post("/api/v1/sessions/{session_id}/run", status_code=202)
    async def run_session(session_id: str, request: Request):
        session = _manager(request).get(session_id)
        session.start()
        return session.status_document()

    @app.get("/api/v1/sessions/{session_id}/replay")
    async def get_replay(session_id: str, request: Request):
        session = _manager(request).get(session_id)
        return await asyncio.to_thread(session.replay)

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str, request: Request):
        await asyncio.to_thread(_manager(request).delete, session_id)
        return Response(status_code=204)

    @app.websocket("/api/v1/sessions/{session_id}/ws")
    async def runtime_socket(websocket: WebSocket, session_id: str):
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            await websocket.close(code=1008, reason="origin is not allowed")
            return
        try:
            session = _manager(websocket).get(session_id)
            session.claim_socket()
        except RuntimeFault as error:
            await websocket.close(code=4404 if error.status_code == 404 else 4409, reason=str(error))
            return
        await websocket.accept()
        await websocket.send_json({
            "protocol_version": "f1rl_runtime_ws_v1",
            "type": "ready",
            "session_id": session_id,
            "data": session.status_document(),
        })
        stopping = asyncio.Event()

        async def send_events():
            while not stopping.is_set():
                event = await asyncio.to_thread(session.next_event, 0.25)
                if event is not None:
                    await websocket.send_json(event)

        sender = asyncio.create_task(send_events())
        try:
            while True:
                try:
                    raw = await websocket.receive_json()
                    command = WS_COMMAND_ADAPTER.validate_python(raw)
                except ValidationError as error:
                    await websocket.send_json({
                        "protocol_version": "f1rl_runtime_ws_v1",
                        "type": "error",
                        "session_id": session_id,
                        "data": {"code": "invalid_command", "message": "message does not match the runtime protocol", "details": error.errors()},
                    })
                    continue
                except ValueError:
                    await websocket.send_json({
                        "protocol_version": "f1rl_runtime_ws_v1",
                        "type": "error",
                        "session_id": session_id,
                        "data": {"code": "invalid_command", "message": "message must be a JSON object"},
                    })
                    continue

                try:
                    if isinstance(command, StartCommand):
                        session.start(paced=True)
                    elif isinstance(command, PauseCommand):
                        session.pause()
                    elif isinstance(command, ResumeCommand):
                        session.resume()
                    elif isinstance(command, ResetCommand):
                        session.reset()
                    elif isinstance(command, StopCommand):
                        session.cancel()
                    elif isinstance(command, SpeedCommand):
                        session.set_speed(command.multiplier)
                    elif isinstance(command, ActionCommand):
                        session.set_action(command.steering, command.throttle, command.brake)
                    await websocket.send_json({
                        "protocol_version": "f1rl_runtime_ws_v1",
                        "type": "ack",
                        "session_id": session_id,
                        "data": {"command": command.type, "status": session.status},
                    })
                except RuntimeFault as error:
                    await websocket.send_json({
                        "protocol_version": "f1rl_runtime_ws_v1",
                        "type": "error",
                        "session_id": session_id,
                        "data": {"code": error.code, "message": str(error)},
                    })
        except WebSocketDisconnect:
            pass
        finally:
            stopping.set()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            session.release_socket()
            try:
                await asyncio.to_thread(_manager(websocket).delete, session_id)
            except RuntimeFault:
                pass

    return app


app = create_app()
