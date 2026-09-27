# `f1rl.api`

## Purpose

FastAPI adapter exposing F1-RL runtime functionality to external clients.

## Planned files

- `app.py` — FastAPI application factory and lifecycle wiring.
- `websocket.py` — realtime simulation state and interactive event transport.
- `routes/` — HTTP endpoints grouped by feature.

## Boundary

The API translates transport data to/from domain/runtime calls. Simulation rules must not live here.
