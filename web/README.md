# `web`

## Purpose

React + TypeScript + Three.js client for the interactive F1-RL sandbox.

## Responsibilities

- Track and scenario editor.
- Three.js scene rendering and camera controls.
- Runtime obstacle/zone placement.
- Vehicle, rangefinder, checkpoint, timer, score, and metric visualization.
- Human-driving input when enabled.
- FastAPI HTTP/WebSocket client integration.

## Boundary

The browser is not the authoritative physics or RL environment. It renders snapshots and sends user intent to the Python core.
