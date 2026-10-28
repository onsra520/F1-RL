# `scripts`

## Purpose

Thin executable entry points for common development and experiment workflows.

## Planned files

- `train.py` — starts a configured training run.
- `evaluate.py` — evaluates an exported/checkpoint policy.
- `serve.py` — starts the FastAPI demo/runtime server.

Business logic belongs in `src/`; scripts should only parse inputs, call project APIs, and report status.
