# F1-RL

**F1-RL** is an interactive reinforcement learning sandbox for autonomous racing and obstacle avoidance.

The project is designed around a Python-first RL core: the simulation, observations, rewards, training, and evaluation live in `src/`; FastAPI exposes runtime state; React + Three.js provide the interactive editor and visualization layer.

## Intended stack

- Python
- Gymnasium
- Stable-Baselines3 / PyTorch
- FastAPI + WebSocket
- React + TypeScript + Three.js

## Repository layout

- `src/` — project core and domain logic.
- `web/` — browser UI, track editor, and Three.js visualization.
- `assets/` — versioned tracks and other source assets.
- `artifacts/` — generated checkpoints, exported policies, and evaluation outputs.
- `configs/` — training and runtime configuration.
- `scripts/` — thin executable entry points.
- `tests/` — unit and integration tests.
- `docs/` — architecture, experiment, and project documentation.

## Architecture rule

Dependencies point inward. The simulation sandbox must not depend on FastAPI, Three.js, or a specific RL algorithm. Web and API layers are adapters around the F1-RL core.

See the `README.md` inside each directory for its responsibilities and planned files.
