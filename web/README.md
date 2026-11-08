# F1-RL Racing Sandbox

Browser editor and runtime client for the canonical Python simulation under `f1rl/`.

## Run

From the repository root, start `run.py` after installing the Python runtime dependencies. It serves the API at `http://127.0.0.1:8000` and the frontend at `http://127.0.0.1:5173`.

For frontend-only development:

```powershell
cd web
npm ci
npm run dev
npm test
npm run build
```

## Features

- Sketch and edit track drafts, move road-attached checkpoints and boosters, and export/import the draft schema.
- Preview roads, props, elevation, and the vehicle in the browser scene.
- Drive through the Python runtime with manual control, Auto Drive, frozen PPO inference, or development Hybrid mode.
- Inspect runtime outcomes and replay recorded frames.

## Runtime and data contracts

- Track drafts use `f1rl_racing_sandbox_draft_v1`. The Python runtime validates and converts supported drafts before simulation; browser meshes remain presentation only.
- The API and WebSocket session adapter call the canonical `f1rl/` environment. There is no second physics engine in the frontend.
- Frozen PPO archives are used for inference. Development results do not constitute scientific evidence or grant training/evaluation admission.
- Replay files use `f1rl_racing_sandbox_replay_v1`. Imported replays are display inputs and do not verify evidence authenticity.
