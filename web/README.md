# F1-RL Racing Sandbox — Frontend V1

Game-like track builder on `frontend/racing-sandbox`. **Browser is for editing/visual preview; Python is authoritative for physics/evaluation.**

## Run

```powershell
cd web
npm ci
npm run dev
# checks:
npm run build
npm test
```

## Features

- **Sketch:** freehand or point-based spline editing, draggable handles, sample route, clear.
- **Build:** procedural road 3D, Orbit Camera, drag-and-drop bridge / ramp / barrier / cone / booster / checkpoint, selection and rotation/remove.
- **Drive:** elevated rear chase camera, manual WASD/arrows/Space **visual-only arcade preview**, or read-only imported TRAIN/DEVELOPMENT replay.
- **Files:** strict editor JSON draft schema, local save, import/export, undo/redo.

## Safety and contracts

- Map schema: `f1rl_racing_sandbox_draft_v1`; points are X/Z, props use normalized centerline `t` and lateral offset. **Not** an accepted Python `TrackWorld` or `TrackAsset`.
- Road height for bridge/ramp is currently **visual only**; Python must validate physical surfaces, collision/landing and booster physics before a map is run authoritatively.
- Browser manual drive makes no physics/scoring/finish claims; does not produce scientific evidence.
- Replay schema: `f1rl_racing_sandbox_replay_v1`, example:

```json
{
  "schema": "f1rl_racing_sandbox_replay_v1",
  "label": "TRAIN C-seed0",
  "source": { "split": "TRAIN", "model_id": "C-seed0" },
  "frames": [
    { "time_s": 0, "position": [0, 0, 0], "heading_rad": 0, "speed_mps": 0 },
    { "time_s": 0.1, "position": [0, 0, 0.5], "heading_rad": 0, "speed_mps": 5 }
  ]
}
```

- PPO replay files are imported locally, do not verify evidence authenticity, and are **not** direct WP3 evaluation JSONL. HELD-OUT is disallowed. Live FastAPI/WebSocket, map-to-TrackWorld compiler, and scientifically validated manual driving remain future work.
