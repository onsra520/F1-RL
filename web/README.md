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

## V1.1 layout and elevated-road builder

- Only top navigation + full-height canvas; the former workspace title/DRAFT subtitle and large Build Kit cards are removed.
- The Build kit is a compact six-slot Minecraft-like hotbar over the scene. Hover/focus reveals a tooltip, drag items to the scene or select a slot with keys `1–6`, and use Escape to clear selection.
- Road visuals have a 1.5m underbody, side walls, end caps, side rails on raised spans, and support pillars. These are **render meshes, not physics collision meshes**.
- Proper non-adjacent **centerline intersections** are detected in the smoothed sketch. The later traversal is raised by a minimum 6m visual separation using a smooth approach/exit profile, producing a simple automatic overpass.
- Retracing collinear segments, nearly-overlapping roads without a centerline crossing, tight endpoint intersections, and arbitrary multilayer road geometry are **not guaranteed** to be bridged. All auto-overpass spans must be validated by a future Python track compiler before authoritative simulation.
- Bridge/ramp tools deform the road profile and show small selectable handles beside the lane; the former cross-road beam and floating ramp slab are gone.
- The chassis is placed at the start segment with enough inward offset to keep the rear wheels on the road, facing the first tangent.
