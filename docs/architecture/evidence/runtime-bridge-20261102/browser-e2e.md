# Runtime Bridge Browser E2E — 2026-11-02

## Run context

- Repository branch: `feature/runtime-simulator-bridge`
- Tested HEAD: `3d4588a0b4f7235afa91e6ef6d10bc651a5f2fb1` (same as fetched `origin/feature/runtime-simulator-bridge` at start of this run)
- Browser: `agent-browser 0.41.0`, managed Chromium `156.0.0.0`, Windows user agent
- Frontend: `http://127.0.0.1:5173`
- Python runtime: `http://127.0.0.1:8000`, started from this checkout with the existing WP3 evidence path and `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
- The frontend was already listening. No product files were changed during this E2E run.

## Results

| Flow | Result | Observed result |
|---|---|---|
| Racing Sandbox startup and render | PASS | UI rendered `Racing Sandbox · F1-RL`; no empty page. |
| PPO: `straight-sprint-v1`, `slalom-v1`, `right-turn-v1`, `ramp-v1` | PASS | A-seed0 / seed 412 sessions streamed telemetry and reached the 500-step limit without a runtime crash. Outcomes displayed valid non-finish; no finish was forced. |
| B-seed1 / Slalom / seed 412 | PASS | Run terminated at 85/500 and UI displayed `Driving Failure · Track Boundary` with the wheel-left-road message. The UI maps the backend `track_boundary_exit:` reason to this driving-failure outcome. |
| Manual input | PASS | Held throttle changed telemetry (speed and position); releasing input returned throttle to zero. |
| Pause / Resume | PASS | Paused and resumed a live Manual session; telemetry continued. |
| Reset / Stop | PASS | Reset restarted the live episode; Stop yielded `CANCELLED / STOPPED` and restored the start control. |
| Replay playback / seeking | PASS | Playback advanced the replay timeline; pause and slider seeking changed the selected time. |
| Replay export serialization | PASS | The UI Export Replay action produced valid `f1rl_racing_sandbox_replay_v2` JSON with 501 frames for `ramp-v1`, seed 412. Artifact: `ramp-replay.json`. |
| Replay native browser download | PARTIAL / BLOCKED | Managed Chromium marked app downloads `Failed - Download error`. A generic Blob download probe also failed; Chrome's download page is captured in `18-download-diagnostics.png`. Blob contents were captured with browser-side test instrumentation to verify serialization. A normal file download through the browser was not verified. |
| Replay import / playback / seeking | PASS | Imported the exported replay JSON; UI showed the development-evidence toast, then playback and seeking worked. |
| Backend disconnect / reconnect | PASS | Terminated the exact runtime listener to simulate a backend outage. UI reported WebSocket disconnection and made Run available again. After backend restart, a new session POST returned 201, WebSocket was accepted, telemetry advanced, and Stop cancelled the session. |
| Session cleanup | PASS | After stopping the reconnected run, `GET /api/v1/sessions/e1f769b7-fda0-4a61-bd41-225dba187175/replay` returned 404. A completed session also returned 404 after its socket closed. |
| Browser console / app errors | PASS | `agent-browser errors` was empty. Console only contained Vite connection messages and the standard React DevTools info message. |
| Backend/WebSocket requests | PASS | Backend log records accepted WebSockets and session traffic. Browser network log records session creation (201) and replay requests (200); disconnect/reconnect created a fresh session. |

## Evidence

Screenshots `01`–`18`, browser console/errors/network logs, backend log, and exported replay JSON are stored in this directory. Key captures:

- `01-rendered.png` — initial Racing Sandbox render
- `03-ppo-straight-terminal.png`, `11-ppo-slalom-a.png`, `13-ppo-right-turn-terminal.png`, `14-ppo-ramp-terminal.png` — PPO runs
- `12-b-seed1-slalom-412.png` — boundary driving failure
- `06-manual-throttle.png`, `07-manual-reset.png`, `08-manual-stopped.png`, `17-reconnected-session-stopped.png` — Manual controls and reconnect
- `15-replay-imported.png`, `16-replay-import-seek.png` — replay import/playback/seek
- `18-download-diagnostics.png` — managed Chromium download errors
- `browser-console.log`, `browser-errors.log`, `browser-network.log`, `backend.log`, `ramp-replay.json`

## Verification boundary

- No application defect was established in these flows, so no source or regression-test changes were made. Frontend/Python suites were not rerun because this commit contains E2E evidence only.
- Browser evidence is from agent-browser's managed Chromium; Edge and Brave were not separately exercised.
- Acceptance is not fully met while native replay-file download remains blocked by the browser environment. Keep Stage 2 open until download works in a supported browser environment or an equivalent end-user export path is verified.
- The raw WebSocket failure frame was not archived. The B-seed1 result is evidenced by the UI screenshot and the app's existing mapping of `track_boundary_exit:` to `Driving Failure · Track Boundary`.
