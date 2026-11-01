# Live PPO Rendering Stutter — Browser E2E

## Environment

- Branch: `feature/runtime-simulator-bridge`
- Baseline: `6909e34a7953a29ce0c529ccb6140acce26feb44`
- Browser: `agent-browser 0.41.0`, managed Chromium `156.0.0.0` on Windows
- Frontend: `http://127.0.0.1:5173`; Python runtime: `http://127.0.0.1:8000`
- Scenario: A-seed2 / Straight Sprint / seed 412 / default 1× simulation speed

## Diagnosis

The runtime emitted backend frames at fixed `dt_s=0.1`. The baseline browser received them at median 110.7 ms intervals. `App` assigned each frame directly to React state, `activePose` read `latest_frame`, and `CarModel` applied that pose only when React reconciled. The car therefore stepped about 1.04 m per delivered frame. `ChaseCamera` eased its position, but its `lookAt` target used the raw pose and changed on the same WebSocket cadence.

The captured physics poses did not show a separate discontinuity: baseline median per-frame displacement was 1.035879 m, consistent with the backend's 0.1 s timestep and observed speed. After the change, the same A-seed2 run retained the same 113-step successful result and median backend pose step. Physics and PPO were not changed.

## Change

`LivePoseInterpolator` holds only the previous and latest authoritative frames, interpolates position/speed and shortest-path heading in the R3F `useFrame` loop, and is shared by the car and chase camera. Duplicate and older frames are ignored. Interpolation stops at the latest received pose; gaps over 250 ms snap to the new authoritative frame instead of inventing motion. Pause holds the current rendered pose, Resume interpolates from that pose, terminal states freeze, and reset clears the buffer. Replay continues to use its existing `replayAt` timeline.

## Browser results

| Check | Result | Evidence |
|---|---|---|
| Baseline live PPO reproduced | PASS | Baseline capture and `frame-metrics.json`: 114 frames, `dt_s=0.1`, median arrival 110.7 ms, median pose step 1.035879 m. |
| A-seed2 / Straight / seed 412 after fix | PASS | `live-ppo-final-midrun.png`, `live-ppo-final-terminal.png`; successful finish at 113 steps, matching baseline. |
| Final clean-page A-seed2 / Straight / seed 412 run | PASS | `final-clean-midrun.png`, `final-clean-terminal.png`; successful finish at 113/500 steps after reset/stop lifecycle guard changes. |
| Car and camera during live run | PASS | `after-live-midrun-1.png`, `after-live-midrun-2.png`; captured at different progress points while the real backend session advanced. |
| Pause | PASS | `paused.png` and `paused-after-wait.png` one second apart: same tick (40), pose and camera stayed stationary. |
| Resume / Reset / Stop | PASS | Resume advanced after pause. Reset returned to the start; `final-reset-after.png` shows the restarted session at 3/500 steps near the initial pose. Stop returned to the idle run screen. |
| Manual Python drive | PASS | Throttle input moved the vehicle; release returned throttle to zero. Stop cancelled the session. See `manual-live.png`. |
| Replay playback / seek | PASS | Runtime replay playback advanced and slider seeking changed the selected time; see `replay-ready.png` and `replay-seek.png`. |
| Console and network | PASS | `final-clean-browser-errors.log` and `final-clean-browser-console.log` are empty after a clean reload and final PPO run. `browser-network.log` records catalog, session creation (201), and replay retrieval (200), with session IDs redacted. The earlier HMR session logged renderer context loss while Vite replaced the scene; it did not recur after reload. |

The managed browser's video encoder fell behind and did not produce a video file. Screenshots and the measured frame summary are retained. No JavaScript page errors were recorded.

## Verification

- `npm test`: 8 files passed, 47 tests passed, including reset-frame ordering and terminal status coverage.
- `npm run typecheck`: PASS.
- `npm run build`: PASS. Vite emitted its existing large JavaScript chunk advisory (>500 kB).
- Python tests were not rerun; no Python source or backend contract changed.
- `agent-browser errors`: empty after the final clean PPO run and the targeted Pause/Resume/Reset/Stop run.

## Final acceptance retest — 2026-11-02

- Browser: headed Microsoft Edge 156 (`Edg/156`) launched from the installed Edge executable by `agent-browser`; backend `http://127.0.0.1:8000`.
- Final code regression: `isRuntimePoseTerminal('resetting')` is false. The live backend emits a transient `resetting` status before `running`; treating it as terminal would close the pose interpolator during reset. `npm test` now passes 8 files / 47 tests; `npm run typecheck` and `npm run build` pass, with the existing large-chunk advisory.
- PPO A-seed2 / straight-sprint-v1 / seed 412: PASS. Edge received 114 frames (ticks 0–113), both checkpoints (`checkpoint-0`, `checkpoint-1`) completed, and the backend reported `successful_finish` / `finished` at tick 113. The final backend pose was `[0.2481682089, 0.5016385523, 112.2257833312]`, heading `-0.1581212521 rad`; the finished UI displayed the matching rounded position `0.2, 0.5, 112.2` and heading `-9.1°`. The terminal interpolator regression asserts the exact authoritative endpoint. See `edge-final-midrun.png` and `edge-final-terminal.png`.
- Pause / Resume / Reset / Stop: PASS. Pause held tick 43 and the same displayed pose over a one-second wait; Resume advanced to tick 51. Reset produced `resetting` then `running`, restarted near the initial pose (tick 6–7), and continued receiving new frames; Stop cancelled the run and returned the UI to idle.
- Manual Drive: PASS. Python physics advanced the car from about `[0.2, 0.5, 4.0]` to `[0.6, 0.5, 8.8]` under throttle. Stop cancelled and cleaned up the session.
- Replay playback / seek: PASS. Playback advanced from 0 to 6.64 seconds; timeline seeking selected 7.56 seconds. The exported file was imported back through the Replay input and the UI showed `Replay loaded`.
- Edge replay download: PASS. `Downloads/f1rl-runtime-replay.json` was inspected as an actual file: 159,673 bytes, schema `f1rl_racing_sandbox_replay_v2`, 114 frames (ticks 0–113), A-seed2 / straight-sprint-v1 / seed 412, SHA-256 `8bff2d4ae7b4a83aebf3b4fb7a36267bf3097e641a9564dccd13d804aec42dee`.
- Console / WebSocket / cleanup: PASS. No browser page errors or WebSocket errors. All six observed sockets closed with code 1000; subsequent session GETs returned 404 for all six IDs.

## Remaining limits

- Managed Chromium and Microsoft Edge 156 were exercised; Brave was not separately exercised.
- The video recorder failed, so motion evidence is a sequence of real-browser screenshots plus captured WebSocket timing and deterministic interpolation regression tests.
