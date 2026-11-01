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

- `npm test`: 8 files passed, 42 tests passed, including reset-frame ordering and stopped/failed terminal status coverage.
- `npm run typecheck`: PASS.
- `npm run build`: PASS. Vite emitted its existing large JavaScript chunk advisory (>500 kB).
- Python tests were not rerun; no Python source or backend contract changed.
- `agent-browser errors`: empty after the final clean PPO run and the targeted Pause/Resume/Reset/Stop run.

## Remaining limits

- Browser evidence uses managed Chromium through agent-browser; Edge and Brave were not separately exercised.
- The video recorder failed, so motion evidence is a sequence of real-browser screenshots plus captured WebSocket timing and deterministic interpolation regression tests.
