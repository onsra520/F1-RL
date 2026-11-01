# Build → Drive handoff browser E2E

Run date: 2026-11-02
Branch: `feature/runtime-simulator-bridge`
Baseline under test: `c6503a013eafe13902420a1efd01caccd1b1a037`
Browser: agent-browser 0.41.0 with Chrome 156.0.8078.80
Frontend: `http://127.0.0.1:5173/`
Python runtime: `http://127.0.0.1:8000/`

## Scenarios and outcomes

- **Build → Drive → Run:** drew a custom U-shaped route unlike Straight Sprint, added a ramp, entered Drive without pressing a separate validate control, and saw `PYTHON VALIDATED CUSTOM TRACK` with 79 road surfaces. Python returned custom track `sandbox-44d64d81206e9120`, SHA-256 `44d64d81206e9120e545fd8957177f4b209632b76d12a4c32ec6d69ce572d751`, and sampled road height up to 2.7969 m. PPO created a real Python session on this custom map.
- **Back to Build → edit → Drive:** added a checkpoint and entered Drive again. The map revalidated to 80 road surfaces and 1 checkpoint with a new backend identity `sandbox-986ab09dd0003682`, SHA-256 `986ab09dd0003682a923a6d600e6948a4425038fac181dd141f49dc1ca6dadc4`. Both the PPO session and Manual session used that same ID/hash. Manual advanced 43 simulation frames and moved from approximately `[-97.3, 0.5, 45.0]` to `[-96.1, 0.5, 45.0]`.
- **Unsupported map:** Python rejected an earlier map draft with `unsupported_surface_intersection`. The UI stayed on Custom, showed the validation error, and kept Run disabled; no TRAIN track was substituted. The supported U-shaped route was then validated and run.
- **Direct Sketch → Drive:** selected the Sample route from Sketch and entered Drive directly. Python returned HTTP 422 `unsupported_surface_intersection`; the UI selected Custom, showed the backend error, and disabled Run. See `sketch-drive-handoff.har` and `sketch-drive-unsupported-blocked.png`.
- **TRAIN selection:** explicitly selected `Use registered TRAIN track`, then started PPO. HAR shows `POST /api/v1/sessions` returned 201 with `track_id: straight-sprint-v1`, followed by replay retrieval. The stopped session returned 404 `session_not_found` on a subsequent status request. See `ppo-explicit-train.har`.
- **Final trace-header smoke:** after adding `X-Trace-ID` to custom session creation, imported the valid custom map into an isolated frontend on port 5174 using the updated Python backend on port 8001. Drive automatically validated 80 road surfaces and 1 checkpoint; PPO created a `custom-development` session with the same custom track ID/hash and an `x-trace-id` request header. The session replay request returned 200, and the stopped session returned 404 `session_not_found`. See `custom-session-trace-handoff.har`.
- **Replay:** exported the Manual episode from the browser UI and inspected the downloaded JSON. It is replay v2 with the Manual episode's matching custom track ID/hash, ramp and checkpoint props, 309 road samples (maximum sampled height 2.7969 m), and 44 frames. The replay track is preserved in `manual-edited-map-replay.json`.
- **Visual evidence:** `custom-ramp-validated.png` shows the validated custom route with ramp; `custom-ramp-checkpoint-validated.png` shows the edited and revalidated map; `custom-replay-track.png` shows replay using the episode track.

## Observations and limits

- Browser page errors: none reported by `agent-browser errors`.
- Runtime requests succeeded against the Python backend; stopped session cleanup returned 404 as expected.
- The browser console showed Vite/React development messages. `THREE.WebGLRenderer: Context Lost.` appeared as a console log during Vite hot reload while source files were changing; the rendered page recovered and screenshots were captured. This was not reported as a page exception.
- Custom PPO integration is proven by the backend session, track identity, and returned frames. On this custom route the frozen policy emitted throttle=1 and brake=1 and stayed near spawn during the short observed run; it was stopped manually. No custom-route PPO finish is claimed.
- Browser coverage here used Chrome via agent-browser, not Microsoft Edge or Brave.

## Captured request evidence

- `ppo-custom-session.har`: initial custom PPO session creation.
- `ppo-edited-map.har`: custom PPO session after map edit and revalidation.
- `manual-edited-map.har`: Manual session after map edit and revalidation.
- `ppo-explicit-train.har`: explicitly selected registered TRAIN PPO session.
- `sketch-drive-handoff.har`: direct Sketch → Drive validation request and explicit unsupported-map response.
- `custom-session-trace-handoff.har`: final custom validation/session/replay flow with trace header against the updated local Python backend.
- `manual-edited-map-replay.json`: browser-exported replay with episode track geometry.

## Automated verification

All commands ran from this branch worktree on 2026-11-02:

- `cd web && npm test` — **PASS**, 8 files / 50 tests.
- `cd web && npm run typecheck` — **PASS**.
- `cd web && npm run build` — **PASS**, with the existing Vite advisory that the main minified chunk is larger than 500 kB.
- `$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'; .\.venv\Scripts\python.exe -m pytest -q tests/runtime/test_custom_tracks.py tests/runtime/test_api.py` — **PASS**, 16 tests.
