# Python Runtime and Racing Sandbox Integration Report

**Date:** 2026-10-30
**Branch:** `feature/runtime-simulator-bridge`
**Base:** `origin/main` at `e87d42b9aab93da5450445451240e311f59a5cdf`
**Implementation HEAD before this report:** `9489454a8f080db03ddd51df899553697a6d68e`

## Summary

The Racing Sandbox now uses a local Python runtime as the authoritative simulation source. The runtime loads only the allowlisted frozen PPO checkpoints, routes registered sessions to TRAIN tracks, and steps the existing Gymnasium environment. Manual driving, live telemetry, replay import/export, and validated development-only custom tracks are available through the same session interface.

This is a local development integration. It does not change the Physics or PPO algorithms, model checkpoints, accepted scientific digest facade, or held-out evaluation data. The direct executions below include one valid step-limited episode and one physics failure; they are integration evidence, not a claim that the policies passed evaluation.

## Architecture and data flow

See the [Archify architecture diagram](RUNTIME-SIMULATOR-INTEGRATION-ARCHITECTURE.html) and its [editable source](runtime-simulator-integration.architecture.json).

The runtime path is:

1. The React/Three.js Racing Sandbox requests catalogs and creates a session over HTTP.
2. The FastAPI runtime validates the request. The catalog checks model identity and checkpoint SHA-256 before loading a frozen model, and resolves registered tracks through the existing TRAIN registry.
3. The session selects exactly one action source: deterministic PPO inference or the manual controller.
4. Actions step the existing Gymnasium environment at its fixed timestep. The runtime streams bounded frame/status events over WebSocket and can return a versioned replay.
5. A custom editor draft passes through a strict development-only validator/converter and a development TrackFactory; it never enters the registered TRAIN catalog.

The backend binds to `127.0.0.1:8000`. CORS allows only `http://127.0.0.1:5173` and `http://localhost:5173`. Sessions are bounded to 2 active and 8 retained, expire after 900 seconds, cap at 5,000 steps, and use a bounded event queue.

## Runtime API and wire protocol

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Runtime and catalog health |
| GET | `/api/v1/models` | Allowlisted model metadata and digests |
| GET | `/api/v1/tracks` | Registered TRAIN tracks |
| POST | `/api/v1/tracks/validate` | Validate and preview a custom draft |
| POST | `/api/v1/sessions` | Create a PPO or manual session |
| GET | `/api/v1/sessions/{session_id}` | Session status and latest state |
| POST | `/api/v1/sessions/{session_id}/run` | Start or resume execution |
| GET | `/api/v1/sessions/{session_id}/replay` | Retrieve a completed replay |
| DELETE | `/api/v1/sessions/{session_id}` | Close and release a session |
| WebSocket | `/api/v1/sessions/{session_id}/ws` | Controls and live events |

WebSocket messages use `f1rl_runtime_ws_v1`. Commands are `start`, `pause`, `resume`, `reset`, `stop`, `speed`, and `action`. Events include `ready`, `status`, `frame`, `progress`, `checkpoint`, `collision`, `terminal`, `error`, and `ack`. Replays use `f1rl_racing_sandbox_replay_v2`; the frontend still imports the prior replay format.

## Frozen model catalog

The runtime allowlist contains the 12 A/B/C/D × seed 0/1/2 checkpoints. These are the expected SHA-256 digests; the catalog verifies the local checkpoint content before loading it.

| Model | SHA-256 |
|---|---|
| A-seed0 | `8cfcde6d4c88bbcc01dbf9b67dee3829c991474b0a8883a4479370f665c01c79` |
| A-seed1 | `4334c44b3370c4f0adaa345ca280d190850fda9599736cc561ed6cbb71425831` |
| A-seed2 | `e2a0eb0deef34fad7bf2bf2322c0c13854869372263a4652d4c6f1c422d43adc` |
| B-seed0 | `d76719fb04c5eee1a03e1b0885989f44faad6543d3a1fea9aa24fe6ac55867d6` |
| B-seed1 | `695f1c649235775eb1446ac0718b69b5951a1242b389221a5318db540a716021` |
| B-seed2 | `18f17c3f07d3e62b43364f3f4c6572b3e08b3307a4ae47137b80342188ad0dae` |
| C-seed0 | `15373c0237e883fa2f980dfecbda4809aabc283c0d77a8b4d783e2ab5bf11492` |
| C-seed1 | `1089a4ddb826d8a52a3960f2415fc920ee1b87cedda6d062d661e4f3244440e7` |
| C-seed2 | `156ee0f2194f3aedb36cce3917cc4121994b53fa0f0c942c2de548b9e2ea32b7` |
| D-seed0 | `318f76813b69f8457dc169b6b3a89dd5f3aa2f2f765170075e0e48ab9f0e0109` |
| D-seed1 | `70892fdb0e93ffdc67c36e18e92c1ba70f7ce9f6d6dc41916370781f6f18fc0c` |
| D-seed2 | `212cca532ed1411246687821002ef55f62329ddcab18af2ff08f8e5b9b154cdc` |

## Track routing and custom drafts

Registered runtime routes are limited to the TRAIN tracks: `straight-sprint-v1`, `slalom-v1`, `right-turn-v1`, and `ramp-v1`. The runtime rejects unknown tracks and does not provide a path to HELD-OUT data.

The custom converter accepts only `f1rl_racing_sandbox_draft_v1` with the exact `schema`, `id`, `name`, `width`, `points`, and `props` fields. It validates finite bounded geometry and resource limits (1 MB input, 2–512 control points, at most 200 props, 1,000 m route length, and at most 512 generated surfaces). It samples the editor-compatible centripetal spline into the existing track representation and maps checkpoint, bridge, ramp, barrier, cone, and booster props to existing simulation objects. A custom PPO session is enabled only when the generated observation/action contract matches the selected model.

Every custom result is marked development-only and carries the source draft digest. It cannot be registered as a TRAIN track. Route crossings that cannot be represented safely by the supported surface model, including stacked-deck cases, return `unsupported_intersection` rather than being approximated.

## Delivery stages

| Stage | Result | Commit |
|---|---|---|
| 1 | Python runtime, model/track catalog, bounded sessions, HTTP/WebSocket protocol, replay and custom validation | `b99c96a96c30acce2be447c401a11a8ab95a9d8a` |
| 2 | Racing Sandbox connection, PPO/manual controls, telemetry, and replay/custom-track UI | `88b214be71ea3654eb649204dc66d1f4c1d925ae` |
| 3 | Keyboard action mapping and manual-control tests | `0396390dd824fc38d4d34551733ba65c3c0f95ef` |
| 4 | Custom track conversion and runtime integration tests | `2de705888d301aa1b977265570ac5c8f1a68e247` |
| Review follow-up | Retry catalog connection, clean up failed sessions, and display position/heading/velocity/acceleration telemetry | `9489454a8f080db03ddd51df899553697a6d68e` |

## Execution evidence

Direct frozen-model runs used seed 412 and the existing read-only evaluation path:

| Model / TRAIN track | Result | Trace SHA-256 |
|---|---|---|
| A-seed0 / straight-sprint-v1 | 500 steps, 501 frames, valid sample; terminated at `step_limit`, did **not** successfully finish | `a3236541feb4fa3571d5ac7011249615bcf8cd0b6c926a991b0f7613409f3404` |
| B-seed1 / slalom-v1 | Failed at step 82 with `SuspensionResolutionError: spatial body/contact sweep outside: invalid endpoint body or wheel support`; 83 frames, invalid sample | `e015d147e2806f72eca9fb3bc6ba7c7c0ce5a4e4a18f6672d01b0f06cacd651` |

Automated runtime coverage also exercises a repeated-seed trace digest, session cleanup on disconnect, live PPO WebSocket frames and pause/reset/resume/stop, manual actions on the actual Gym environment, custom manual frames and development replay provenance, custom PPO WebSocket frames, and invalid/unsupported draft rejection.

## Verification

- Focused backend regression: **70 passed in 74.71 seconds** across `tests/runtime`, `tests/learning/evaluation/test_read_only_rollout.py`, and `tests/research/test_wp3_inference_adapter.py`.
- Frontend: **21 tests passed**; `npm run typecheck` and `npm run build` passed.
- Scoped Ruff: `uvx ruff check src/runtime tests/runtime` passed.
- Python compile check and `uv pip check` passed.
- `npm audit --omit=dev`: 0 production advisories. Full development dependency audit reports 6 advisories (3 moderate, 1 high, 2 critical) in the Vitest/tinypool development dependency chain; upgrading requires a breaking Vitest major version.
- Archify diagram validation: 9/9 checks passed; showcase composition reported no warnings or errors. HTML artifact: 806,517 bytes, SHA-256 `4d51b76e75adc70b17ee0626e986268df4709197b6fa16fa1145e25c4e776137`.
- Browser visual check was skipped because Chrome/Chromium is unavailable in the environment. The receipt is in [the visual-check sidecar](RUNTIME-SIMULATOR-INTEGRATION-ARCHITECTURE.visual-check.json).
- Full repository pytest was not run: the configured WP3 evidence root `D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1` is absent. No HELD-OUT data was accessed.

## Running locally

From the repository root, start the runtime:

```powershell
.\.venv\Scripts\python.exe -m src.runtime
```

In another terminal, install and start the frontend:

```powershell
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/`. The Sandbox health indicator reports whether the local Python runtime is connected. If it is stopped, start the backend and use **Retry connection**.

## Readiness and known limits

The implementation is ready for local development and interactive integration. The measured policy runs retain their actual failure and step-limit outcomes; this report makes no scientific performance claim and does not replace the frozen evaluation protocol.

The custom converter supports the current Racing Sandbox draft contract and existing simulation primitives. Unsupported crossing/stacked geometry is rejected explicitly. Browser-level visual review remains pending due to the missing browser executable.

The code review found no standards or spec blockers. One low-severity maintainability note remains: `_runtime_fault` maps some `ValueError` cases by matching message text; typed domain errors would make those error codes independent of wording.


## November 1, 2026: Windows full-regression hardening (follow-up to earlier verification)

Validation branch: `feature/runtime-simulator-bridge`. Clean source HEAD tested: `a4b66f714d70bd85264c35213d3ff10e3cab061e`; separate clone: `D:\F1-RL-RUNTIME-SIMULATOR-BANG-CONTINUE-20261101`. This section supersedes the earlier statement that full pytest had not been run. The correct locally available **TRAIN/WP3** evidence root was `D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1` (not the missing path assumed in the earlier report).

The first complete run finished with **3105 passed, 2 failed, 9 errors, 3 skipped in 2941.00 seconds**. All 11 non-passing cases were `subprocess.TimeoutExpired`, not demonstrated Physics assertion mismatches: nine setup errors shared a source-bound WP7 spatial execution fixture (240-second subprocess timeout); one rotating-owner execution case exceeded 240 seconds; one GPU/runtime fingerprint subprocess exceeded 30 seconds. Three Windows symlink/privilege skips were expected environmental restrictions. No Physics, reward, PPO checkpoint, training provenance, admission contract, test assertion, or timeout guard was modified.

Reproduction controlled CPU library oversubscription without weakening the execution contract: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`, with the unchanged 30s/240s subprocess timeouts. Isolated retests: GPU/runtime binding **1 passed in 11.25s**; source-bound straight-sprint rotating-owner execution **1 passed in 84.18s**; all four previously blocking spatial fixture track cases **4 passed in 125.64s**. This supports an execution-environment timing explanation; it does not establish a Physics defect or certify that timing behavior is identical on other hosts.

**Full suite rerun: 3116 passed, 3 skipped, 0 failed, 0 errors in 1601.82 seconds (26m41s), completed 2026-11-01 15:19:42 ICT, pytest exit code 0.** Command executed from the clean HEAD with `F1RL_WP3_EVIDENCE=D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1` and the four CPU thread env vars set to 1: `D:\F1-RL-RUNTIME-SIMULATOR-BANG\.venv\Scripts\python.exe -B -m pytest -q -rs`. The 3 remaining skips were `tests/learning/provenance/test_file_fingerprints.py`, `tests/research/test_wp2_execution_permit.py`, and `tests/track_assets/test_manifest.py`, each because Windows symlink creation was unavailable. No previously failing/erroring test was skipped or deselected.

Evidence stored locally (not committed to Git): `D:\F1-RL-RUNTIME-SIMULATOR-RECEIPTS\full-pytest-clone-20261101.log` (initial failure), `target-gpu-wp7-20261101.log`, `target-spatial-wp7-20261101.log`, `target-spatial-fixture-20261101.log`, `full-pytest-recheck-20261101.log`, `regression-run-config-20261101.json`, and `RUN-RECHECK-AND-FULL-20261101.ps1`. Reusable source-controlled Windows runner: `scripts/diagnostics/run_full_regression_windows.ps1` (requires explicit Python interpreter and evidence paths, clean worktree, records separate logs). The successful full run predates this documentation-only / PowerShell-runner commit; registered Python source and model/evidence bindings remain unchanged.

**Gate status:** engineering Python regression now PASS on this host. The three environmental symlink skips remain explicitly disclosed. Real-browser E2E, B-seed1 Slalom physical-invalid diagnosis and custom stacked-deck limitations are separate acceptance concerns, not resolved by this regression rerun. WP7 scientific admission and HELD-OUT remain BLOCKED; no PPO training/retraining or HELD-OUT access was performed. Do not merge to `main` without separate approval.


## Final runtime hardening follow-up (2026-11-01)

See [RUNTIME-SIMULATOR-FINAL-HARDENING-20261101.md](RUNTIME-SIMULATOR-FINAL-HARDENING-20261101.md) for the Stage 1–4 engineering acceptance matrix, real Edge browser receipts/screenshots, exact B-seed1 Slalom fail-closed diagnosis, supported custom-track execution, unsupported stacked-deck decision, cleanup/CORS tests and remaining scientific blockers. The earlier unavailability statement for browser E2E has been superseded by actual Microsoft Edge coverage on the separate clone. The original clean-source full pytest PASS remains 3116 passed/3 Windows symlink skips at `e664f50`. Run the post-commit full suite on the final clean hardening commit for its definitive test count; never waive the clean-source admission guard.
