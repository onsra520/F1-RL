# F1-RL — Handoff cho Khánh Băng: Final Runtime Simulator Hardening

**Date:** 2026-10-30
**Owner (next):** Khánh Băng
**Repository:** onsra520/F1-RL
**Work branch:** feature/runtime-simulator-bridge
**Checkpoint merged to main:** 1fe7787d39f83545753aa4fc2affc291cf00efdc
**Worktree:** D:\Projects\.worktrees\F1-RL\runtime-simulator-bridge
**Merge policy:** Continue and push on the feature branch; **do not merge or change main** until a separate acceptance request.

## 1. Goal and honest baseline

All four integration stages have working implementation on the checkpoint: (1) verified frozen-model registry + Python replay bridge; (2) live PPO over FastAPI/WebSocket; (3) backend-authoritative manual control; (4) development custom-track converter, validation, and supported-track execution. Backend owns Python Gymnasium/Physics; the React/Three.js editor is not scientific Physics.

This merge intentionally established an **engineering checkpoint**, NOT final 4-stage product acceptance. The remaining work is to finish **Final Integration Hardening** and to close or explicitly classify unsupported cases. Do not infer that the models can finish their tracks or that scientific admission was granted.

Verified before handoff:
- Focused Python regression: **89 passed** (runtime, read-only rollout, WP3 inference adapter, historical integrity, WP7 boundaries) on feature HEAD.
- Frontend: **21 passed**; typecheck and production build PASS.
- Frozen verifier: **12 models / 7 reports / 96 cell logs PASS**; Ruff on runtime/tests, dependency check, and feature diff check PASS.
- Recorded real runs at seed 412: A-seed0 / straight-sprint-v1, 500 steps / 501 frames, valid but **no finish**; B-seed1 / slalom-v1, **SuspensionResolutionError** at step 82, invalid sample.
- **Full repository pytest NOT YET RUN on runtime branch. Browser visual/E2E NOT YET RUN.**
- Historical provenance remains constrained, including original_source_reconstructed=false. WP7 scientific acceptance and HELD-OUT remain BLOCKED.

Read first:
- docs/architecture/RUNTIME-SIMULATOR-INTEGRATION-REPORT.md
- docs/architecture/RUNTIME-SIMULATOR-INTEGRATION-ARCHITECTURE.html
- docs/architecture/FINAL-DEEP-SOURCE-REFACTOR-AUDIT.md
- src/runtime/{api,catalog,custom_tracks,frames,protocol,session}.py
- src/evaluation/rollout.py
- web/src/{App.tsx,lib/runtime.ts,lib/replay.ts}

## 2. Priority-0 tasks (finish, test, record)

### P0-A — Run full regression with the **correct WP3 evidence path**

The prior report used a missing path D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1. The provisioned original directory **exists** at:
D:\Projects\.evidence\F1-RL\WP3-EVIDENCE\wp3-eval-issue51-v1

Run full regression on a **clean committed** source HEAD; source-bound tests may reject dirty/untracked files. Prefer the worktree's .venv Python over uv run to avoid generating untracked uv.lock.

~~~powershell
cd D:\Projects\.worktrees\F1-RL\runtime-simulator-bridge
$env:F1RL_WP3_EVIDENCE = 'D:\Projects\.evidence\F1-RL\WP3-EVIDENCE\wp3-eval-issue51-v1'
.\.venv\Scripts\python.exe -B -m pytest -q -rs
~~~

Report exact counts, head SHA, evidence path, failures, and skipped test names/reasons. Historical baseline had three Windows symlink-related skips; verify that any new skips really have the same environmental cause. Do not disable a guard or weaken a test.

### P0-B — Real-browser E2E on Windows (Edge or Brave)

Earlier agent report marked Chrome/Chromium unavailable, but independent machine inspection found:
- Microsoft Edge: C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
- Brave: C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe

Use one of these browsers (or compatible Playwright executable configuration), verify actual DOM and 3D scene, JavaScript console, network/WebSocket events, and produce a screenshot/test receipt. Unit tests, HTTP TestClient, and an HTML architecture-diagram preview are **not** substitutes for application-level browser E2E.

Start two terminals from the feature worktree:
~~~powershell
# Terminal A
cd D:\Projects\.worktrees\F1-RL\runtime-simulator-bridge
.\.venv\Scripts\python.exe -m src.runtime

# Terminal B
cd D:\Projects\.worktrees\F1-RL\runtime-simulator-bridge\web
npm run dev
~~~

Open http://127.0.0.1:5173/. Cover:
1. Runtime online/offline, retry connection, error display, model catalog and TRAIN track selection.
2. A-seed0 / Straight real PPO run; vehicle motion visually matches Python frames; camera orientation, speed, and trajectory match backend; valid step-limit must not show INVALID.
3. B-seed1 / Slalom error; UI shows true cause and terminates/cleans the session.
4. Manual WASD/arrows/Space uses Python controls (not React arcade). Blur/key-release stops holding actions.
5. Pause/resume/reset/stop, speed multiplier, replay load/play/pause/seek/export/import, no orphaned sessions.
6. Custom supported track: Sketch -> Build -> Validate -> Manual and compatible PPO. Unsupported cross/stacked track: clear validation error before run.
7. Disconnect, backend unavailable, repeated sessions, slow frames; no crashes or fake telemetry.
8. The graphical track/vehicle positioning must use backend reference road samples where available, avoiding unsupported visual-to-physics equivalence claims.

### P0-C — Correct frontend outcome classification

In web/src/App.tsx, the runtime result UI previously displayed **INVALID / FAILED** for any not-successful-finish, even a valid 500-step limit. Show mutually distinguishable states:
- Successful Finish (valid + successful)
- Valid · Step Limit / valid non-finish (valid + not successful)
- Invalid · Physics/Simulator Error (invalid)
- Cancelled / Stopped

Do not conflate transport failure with a legitimate non-finishing episode. Add frontend tests for all outcome classes and verify rendered messages in browser.

### P0-D — Investigate B-seed1 Slalom physics error, without changing Physics

Reproduce seed 412, B-seed1 / slalom-v1, step ~82:
SuspensionResolutionError: spatial body/contact sweep outside: invalid endpoint body or wheel support.

Capture final valid pose, action, ground/contact mode, relevant simulator diagnostics, and exact failure path. Decide whether the error is an expected fail-closed physical invalid transition or an actual existing runtime/physics defect. Confirm backend emits a truthful invalid terminal state/replay with no hung session.

**Do not quietly modify the registered Physics, PPO model, reward, admission, source hashes, or evidence to force the model to finish.** If a Physics algorithm fix is truly needed, document evidence and propose separately for explicit approval. Failure to finish is not itself an integration regression.

### P0-E — Custom track scope and Stage 4 limitations

Current converter in src/runtime/custom_tracks.py explicitly rejects automatic overpasses producing stacked road decks with unsupported_intersection; frontend editor can visualize them. Preserve fail-closed behavior until validated compatibility exists.

- Verify road width, road surface support, start/finish, navigation/checkpoint ordering, elevation, supported ramps/bridges, booster/obstacles/cones, custom-manual and contract-compatible custom PPO.
- Distinguish **visual-only supported by editor** versus **Python-playable supported by converter**; give specific pre-run validation feedback.
- Audit for backend/frontend geometry mismatch and regression under repeated editing, import/export, save/reload.
- If stacked-deck support requires reworking registered Physics, mark **BLOCKED / requires separate approval**, not COMPLETE and not a fabricated visual approximation.
- If safe support using existing validated primitives is possible without altering protected Physics/contracts, implement with strict tests, including vertical overlap, support choice, contact and deterministic replay.

## 3. P1 hardening

- Verify WebSocket lifecycle and single-controller ownership, bounded queues and frame backpressure; avoid dropping critical terminal/error events without recovery.
- Verify restart/cancel/disconnect cleanup, no idle background worker, max 2 active and 8 retained sessions, timeout/limit behavior.
- Verify reproducible fixed-seed runs, action bounds, finite telemetry, model/track hashes, session provenance.
- Replace fragile error-message substring classification in src/runtime/api.py::_runtime_fault with typed domain errors if it can be done without changing behavior.
- Test security boundaries: localhost-only, restrictive origins, no arbitrary model paths, no HELD-OUT or scientific training routes, invalid commands fail closed.
- Record practical CPU, memory, end-to-end latency / FPS and behavior when the frontend slows. Do not equate playback/render speed with fixed physics timestep.
- Keep existing visual layout and editor UX. Do not redesign unrelated screens.

## 4. Acceptance gates and reporting

Must demonstrate:
1. Real frozen model -> Python env -> inference -> live backend frames -> Three.js car, plus stored replay.
2. Manual and PPO genuinely use the same Python physics. A failed/non-finishing policy outcome remains truthful.
3. Supported custom track drives and replays; unsupported features return clear actionable reasons. State any accepted Stage 4 limitations explicitly.
4. Browser E2E PASS in Edge/Brave, including visual/console/network receipts.
5. Full pytest with proper WP3 evidence PASS (or an exact reproducible blocker); runtime tests and frontend tests PASS; typecheck/build/compile, scoped Ruff, dependencies, 12 YAML/46 workflow routes as applicable.
6. Frozen models/evidence untouched and verifier PASS 12/7/96; no unauthorized HELD-OUT or training; clean worktree.
7. If any gate is BLOCKED, report the reason rather than declaring full completion.

Update docs/architecture/RUNTIME-SIMULATOR-INTEGRATION-REPORT.md and this handoff with exact command, date, clean validation SHA, observed run outcomes, accepted limitations, images/logs where relevant, and remaining blockers.

## 5. Git/operational constraints

- Work **only** on feature/runtime-simulator-bridge (use its worktree); commit and push verified fixes to this branch.
- **main must remain at checkpoint 1fe7787** until user explicitly authorizes another merge.
- Do not overwrite local untracked D:\Projects\F1-RL\REAME.md.
- No GitHub Actions; no PPO training/retraining, HELD-OUT access, scientific admission changes, altering frozen model checkpoints/evidence/manifests/hashes, or source-binding contract weakening.
- Never force-push, rewrite history, or silently repair evidence/CRLF.
- Use genuine runtime fixtures; no scripted successful driving, fake trajectories or forged scientific claims.
- On any blocker requiring protected behavior changes: preserve completed branch work, document blocker, and request explicit technical approval before touching it.

**Handoff expected from Băng:** branch, commit SHA, Stage 1–4 status (complete/partial), browser E2E result, full pytest counts, exact B-seed1 diagnosis, custom overpass decision, evidence verifier, documented blockers, and recommendation for the next merge review.

## Continuation receipt — 2026-11-01 (full regression)

- Separate clone: `D:\F1-RL-RUNTIME-SIMULATOR-BANG-CONTINUE-20261101`; branch `feature/runtime-simulator-bridge`. Validated clean source SHA `a4b66f714d70bd85264c35213d3ff10e3cab061e`.
- The actually available local WP3 evidence path was `D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1`. Earlier instructions using `D:\Projects\.evidence\F1-RL\WP3-EVIDENCE\wp3-eval-issue51-v1` do not reflect this clone machine.
- Initial full regression: **3105 passed / 2 failed / 9 errors / 3 skipped**. All 11 non-passing cases were 240-second isolated WP7 execution or 30-second GPU subprocess timeouts, not Physics assertions. No scientific tests or guards were disabled.
- Corrective measure: run tests with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, preserving original timeout limits. Targeted GPU test passed (11.25s), rotating-owner straight sprint passed (84.18s), four source-bound spatial fixture cases passed (125.64s).
- **Full regression rerun PASS: 3116 passed, 3 skipped, 0 failed, 0 errors, 1601.82s; completed 2026-11-01 15:19:42 ICT.** Skips were unchanged Windows symlink permission limitations. Exact external logs and environment receipt: `D:\F1-RL-RUNTIME-SIMULATOR-RECEIPTS\`; full log `full-pytest-recheck-20261101.log`.
- Reusable Windows regression command lives in `scripts/diagnostics/run_full_regression_windows.ps1`, using explicit Python/evidence paths and a clean HEAD. This is execution-environment hardening, **not** a change to registered Physics, PPO model, source-hash rules, reward, evaluation/admission contracts, or historical evidence.
- Python full-regression gate now PASS on this host. Other final-integration gates (broader Edge browser E2E, precise Slalom B-seed1 diagnosis and unsupported stacked deck) are still separate. WP7 scientific admission/HELD-OUT BLOCKED. No merge into `main` authorized.


## Final engineering continuation — 2026-11-01

The detailed acceptance matrix, actual Edge screenshots, WP3 frozen verifier, B-seed1 tick-82 diagnostic and supported/unsupported custom-track decision are recorded in [RUNTIME-SIMULATOR-FINAL-HARDENING-20261101.md](RUNTIME-SIMULATOR-FINAL-HARDENING-20261101.md). A genuine terminal UI visibility issue was fixed so a valid PPO step-limit or invalid simulator outcome stays visible after replay loads; manual and replay can still be opened explicitly. A local-only backend port override and CORS for localhost port 5174 allow isolated browser E2E without interfering with an existing local editor. The WebSocket cleanup test waits for actual HTTP 404 within a bounded deadline because teardown is asynchronous.

Stage 1 registry / bridge: PASS (TRAIN development). Stage 2 live PPO / replay: PASS for observed A and B workflows. Stage 3 backend manual: PASS including keyboard release and pause/resume/reset/stop. Stage 4: Python-playable supported custom track PASS; stacked-deck automatic crossing explicitly BLOCKED as `unsupported_intersection` until separately approved Physics work. No science or HELD-OUT permission. The original clean full regression PASS is documented above; final source-bound full regression must run after the one final feature-branch commit, because WP2 manifest tests intentionally reject any dirty worktree. Do not merge main before a separate review.


## Close Runtime Integration continuation — 2026-11-01

Working branch remains `feature/runtime-simulator-bridge`, based on audited HEAD `f7b228cd1952705a0fa866e2e68411c56efb3ee0`; remote had no newer branch commit. Custom Track selection is now fail-closed: invalid validation keeps Custom selected and Run disabled, stale validation is invalidated on map edit/undo/redo/import, and only the explicit registered-track action changes selection to TRAIN. Runtime request resolution cannot fall back from Custom to TRAIN. Four frontend regression cases cover invalid selection, stale validation, explicit Custom/TRAIN routing, and PPO-incompatible Custom. The detailed implementation and E2E receipts are in [RUNTIME-SIMULATOR-FINAL-HARDENING-20261101.md](RUNTIME-SIMULATOR-FINAL-HARDENING-20261101.md).

- Frontend tests: **33 passed**; TypeScript typecheck and production build PASS. Existing large-bundle advisory remains.
- Isolated Python runtime regression: **76 passed**. A concurrent run hit the strict WebSocket cleanup timeout while unrelated suite/browser processes were active; the isolated rerun passed with the unchanged assertion.
- Frozen evidence verifier: **12 models / 7 reports / 96 cell logs PASS**.
- Edge selection E2E: invalid Custom stays selected, error is visible, Run is disabled, and no session is created; editing a validated Custom draft stales validation; valid Custom creates a custom-development session; explicitly switching to TRAIN creates a TRAIN session. Previously recorded Edge PPO, Manual, Replay, reconnect, and cleanup flows remain the evidence for unchanged paths.
- **Full pytest PASS on clean committed source SHA `c6f3e06bafa3549ac733bc9aa465459f4c92ff86`: 3117 passed / 3 skipped / 0 failed / 0 errors, 1659.64 seconds.** The checked-in Windows runner first passed its targeted WP7 precheck (4 passed / 140.09s), then ran `python -B -m pytest -q -rs` with `F1RL_WP3_EVIDENCE=D:\Projects\.evidence\F1-RL\WP3-EVIDENCE\wp3-eval-issue51-v1` and OMP, MKL, OpenBLAS, and NumExpr each limited to 1 thread. The only skips were the existing symlink privilege restrictions in `tests/learning/provenance/test_file_fingerprints.py`, `tests/research/test_wp2_execution_permit.py`, and `tests/track_assets/test_manifest.py`. External receipts: `D:\Projects\.temp\F1-RL-runtime-close-c6f3e06\`.

WP3 remains frozen and used only as the configured regression evidence. No Physics, PPO algorithm, scientific-contract, checkpoint, or frozen-evidence content was changed. The next roadmap work that touches WP7 scientific contracts/registered Physics or stacked-deck support conflicts with the current scope guard and remains blocked for separate authorization; no HELD-OUT, training, GitHub Actions, or `main` work was started.
