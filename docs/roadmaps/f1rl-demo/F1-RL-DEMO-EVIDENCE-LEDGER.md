# F1-RL — Demo Evidence Ledger

> **Document version:** 1.8 (2026-11-11). **Role:** inspectable factual evidence register, NOT a full demo success report.
> **Master contract:** [Roadmap](F1-RL-DEMO-COMPLETION-ROADMAP.md). **Operational resume:** [Handoff](F1-RL-DEMO-AGENT-HANDOFF.md).  
> **Initial global verdict:** `NOT_ACCEPTED / NOT_EVALUATED`. No real PPO training, live demo acceptance or scientific HELD-OUT proof is claimed here.

## 0. M1 layered-road continuation evidence (2026-11-10)

- Execution worktree: `D:\Projects\.worktrees\F1-RL\f1rl-demo-completion`; branch `feat/f1rl-demo-completion`; source implementation commit `b578d83` (based on `161320216572cb736a1a7922f3656bc4f943ec4a`). Roadmap docs are being committed separately; `main` remains untouched. The commit adds opt-in `spatial_layered_road_edge_lifecycle_v5`; V4 remains the default and its strict projected-atlas behavior remains unchanged. V5 is available to explicit custom Auto Drive development only; training and smoke admissions remain closed.
- Geometry/API targeted command: `D:\Projects\.venvs\f1rl-wp3-parallel-training\Scripts\python.exe -m pytest tests/sandbox/test_multi_surface_support.py::test_layered_atlas_requires_nonintersecting_planes_and_body_clearance tests/runtime/test_custom_tracks.py::test_custom_track_accepts_only_body_clearance_safe_grade_separated_crossings tests/runtime/test_spatial_geometry_context.py::test_layered_geometry_context_is_opt_in_and_body_clearance_bounded tests/runtime/test_api.py::test_custom_auto_layered_profile_is_explicit_and_source-bound -q` — **PASS**, 4 passed, 1 Starlette/httpx deprecation warning.
- Earlier broad affected Python check: `... pytest tests/runtime/test_spatial_geometry_context.py tests/runtime/test_custom_tracks.py tests/runtime/test_api.py tests/runtime/test_session.py tests/sandbox/test_multi_surface_support.py tests/integration/test_wp3_applied_crash_end_to_end.py -q` — 95 passed. That result predates the final revisions and is superseded by the final command below. V5 remains rejected by the V4-only crash/roll terminal validators; no V5 terminal contract was introduced.
- Frontend: `npm test` in `web/` — **PASS**, 11 files / 105 tests. `npm run build` — **PASS**, TypeScript and Vite production build; Vite reports the existing >500 kB chunk advisory. `impeccable.cmd detect --json web/src/App.tsx web/src/lib/runtime.ts` — **PASS**, no findings.
- Syntax: `python -m py_compile` over all modified Python source files — **PASS**.
- Initial failed Auto Drive probe (retained historical evidence): seed 412, profile `experimental_v5_layered`, stopped at step 77 before traversal with a physical wheel-edge-stop. It was a valid failure of the earlier implementation and was not deleted. Final revised route result is recorded below.
- Readiness decision: **V5 Auto Drive feasibility PASS; M1 full acceptance BLOCKED** (2026-11-10). One custom layered V5 route completed at tick 698 (451.281316836m, valid sample, no physical events), but TRAIN/development coverage lacks intersection/overpass and combined custom geometry. This is simulator feasibility, not policy evidence. V5 interface probing is not physics-semantic checkpoint compatibility. No PPO training, M2 smoke, final acceptance track, held-out access, or main merge occurred. Training runtime budget unchanged; Auto Drive probes are not PPO workload.

## 0. M0 reconciled source and engineering evidence (2026-11-10)

- Both documentation PR #60 and parallel-supervisor PR #61 have been merged into `main`. Local and origin main: `ab8951a9c68f009d7b2a14127b13c07426a2460a`; execution worktree: `feat/f1rl-demo-completion` at that baseline.
- **REAL_SMOKE / REAL_BENCHMARK historical attempts were independently sampled and 5/5 model-final.zip SHA-256 digests matched**. Actual PPO source for those attempts: `8582d9d08826074dbd2b6d32609d9d58639f771d`; current merged source differs by a benchmark attempt metadata-only fix (do not relabel old evidence).
- Historical result totals: real smoke 256 steps / 10 optimizer updates; sequential 2 x 4,096 steps, wall 696.250s; parallel 2 x 4,096 steps, wall 367.500s, measured speedup 1.8946x. Single pair only; no full training or final PPO finish evidence.
- Frozen WP3 v3 SHA-256 verified unchanged: `8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4`.
- Fresh M0 preflight: Python 3.12.15, 32 logical CPUs (historical verified), free RAM 27,105,672 KiB, D: free 611,822,104,576 B; no active matching PPO jobs. Measurements are a snapshot, not a guarantee for later training.
- Targeted regression independently rerun after the metadata fix: **68 PASS, 1 SKIPPED**; pytest option `-o addopts=` was required because optional pytest-xdist is absent in that venv. No full-system regression or scientific admission acceptance is claimed.
- **Post-merge independent targeted tests on `main` `ab8951a9c68f009d7b2a14127b13c07426a2460a`: 121 passed, 2 skipped in 92.29s** (9 selected research suites, command `python -m pytest -o addopts= -q ... --maxfail=1`; xdist unavailable). This is not the full repository suite.
- Historical 5/5 checkpoint SHA-256 and telemetry hashes were independently checked; each checkpoint loaded with SB3 PPO.load and produced finite predictions. Historical attempt headers used a 40-character Git tree OID under `source_tree_sha256`; this metadata bug was fixed before PR #61, while original attempts remain immutable.
- **Post-merge real PPO smoke PASS**, attempt `smoke-20261110T133738Z-a02802b3`, execution branch `feat/f1rl-demo-completion` HEAD `6452517e3e887ee83f9d98b3323e3e5089d4c187`, 256 transitions, 10 optimizer updates, valid finite gradients/weights/telemetry, checkpoint independently reloaded and predicted, no HELD-OUT, cleanup PASS. Artifact root `D:\Projects\.archives\F1-RL\wp3-postmerge-real-smoke-20261110`. Checkpoint SHA-256 `09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f5bf`; 64-character source tree digest `db7e03e24b3571fe01fde83b9f061cbe8170db2a16b5e403dc0b6efc3ca2909c`.
- **Performance caution:** post-merge 256-step smoke wall 90.718s, rollout 85.255s, vs original smoke wall 22.328s and rollout 18.040s. No controlled source-identical repetition has determined whether the cause is host load or an environment/performance regression; do not extrapolate full-training ETA from either sample.
- Current M0 state: **PASS** (prior evidence, 2026-11-10). V5 Auto Drive feasibility **PASS**; M1 full acceptance **BLOCKED** pending expanded TRAIN/development geometry coverage. M2-M10 remain PENDING; V5 is not admitted for training or smoke and V4 remains the admitted profile.
- Documentation audit: all 56 tracked Markdown files (root, configs, assets, docs, runtime/tests, and web) were read; the 90 files under `docs/` total 4,555,210 bytes. The 24 JSON records were parsed for source/status/scope metadata; the 30-entry handoff SHA256SUMS manifest verified 30/30. Twelve historical Phase 2C JSONL records were parsed as TRAIN-only evidence (4 finishes, 8 valid physical crashes; no held-out geometry). The four replay-driver scripts were classified as immutable evidence tooling; they were not rerun. This supersedes the earlier approximate 4,551,226-byte inventory.
- The generated architecture JSON/HTML is bound to historical revision `99c5313c8bf98c164518b880e626383ad7f1c6f5`; it is not the current system map. Current source flow: Web UI (`web/src/App.tsx`) → FastAPI/WebSocket (`run.py`, `f1rl/local_demo/runtime/api.py`) → `RuntimeSession` (`runtime/session.py`) → `RuntimeEnvironmentAdapter` (`runtime/adapter.py`) → Gym/factory and sandbox world (`learning/gym_environment.py`, `sandbox/world.py`). Custom editor drafts enter `runtime/custom_tracks.py::convert_custom_track`, which creates `TrackWorld` (`sandbox/track/model.py`) and a development factory. Track surfaces are projected planar patches (`sandbox/track/patch.py`).
- Current subsystem map: Physics V4 motion and support are in `f1rl/sandbox` plus `f1rl/runtime`; custom geometry is built/validated in `f1rl/local_demo/runtime/custom_tracks.py`; PPO wraps SB3 in `f1rl/learning/algorithms/ppo.py` and receives environments via `learning/gym_environment.py`; inference loads frozen policies through `RuntimeCatalog`; the supervised parallel runner is `f1rl/local_demo/parallel_campaign.py`; the local demo is served by `runtime/api.py` and streamed by `runtime/session.py`. Training admission remains in the research/learning gate path, independent from local-demo PPO inference.

| Subsystem | Entry point and dependency flow | Contract and regression coverage | Current gap / M1 impact |
|---|---|---|---|
| Simulator / Physics V4 | `sandbox/world.py::step_world` → motion integrators in `sandbox/*dynamics.py` → surface support/collision helpers | Physics profile in `f1rl/runtime/physics.py`; geometry in `track/patch.py`; regression suites `tests/sandbox/`, `tests/integration/` | `_atlas` is explicitly a height-consistent 2.5D atlas; it is shared by multiple integrators, so changing overlap semantics affects all of them. |
| Track Builder | `web/src/App.tsx` → `POST /api/v1/tracks/validate` (`runtime/api.py`) → `custom_tracks.py::convert_custom_track` → `TrackWorld` | Draft schema `f1rl_racing_sandbox_draft_v1`; result `f1rl_custom_track_conversion_v1`; `tests/runtime/test_custom_tracks.py`, `tests/runtime/test_api.py` | V5 marks body-clearance-separated overlaps for explicit Auto Drive; one end-to-end custom crossing route completes. This is not an exhaustive custom geometry matrix. |
| RL / PPO | `learning/algorithms/ppo.py` → `learning/gym_environment.py::F1RLEnv` → runtime physics factory | observation/action contracts, including `steering-throttle-brake-v2`; `tests/learning/test_action_contract.py`, `test_observation_contract.py`, `test_gym_environment.py` | Existing checkpoint load/prediction proves API shape only. Layered geometry would alter runtime state/physics assumptions and requires fresh semantic compatibility review; loading alone is insufficient. |
| Parallel training supervisor | `local_demo/parallel_campaign.py::run_parallel_campaign` → isolated worker attempts and admission/manifest checks | frozen v3 campaign config and source-bound attempt manifests; supervised separately from the interactive API | Existing 1.8946x result is a single matched 2-job benchmark pair. It does not address M1 geometry or authorize a training campaign. |
| Web/live demo | `run.py` → FastAPI `runtime/api.py` → `RuntimeSession`/WebSocket → `RuntimeEnvironmentAdapter` → Gym environment | `/api/v1/health`, track/session endpoints; tests `tests/runtime/test_api.py`, `test_session.py`, `test_simulation_step.py` | Track validation now reports observation/action, physics-profile, and Pure PPO runtime compatibility separately. Legacy `ppo_compatible` means the full Pure PPO runtime gate; V5 layered tracks report false and are Auto Drive only. Native visual rendering was not run. |
| Admission / research evidence | `f1rl/research/wp7_admission.py` and learning admission checks → training/evaluation coordinators | TRAIN/HELD-OUT separation, source/track manifests, owner-scoped exception record | Scientific admission and HELD-OUT remain blocked/sealed; no M1 engineering result changes those gates. |
- Original defect was reproduced at base HEAD `161320216572cb736a1a7922f3656bc4f943ec4a`: the crossing draft `[(0,0),(100,100),(0,100),(100,0),(160,0)]` was rejected as `unsupported_surface_intersection`. The opt-in V5 atlas rule admits grade-separated projected overlaps only when planes do not intersect and vertical clearance exceeds the full vehicle body diagonal plus 0.2m; sequential generated patch seams retain a 0.01m tolerance. V4 remains strict. The initial failed probe is retained above; subsequent fixes and the final complete route establish M1 readiness.
- Reproduction (expected gate failure): `D:\Projects\.venvs\f1rl-wp3-parallel-training\Scripts\python.exe -c "from pathlib import Path; from f1rl.local_demo.runtime.catalog import RuntimeCatalog; from f1rl.local_demo.runtime.custom_tracks import convert_custom_track; draft={'schema':'f1rl_racing_sandbox_draft_v1','id':'audit-crossing','name':'Audit crossing','width':11,'points':[{'x':x,'z':z} for x,z in [(0,0),(100,100),(0,100),(100,0),(160,0)]],'props':[]}; convert_custom_track(draft, RuntimeCatalog(Path.cwd()))"` exited 1 with `TrackValidationError: road surfaces could not be certified; reshape turns and keep crossings separated` (2.60s, Python 3.12.15).
- Current baseline compatibility is limited to checkpoint deserialization, finite predictions, and declared observation/action-space compatibility (`obs=16`, `steering-throttle-brake-v2`). It does not establish driving performance, physics equivalence, or compatibility with a future layered-track contract. No held-out geometry was opened.
- Historical M1 evidence scope: independent left/right TRAIN-v2 controls at source `1de279c37e62a27531d661480d20fa6e095c4e84` finished/replayed exactly in 497 steps each. These runs prove paired route/controller feasibility under that source-bound V4 profile only. They did not prove custom bridge intersections. The old converter rejection and step-77 physical crash were addressed by the current candidate; they are retained only as historical failures. Current M1 acceptance is limited to registered TRAIN coverage plus the single custom V5 layered crossing route and scoped regressions; a full custom matrix across all listed features remains future work.
- Focused regression ran from execution worktree HEAD `161320216572cb736a1a7922f3656bc4f943ec4a` (source-only diff from post-smoke `6452517` is empty): 337 passed, 1 skipped in 30.66s across sandbox physics, track assets, custom-track/runtime and learning contract suites. Command: `python -m pytest -o addopts= -q tests/sandbox/test_bounded_patch_clearance.py tests/sandbox/test_body_collision.py tests/sandbox/test_world_step.py tests/sandbox/test_multi_surface_support.py tests/sandbox/test_torque_free_swept_ccd.py tests/sandbox/test_directed_center_crossing.py tests/track_assets tests/runtime/test_custom_tracks.py tests/runtime/test_simulation_lifecycle.py tests/runtime/test_simulation_step.py tests/runtime/test_runtime_snapshot.py tests/runtime/test_api.py tests/runtime/test_session.py tests/learning/test_action_contract.py tests/learning/test_action_mapping.py tests/learning/test_observation_contract.py tests/learning/test_observation_motion.py tests/learning/test_observation_target.py tests/learning/test_gym_environment.py --maxfail=1`; Python 3.12.15 venv. One Starlette/httpx deprecation warning.
- Web checks: `npm ci` completed; `npm test -- --reporter=dot` PASS (11 files, 105 tests); `npm run build` PASS. Bundle-size warning >500 KB. Launcher compatibility checks passed; `/api/v1/health` and frontend HTTP returned 200. Browser rendering/FPS was NOT RUN because native UI bridge was unavailable.
- M0 direct checkpoint compatibility: post-merge smoke checkpoint `09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f5bf` loaded with SB3 PPO and predicted finite bounded 3D actions for the 16-feature observation; this verifies API/shape compatibility only, not driving performance. Frozen config SHA remains `8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4`.
- M1 reference-controller feasibility: `left-turn-train-v2` and `right-turn-train-v2`, Physics V4, cap 1,100; each completed 497 steps and checkpoints 0/1; controller/replay exact match. No PPO model, training authorization, held-out file read or scientific evidence. Archive summaries: `D:\Projects\.archives\F1-RL\demo-completion\m1-geometry-20261110\STEP3-INDEPENDENT-V4-CONTROLS\m1-left-turn-source-bound-20261110\summary.json` SHA-256 `8ed6c11d024e5a8058e639d97eaefdc817124aafed527a058d61b4aa448d47e3`; right summary SHA-256 `b202ad09d8e310db29ad6548255c5e5291d3d727e79cd54e07db751bff121cf2`. `heldout_geometry_file_opened=false` in both manifests.
- M1 completion evidence: full Auto Drive API crossing route completed at tick 698 with valid sample and no physical events. The layered sampler checks both lower-road and upper-deck support; seam continuity and clearance regressions pass. V5 crash/roll terminal attempts remain intentionally rejected by the V4-only validators. Checkpoint probes establish action/observation interface compatibility only; they do not validate equivalent dynamics or performance. No final custom acceptance tracks were authored.
- Final affected Python verification: `D:\Projects\.venvs\f1rl-wp3-parallel-training\Scripts\python.exe -m pytest tests/runtime/test_custom_tracks.py tests/runtime/test_api.py tests/runtime/test_session.py tests/runtime/test_spatial_geometry_context.py tests/sandbox/test_multi_surface_support.py tests/sandbox/test_bounded_patch_clearance.py tests/research/test_wp3_finite_tire_footprint_v4.py tests/research/test_wp3_spatial_partial_wheel_v4.py tests/research/test_wp7_dual_turn_v4_feasibility.py tests/integration/test_wp3_applied_crash_end_to_end.py tests/integration/test_custom_layered_track_runtime.py -q` — **PASS**, 137 passed, 1 Starlette/httpx deprecation warning, 138.69s. Final focused check — **PASS**, 90 passed, 1 deprecation warning, 21.64s. `tests/runtime/test_physics_stability.py -q --tb=no` — **PASS**, 1 passed in 7.13s.
- Final Auto Drive integration test `tests/integration/test_custom_layered_track_runtime.py` covers the full custom route; test result **PASS**. Observed route metrics: tick 698, progress 451.281316836m, terminal `successful_finish`, valid sample, no physical events. Raw summary: `D:\Projects\.archives\F1-RL\demo-completion\m1-layered-route-20261110\route-summary.json`, SHA-256 `ad8358d57a74afeee99c4cb7c7d69f7481acc8f992c276cc9da4da6592ff3363`. Physics config SHA-256 `52e6cbca398f2132bcb96f99e77556f443bdd48338dd7c65f987b5d68c64ea68`.
- P1-02 compatibility fix: `/api/v1/tracks/validate` returns `compatibility.observation_action`, `compatibility.physics_profile`, `compatibility.pure_ppo_runtime`, and `compatibility.required_physics_profile`. For the layered crossing, API regression verifies `true/false/false/experimental_v5_layered`; legacy `ppo_compatible` now means complete Pure PPO runtime compatibility and is false. Attempting a PPO session returns HTTP 422 `incompatible_physics` with an Auto Drive only explanation instead of `requires_layered_v5`. Focused regression command: `python -m pytest tests/runtime/test_api.py::test_custom_auto_layered_profile_is_explicit_and_source_bound -q`; **PASS**.
- Historical P2-01/P2-02 harness attempt (superseded by §0.1): the route reached successful finish in 125.69s, but the test requested replay after WebSocket disconnect had deleted the session. The test now requests replay before disconnect and the final run passes as recorded in §0.1.
- At the initial continuation note, P2-03 benchmarking was pending; superseded by the partial measured profile in §0.1. The reported full-suite timing difference remains unattributed.

## 0.1 Independent audit follow-up (2026-11-11)

- **P1-01 — M1 status corrected:** V5 Auto Drive feasibility remains PASS, but M1 full coverage is BLOCKED. The registered TRAIN/development distribution lacks intersection/overpass and combined-feature coverage. The roadmap leaves the coverage item unchecked; M2 remains PENDING. This follow-up adjusts the gate verdict and does not claim the missing distribution was implemented.
- **P1-02 — API compatibility split:** `/api/v1/tracks/validate` now returns `compatibility.observation_action`, `compatibility.physics_profile`, `compatibility.pure_ppo_runtime`, and `compatibility.required_physics_profile`. For the layered crossing these are `true`, `false`, `false`, and `experimental_v5_layered`; legacy `ppo_compatible` now aliases the complete Pure PPO runtime result and is false. A Pure PPO create request returns HTTP 422 `incompatible_physics` with an Auto Drive-only explanation instead of `requires_layered_v5`. Command `python -m pytest tests/runtime/test_api.py tests/runtime/test_custom_tracks.py -q --tb=short`: **PASS**, 52 passed in 11.88s, one Starlette/httpx deprecation warning.
- **P2-01/P2-02 — API/WebSocket route and obstacle regression:** `python -m pytest tests/integration/test_custom_layered_track_runtime.py -q --tb=short`: **PASS**, 1 passed in 132.62s. The test validates a layered track with a barrier and cone, creates an explicit V5 Auto Drive session, streams frames over WebSocket through successful finish, checks all streamed frame collision counts/IDs remain zero, and reads the DEVELOPMENT replay while the socket is open. A preceding test-harness attempt read the replay after socket disconnect, which deletes the session; ordering was fixed and the final run passed. This covers obstacle avoidance and absence of false collision on this combined route; it does not admit V5 crash terminals or establish a full geometry matrix. Existing V4 crash integration coverage remains unchanged.
- Web verification after the compatibility/UI update: `npm --prefix web test -- --reporter=dot` — **PASS**, 105 tests / 11 files; `npm --prefix web run build` — **PASS**, with the existing 500 kB chunk-size advisory (largest JS chunk 922.09 kB).
- **P2-03 — focused performance profile (PARTIAL):** one-process benchmark pinned to logical CPU 0 (`SetProcessAffinityMask`, affinity `0x1`) on a 32-logical-CPU host. A pre-existing ComfyUI process was using CPU concurrently and was not modified, so wall-time is not an idle-host comparison; this does not reproduce the reported 596.52s full-suite run. On the layered crossing with two props, seed 412, 787 ticks completed in 107.726s wall / 101.391s process CPU, successful finish, no physical events. Unprofiled per-tick environment step: mean 135.431ms, p50 139.774ms, p95 236.396ms; process CPU mean 127.700ms. Auto Drive predict mean 1.450ms, p95 1.893ms. Thus the dominant cost is simulation/physics stepping, not controller inference, and this workload ran below 1× real time. Raw timing JSON: `D:\Projects\.archives\F1-RL\demo-completion\p2-perf-20261111\layered-route-timing.json`, SHA-256 `818009f3e592f9ba0e778a578cacce7d82f24b467fb0d5e7fbaadae5a66b448c`.
- cProfile run on the same route identified cumulative hotspots in `Simulation.step` (279.7s), `step_world` (277.2s), spatial lifecycle integration / recursive `advance` (232.2s), `stage` (152.7s), finite tire-footprint coverage (68.2s), swept chassis/patch clearance (65.9s), hard-route checks (50.6s), and body-obstacle sweeps (41.1s). cProfile overhead made this run unsuitable as a timing baseline; retain only its hotspot ranking. Raw profile JSON: `D:\Projects\.archives\F1-RL\demo-completion\p2-perf-20261111\layered-route-profile.json`, SHA-256 `c91e1acbe6ac750e63fc307342d2f0be352d0d5fed403e3bb5a6db6f146ba3bd`. The suite timing discrepancy is still unexplained; no physics optimization was made because it needs a separate contract-preserving change and benchmark.
- Current gate state: M0 **PASS**; V5 Auto Drive feasibility **PASS**; M1 full acceptance **BLOCKED**; M2 **PENDING**. No PPO training, held-out access, or main merge.
- Checkpoint contract probes: all 12 registered checkpoints loaded on a V5 environment and returned finite bounded actions with observation `(16,) float32` and action `(3,)`; interface smoke only. Separately, all 12 registered checkpoints predicted bounded actions and each completed one nonterminal transition in a straight-sprint Physics V4 Gym environment. This does not establish performance or V5 dynamic equivalence.
- TRAIN coverage inventory: straight sprint (flat, acceleration/finish); slalom (four obstacles, boost, three checkpoints); right/left turn TRAIN variants (two surfaces and slow zone); ramp (three surfaces, 8.35m rise). This is registered-training distribution coverage, not proof a learned policy can handle each feature.
- M0 local HTTP evidence archive: `D:\Projects\.archives\F1-RL\demo-completion\m0-audit-20261110\local-runtime-health.json`, SHA-256 `105D3988635283E96A2BD3D62F9A69F2713B835729311BB31FCB95BAA48C3541`. Server process was stopped and ports verified closed. Browser visual rendering remains NOT RUN.
- Independent docs/source reconciliation inventoried 103 tracked documentation/evidence files on main. Historical artifacts retain conflicting source/checkpoint eras; prioritize dated source-bound evidence and exact immutable attempt manifests. `f1rl/learning/README.md` action wording is stale against current v2 manifest and source. These are documentation drift, not a basis for changing frozen contracts.

## 0.3 Historical V4 dual-turn witness restore (2026-11-11)

- The four named JSON witnesses were absent from this worktree and were recovered byte-for-byte from GitHub branch `handoff/physics-v4-khanh-bang-20261106`, exact remote HEAD `6e30ebd78b010d3f20d120b936c0283eab6ca78a`. `git ls-remote` confirmed that branch head. The documents declare Physics `spatial_road_edge_partial_lifecycle_v4`, the directed lifecycle v4 race contract and TRAIN-only nominal left/right finishes; their summary hashes agree with references in the acceptance witness.
- Restored outside Git to `D:\Projects\.archives\F1-RL\demo-completion\source-witnesses-6e30ebd7`. `RESTORE-MANIFEST.json` SHA-256: `d77a60294dda8c7d5281521fc0d2724216c6c5c054bf910309e70a314f9c3480`.
- Restored file SHA-256: `STEP3-PHYSICS-V4-DUAL-TURN-ACCEPTANCE.json` `0bbc1636fdb330d58d0808ad83c2b0d73e7d1b686d597d87796510c6788bd5d4`; `STEP3-DUAL-TURN-PHYSICS-V4-PORTABLE-WITNESS.json` `a2b2676011edc97278653e8eecd476c7138c5b4cb4090479f95d174a70e667fc`; right summary `6f7bb46a0ce6aad87edd547c1be1809022db8b24c228990c8cedd295707ac0bd`; left summary `6dea077e3bd4a47a4f8d75309f75b9824496a8dbaaccbca597737e52d0a12004`.
- Scope limit: the remote commit contains only these four JSON witnesses, not the referenced control tapes, episode JSONL files or tick traces. Their declared digests can be checked for internal consistency but the raw replay evidence cannot be rehashed or independently replayed from this restore. Preserve this as historical source evidence, not a current-source runtime test.

## 1. Evidence integrity rules

- **Evidence classes:** `SOURCE_VERIFIED`, `STATIC_TEST`, `MOCK_TEST`, `REAL_SMOKE`, `REAL_BENCHMARK`, `REAL_TRAIN`, `REAL_DEV_EVAL`, `REAL_ACCEPTANCE`, `LIVE_BROWSER`, `INDEPENDENT_REVIEW`. Do not equate one class with another.
- Every numeric claim requires metric definition, exact command and environment, source commit, timestamp, raw evidence path, SHA-256 and an independently verifiable report. If absent, write `UNKNOWN`/`PENDING`, not `PASS`.
- Local absolute paths point to artifacts **on the user's Windows machine**, not ChatGPT sandbox files; verify existence and SHA before citation. Paths in this document are conventions until an artifact exists.
- Do not edit or delete failed, cancelled or incomplete attempts. Append a new row with new `attempt_id` and immutable output root. If evidence is superseded, retain and link it.
- Keep **test evidence** separate from optimizer activity. A valid real training row needs actual `PPO.learn()` execution with recorded optimizer updates, weights change, finite gradients and checkpoint/telemetry integrity.
- HELD-OUT sealed: no listing, sampling, opening, tracking or touching registered scientific HELD-OUT material as a demo metric. Demo custom tracks are a separately generated post-freeze suite.
- When documentation is updated, source commit references and archive hashes **must refer to the actually executed build**, not simply the documentation PR HEAD.

## 2. Verified and historical starting references

| ID | Class | Evidence/reference | Current interpretation |
|---|---|---|---|
| SRC-001 | SOURCE_VERIFIED | GitHub `onsra520/F1-RL`, `main` at `3683981d99f70e1dca2bf600a7d0db93a2280ab9`, checked 2026-11-10 | Confirmed remote baseline **at check time**; re-check on every new session. Not a claim that code is demo-ready. |
| SRC-002 | SOURCE_VERIFIED | `configs/research/wp3-ppo-tuning-campaign-v3.json`, frozen SHA-256 `8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4` from previous independent local audit | Must be re-hashed from exact execution source before any real run; do not mutate. |
| HIST-001 | STATIC_TEST + MOCK_TEST (historical) | Local `feat/wp3-parallel-training-campaign` HEAD reported/previously checked at `4c8808f39e1cfdbd71c38b26ac5892c7a0719a81` | Earlier independent read-only source review found no remaining scoped P0 issues; **not merged to main** as last known. Remote branch status and current local source require revalidation. |
| HIST-002 | MOCK_TEST (historical) | Earlier targeted run: **63 tests PASS**, plus **3 evaluation/report tests PASS**; full 12-job **mock** pipeline exercised | Evidence is for orchestration/retry/chart/report/manifest with fake PPO jobs, **NOT** proof of real training or wall-clock PPO speedup. Prior run artifacts/console transcript must be located if relied upon. |
| HIST-003 | SOURCE_VERIFIED (historical contract) | Frozen v3: C-seed0 first; 12 jobs (A/B/C/D × 0/1/2), 65,536 timesteps/job, 4,096 periodic checkpoint, CPU, 1 environment | Source contract only. No real C0, full campaign or development candidate results verified. |
| HIST-004 | SOURCE_VERIFIED (historical config) | LOCAL_DEMO smoke profile `configs/research/local-demo-ppo-smoke-v1.json`: C-seed0, 256 timesteps, 1 rollout, CPU, 1 env, max wall 1,800s | Recheck path/schema before use. This is **not** the full 65,536-step engineering C0 gate. |

**No created real artifacts at inception:** `MODEL-FREEZE.json`, `DEMO-ACCEPTANCE-PLAN.json`, `DEMO-ACCEPTANCE-RESULT.json`, 70-attempt logs, real-PPO throughput benchmarks and 10-minute live browser metrics are all **PENDING**. Never backfill them with synthetic results.

## 0.2 M1 final hardening evidence (2026-11-11)

- **Gate:** M1 engineering **PASS / READY_FOR_M2**. Overall demo remains **NOT_ACCEPTED**; M2, PPO training, final acceptance tracks, held-out access and main merge were not performed.
- **Versioned development fixtures:** `tests/runtime/fixtures/m1-development-geometry-v1.json`, schema `f1rl_m1_development_geometry_v1`, version 1, scope `TRAIN_DEVELOPMENT_ONLY`; SHA-256 `d215db696679cba6c1fa9515a261a2f038173b2c397ed17489744f2a8f7924f6`. It covers a grade-separated intersection/overpass with checkpoints, cone/barrier, and a combined left/right/S-turn course with obstacles, ramp/elevation and checkpoint. The official registered TRAIN roster was not changed; no final acceptance or HELD-OUT track was created/read.
- **Geometry/Physics/reference Auto Drive:** `.venv\Scripts\python.exe -m pytest -q tests/integration/test_m1_development_geometry.py` — **PASS**, 2 passed in 233.34s. Both fixtures pass conversion/world checks, route width, finite navigation, checkpoint/finish counts and zero-collision complete reference-controller replay. The intersection uses explicit Layered Physics V5; the combined elevated custom Auto route also selected V5. V5 remains custom Auto Drive only. Physics V4 behavior/admission was not changed.
- **API/WebSocket E2E hardening:** `.venv\Scripts\python.exe -m pytest -q tests/integration/test_custom_layered_track_runtime.py` — **PASS**, 2 passed in 103.52s, one existing Starlette/httpx deprecation warning. It uses the versioned intersection fixture, bounds each WebSocket receive at 20s and total run at 240s, records last event/tick/frame count, distinguishes API/WS and Physics terminal failure from timeout, fetches session diagnostics on timeout, disconnects/deletes the session and verifies it is gone. The same command includes a bounded-timeout regression. Final log: `api-websocket-e2e-final.log`.
- **Python/contract regressions:** `.venv\Scripts\python.exe -m pytest tests/runtime/test_custom_tracks.py tests/runtime/test_api.py tests/runtime/test_session.py tests/runtime/test_spatial_geometry_context.py tests/sandbox/test_multi_surface_support.py tests/sandbox/test_bounded_patch_clearance.py tests/research/test_wp3_finite_tire_footprint_v4.py tests/research/test_wp3_spatial_partial_wheel_v4.py tests/research/test_wp7_dual_turn_v4_feasibility.py tests/integration/test_wp3_applied_crash_end_to_end.py -q` — **PASS**, 136 passed in 40.17s; output `python-regression.log`. `.venv\Scripts\python.exe -m pytest tests/learning/test_observation_contract.py tests/learning/test_action_contract.py tests/runtime/test_physics_stability.py -q` — **PASS**, 44 passed in 5.90s. `.venv\Scripts\python.exe -m pytest tests/learning/algorithms/test_ppo_runtime_adapter.py -q` — **PASS**, 3 passed in 1.47s. No Physics implementation or PPO contract changed.
- **Frontend:** `npm --prefix web test -- --reporter=dot` — **PASS**, 105 tests / 11 files. `npm --prefix web run build` — **PASS**, TypeScript and Vite build; existing >500 kB advisory remains (largest bundle 922.09 kB). Logs: `web-tests.log`, `web-build.log` in the archive root below.
- **P2-03 low-load V5 benchmark:** pre-run Windows CPU total samples were 7.0, 4.1, 6.3, 0.3, 3.1% (mean 4.16%), 32 logical CPUs. Benchmark on seed 412, same 787-tick layered intersection route with cone/barrier/checkpoints: wall 102.765s, process CPU 100.984s, Physics-step wall mean/p95 98.271/185.254ms, Physics-step CPU mean/p95 96.728/187.500ms, simulation 78.7s, realtime factor 0.766×, successful finish, 0 collision frames. Physics stepping remains the observed hot path; no optimization was made. Historical CPU-pinned/noisy-host result was wall 107.726s, CPU 101.391s, wall-step mean/p95 135.431/236.396ms on a route without these checkpoint props. The 5.0s CPU difference is small; this is one run per condition, not a statistically controlled speedup claim.
- Raw benchmark `D:\Projects\.archives\F1-RL\demo-completion\m1-hardening-20261111\v5-benchmark.json` SHA-256 `7AE4C1B875C5A0301597C4FD36E46CF2FB836684653FA05F3AE2F4F39CF876E4`; host-load snapshot SHA-256 `F2CEE1AC2136B9734DB188195C41077F2959A8987891E59468FF0173705588FA`. Benchmark binds execution base `8e7b99c4d462bdbfdf6d010e068404e03e717a08`, worktree source-tree digest `e4464af19da4eda9f83fc44325bf223a8866f7bde7d20148fcfd6a64c6a53df9`, fixture SHA above, Physics V5 config SHA-256 `52e6cbca398f2132bcb96f99e77556f443bdd48338dd7c65f987b5d68c64ea68`, and registered Physics V4 config SHA-256 `c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37`.
- **Profile limitation:** the combined elevated route initially failed under the default V4 session at tick 0 with `SuspensionResolutionError: spatial contact requires continuous 2.5D road atlas`; explicitly selecting the already-supported experimental V5 custom Auto profile let the same geometry finish. This task did not modify V4 or expand V5 admission; the combined fixture's passing runtime evidence is specifically V5 Auto Drive.
- Active-workload budget charge: this bounded real benchmark took **102.765s**, bringing the provisional cumulative total from 1,263.031s to **1,365.796s / 86,400s**; **85,034.204s remain**. No PPO training/smoke/evaluation workload occurred.
- **Evidence root:** `D:\Projects\.archives\F1-RL\demo-completion\m1-hardening-20261111`. `SHA256SUMS.json` inventories 16 source/config files and 12 output artifacts; SHA-256 `0330c4cc74b87ef7e43a317fa6f1c003da3e5a3abc47e5b66404742f89027628`. Evidence summary SHA-256 `18e12e033c4d4d992e4f314901dc7fc6ddfa22521ec641b3f481ab8271023bcd`.

## 3. Milestone evidence register (append/update only with provenance)

| Gate | State | Minimum evidence to link | Actual source SHA | Attempt(s) | Verified by |
|---|---|---|---|---|---|
| M0 Source admission | PASS | clean refs/worktree; 337/1 regression; web tests/build/HTTP health; checkpoint API compatibility; resource/budget snapshot | `161320216572cb736a1a7922f3656bc4f943ec4a` (source diff empty from smoke source `6452517`) | `m0-audit-20261110` | independent local audit |
| M1 V5 feasibility + full engineering coverage | PASS / READY_FOR_M2 | Versioned development intersection/overpass and combined-feature fixtures finish with reference Auto Drive; bounded API/WebSocket E2E, Physics V4/PPO contract regressions and controlled V5 benchmark pass. V5 remains Auto Drive only; PPO/final acceptance unclaimed | execution base `8e7b99c4d462bdbfdf6d010e068404e03e717a08`; worktree source-tree digest in §0.2 | `m1-hardening-20261111` | agent self-verification; independent review pending |
| M2 Real smoke | PASS / REUSED_COMPATIBLE | 256 transitions, 10 optimizer updates, finite gradients and changed weights; exact checkpoint/telemetry/profile hashes and cleanup verified | `6452517e3e887ee83f9d98b3323e3e5089d4c187` | `smoke-20261110T133738Z-a02802b3` | independent artifact/hash audit, 2026-11-11 |
| M3 PPO benchmark | IN_PROGRESS | Historical attempts retained but profile SHA `1b9f…` does not match profile at claimed source; fresh same-job 1-vs-2 run required | planned current source `f4b57a12428b8749db8c11d87fb00e95866df7f8` | planned root `m3-real-benchmark-20261111` | pending |
| M4 PPO training | PENDING | full C0 gate if v3, individual model attempts, periodic checkpoints, training integrity, source/seed linkage | — | — | — |
| M5 Dev evaluation | PENDING | TRAIN/dev-only roster, per-track rates, steering/speed/braking diagnostics, change/regression/compatibility reports | — | — | — |
| M6 Model freeze | PENDING | exact frozen checkpoint SHA, model/obs/action/physics/source/version manifest | — | — | — |
| M7 New custom suite | PENDING | post-freeze track generator commit/seed, 3 JSON hashes and independent geometry validation | — | — | — |
| M8 Pure PPO evaluation | PENDING | 10 runs × 7 tracks, predeclared seeds, per-run immutable results, 8/10 each + no-override proof | — | — | — |
| M9 Live web runtime | PENDING | browser trace, correlated backend state timestamps, 1× pace, FPS, p95 ≤150ms, action causality | — | — | — |
| M10 Demo release | PENDING | startup guide, PR URL, final summary/verdict, reviewer decision, owner merge approval | — | — | — |

**Template for one fully evidenced gate entry:**

```yaml
milestone_id: M2
attempt_id: <unique-immutable-attempt>
status: PENDING               # PASS | FAIL | BLOCKED | NOT_ACCEPTED only after verification
evidence_class: REAL_SMOKE
source_branch: <exact>
source_commit: <40-char-sha>
source_tree_hash: <exact>
config_sha256: <sha>
model_sha256: <sha-or-null>
track_sha256: <list-or-null>
host_snapshot_path: <absolute-path>
command: <exact argv/working directory>
exit_code: <integer-or-null>
start_utc: <ISO8601>
finish_utc: <ISO8601>
metrics: <units-and-calculation>
raw_artifact_root: <absolute path under .archives>
artifact_manifest_sha256: <sha>
time_charged_seconds: <measured>
cumulative_active_seconds: <measured>
result_reason: <reproducible rationale>
reviewer: <identity-or-PENDING>
verified_at_utc: <ISO8601-or-null>
```

The above is a **schema example**, not an executed M2 record.

## 4. Cumulative runtime budget ledger

**Owner cap:** `24 hours = 86,400 active execution seconds` across all smoke, benchmark, real training, real evaluation and live runtime acceptance **combined**. At most **two improvement cycles** after first model-training baseline.

| Session/attempt | Milestone | Start / finish UTC | Measured active seconds | Cumulative seconds | Remaining seconds | Status | Raw timer log/hash |
|---|---|---|---:|---:|---:|---|---|
| INITIAL (historical) | - | Pre-reconciliation snapshot | 0 newly charged by documentation | unknown at that time | unknown at that time | SUPERSEDED | - |
| 2026-11-10 WP3 real PPO smoke + benchmark | M0 prior-workload reconciliation | 7 attempt directories incl. failed smoke | **1,172.313** | **1,172.313** | **85,227.687** | VERIFIED_FROM_RESULT_JSON | `D:\Projects\.archives\F1-RL\wp3-real-ppo-benchmark` |
| 2026-11-10 post-merge 256-step smoke | M0 independent integration test, NOT M2 gate acceptance | 2026-11-10T13:37:38Z to 13:39:09Z | **90.718** | **1,263.031** | **85,136.969** | REAL_SMOKE_PASS, model/inference verified | `D:\Projects\.archives\F1-RL\wp3-postmerge-real-smoke-20261110` |
| 2026-11-10 M1 V5 development benchmark | M1 physics benchmark, no PPO | 2026-11-10T18:09:56Z–18:11:39Z | **102.765** | **1,365.796** | **85,034.204** | REAL_BENCHMARK_PASS, one bounded run | `D:\Projects\.archives\F1-RL\demo-completion\m1-hardening-20261111\v5-benchmark.json` |
| 2026-11-11 M2 smoke evidence reuse | M2 gate reuse; no new execution | Historical source-bound artifact; no active time added | **0.000** | **1,365.796** | **85,034.204** | REUSED_COMPATIBLE; checkpoint/telemetry SHA independently verified | `D:\Projects\.archives\F1-RL\wp3-postmerge-real-smoke-20261110` |

Historical M3 benchmark records are excluded from current M3 acceptance because both runs bind profile SHA `1b9f787b…`, while the profile at their claimed source commit has Git blob SHA-256 `60cd298d…`. New M3 attempts use current profile bytes and a clean committed source; charge measured elapsed wall time only after attempt completion and verified cleanup.

* **Charge breakdown** (seconds from immutable result.json): smoke attempts 0.016 + 37.453 + 22.328 + 26.438 + 22.328 = **108.563**; sequential **696.250**; parallel **367.500**. Historical subtotal **1,172.313 s** (~19m32.313s). Including post-merge smoke, **1,263.031 s** (~21m03.031s); including the M1 benchmark, **1,365.796 s** (~22m45.796s). All prior attempts, including failed smoke, count; the budget must never reset between sessions.
* **Budget epoch:** WP3 real-PPO engineering run dated 2026-11-10 UTC, adopted for demo-completion budget accounting. This amount is a lower bound if other real workloads are later found; re-audit before expensive training.

**Critical rule:** Do not interpret the placeholder `0 newly charged` as proof that no earlier actual GPU/CPU training consumed an owner-authorized budget. First M0 must discover prior sessions and define this roadmap's budget epoch. Only initialize remaining budget to 86,400 seconds after confirming it is a **new owner-authorized campaign epoch**. Persist the epoch ID and never reset it on a new agent session.

Active-time accounting example: benchmark sequential A and B contributes the **elapsed experiment wall time**, not the sum of concurrent child wall times. For an actual campaign, count from launch to verified bounded cleanup, including failed run time and child monitoring. Exclude inactive waiting and ordinary editing; do not double-count overlapping subprocess time. Track real per-process seconds separately as diagnostic.

**Resource admission register (not proof of current free memory):**

| Control | Owner limit / condition | Latest measurement |
|---|---|---|
| Concurrent PPO training jobs initially | 2 | PENDING |
| Per-worker tree RSS | ≤4 GiB | PENDING |
| Minimum available host RAM reserve | ≥8 GiB | PENDING |
| CPU/Torch threads | Verify config, initial expected 8 intra-op + 1 inter-op/job | PENDING |
| Concurrent workers=4 | only after real measured scaling + new admission | NOT_AUTHORIZED_BY_BENCHMARK_YET |
| Scientific HELD-OUT access | Prohibited | SEALED |
| Auto/Hybrid counted in acceptance | Prohibited | N/A |

## 5. Development and change-impact ledger

Every improvement cycle requires an explicit line. **Maximum two** cycles total.

| Cycle | Trigger (dev-only evidence) | Modified contracts | Compatibility classification | Tests | New train attempt / model SHA | Outcome | Time charged |
|---|---|---|---|---|---|---|---|
| Baseline | Not evaluated | — | — | — | — | PENDING | — |
| Improvement 1 | — | — | — | — | — | PENDING | — |
| Improvement 2 | — | — | — | — | — | PENDING | — |

For every physics/reward/observation/action change: archive source diff + dependency impact; old-vs-new observation feature *semantics* and order; action units/ranges and applied-control path; dynamics/collision/reset; affected model SHA(s); optimizer resume eligibility; and exact reason to retain, fine-tune or retrain. Do not relabel incompatible checkpoint as PASS just because it loads.

## 6. Final model freeze register

| Field | Required value | Current |
|---|---|---|
| Freeze UTC time | Actual timestamp preceding custom suite generation | PENDING |
| Model/checkpoint exact path + SHA-256 | One immutable chosen development-evidenced candidate | PENDING |
| Source commit/tree | Exact execution build | PENDING |
| Physics V4 / SAT source & config SHA | Full semantics | PENDING |
| Observation layout/order/normalization | Version + digest | PENDING |
| Action map/ranges and inference mode | Version + digest; Pure PPO | PENDING |
| Reward/curriculum/training config | Version + digest | PENDING |
| Python/Torch/runtime dependencies | Version lock + SHA | PENDING |
| `MODEL-FREEZE.json` location/hash | External immutable artifact | PENDING |

If **any** compatibility-relevant source/model semantic changes after freeze, invalidate its acceptance suite and obtain a fresh freeze. Previous acceptance evidence remains archived as historical; it does not count for the revised model.

## 7. Post-freeze custom track coverage (three distinct new tracks)

**These are planned feature roles, not already-created tracks.** Derive final exact geometry and seeds *after freeze*; pre-register all tracks and attempts before running frozen PPO.

| Slot | Required feature role | Track SHA | Generation timestamp (must be after freeze) | 10 reset seeds locked? | Valid geometry before PPO? |
|---|---|---|---|---|---|
| CUSTOM-01 | left + right + successive/S turns | PENDING | PENDING | NO | PENDING |
| CUSTOM-02 | cones, barrier, braking/reacceleration, turns | PENDING | PENDING | NO | PENDING |
| CUSTOM-03 | slope/ramp, bridge, grade-separated crossing/intersection, elevation | PENDING | PENDING | NO | PENDING |

Across the **suite**, all agreed Track Builder feature families must be present. Do not quietly omit a feature because it harms PPO results. Validation of route feasibility must not include testing the frozen PPO. Any pre-PPO geometric rejection gets a preserved reason and deterministic seed replacement record.

## 8. Pure PPO acceptance scorecard — no pooled averaging

**Pre-registered test design:** 10 attempts per row with explicit distinct reset-seed assignments and logged policy inference mode. Do not rerun FAILs to reach 8/10, skip invalid outcomes, or change track after results. Each cell starts `PENDING`.

| Track | Trial results (1–10) | Valid finishes | Required | Status | Evidence dir/hash |
|---|---|---:|---:|---|---|
| TRAIN straight-sprint-v1 | PENDING | —/10 | ≥8 | PENDING | — |
| TRAIN slalom-v1 | PENDING | —/10 | ≥8 | PENDING | — |
| TRAIN right-turn-v1 | PENDING | —/10 | ≥8 | PENDING | — |
| TRAIN ramp-v1 | PENDING | —/10 | ≥8 | PENDING | — |
| CUSTOM-01 post-freeze | PENDING | —/10 | ≥8 | PENDING | — |
| CUSTOM-02 post-freeze | PENDING | —/10 | ≥8 | PENDING | — |
| CUSTOM-03 post-freeze | PENDING | —/10 | ≥8 | PENDING | — |

Each trial should include: `trial_id`, `model_sha`, `source_sha`, `track_sha`, `reset_seed`, `inference_deterministic`, `pure_ppo`, `override_count`, `checkpoint_crossings`, `finish_crossing_valid`, `route_progress`, `collision_count`, `offroad/timeout`, `sim_steps`, `wall_seconds`, `policy/action telemetry_sha`, `artifact_sha`, `outcome_reason`.

**Any single track below 8/10 = M8 FAIL and final `NOT_ACCEPTED`**, even if total ≥56/70. A user-visible fallback demo may be presented but is not an accepted PPO result.

## 9. Live browser performance ledger

| Metric | Measurement definition | Gate | Actual / raw path |
|---|---|---|---|
| Pure PPO causal path | per-step observation → policy steering/throttle/brake → applied controls → physics → sequenced snapshot | 100% real policy path; 0 Auto/Hybrid overrides | PENDING |
| Simulation pace | Δsim_time / Δwall_time over steady representative live segment | 1× real-time target; report achieved distribution | PENDING |
| Browser FPS | rendered frames measured in actual browser, during moving car & track builder visualization | ≥30 FPS | PENDING |
| Backend state to visible paint | correlation by step_id and synchronized clocks/offsets; raw timestamp series | p95 ≤150 ms | PENDING |
| Stream integrity | monotonic step_id, dropped/stale/out-of-order state counts, no ghost restart | No state regressions or duplicate loops | PENDING |
| Ten-minute interactive session | actual Run, monitor, restart/new track/end terminal transitions | PASS without recording/replay | PENDING |

Avoid fabricated latency by comparing unrelated clock origins. Specify hardware, browser, viewport, monitor-refresh, CPU load, sample count, warmup and measurement windows.

## 10. Artifact layout and retention contract

```text
D:\Projects\.archives\F1-RL\demo-completion\
  <campaign-id>\
    source-admission\
    M1-track-validation\
    M2-real-smoke\attempt-0001\
    M3-real-benchmark\{sequential,parallel-2,parallel-4}\
    M4-training\<model-id>\<attempt-id>\
    M5-development-eval\<cycle>\
    M6-freeze\MODEL-FREEZE.json
    M7-custom-suite\DEMO-ACCEPTANCE-PLAN.json
    M8-acceptance\<track-id>\<trial-id>\
    M9-live-runtime\<session-id>\
    M10-release\DEMO-ACCEPTANCE-RESULT.json
```

All roots are **planned conventions**, not claims that directories exist. Store immutable SHA-256 manifest per attempt and capture source/branch/head at execution time. Logs/metrics/large files remain outside Git; commit only digest/relative references and concise reports. Never delete previously archived evidence to free space without explicit owner permission and a verified backup policy.

## 11. Handoff / reviewer reconciliation

When a milestone changes state, atomically as reasonably practical:
1. Commit revised roadmap checkbox/gate status + this ledger row + agent handoff; record evidence paths and hashes.
2. Recheck remote/local branch refs, identity and output ownership after any interruption.
3. Independently sample actual artifacts and commands; do not merely trust persisted `PASS`.
4. Record reviewer verdict separately from author/self-assessment; keep `PENDING_INDEPENDENT_REVIEW` until checked.
5. Any contradiction between ledger and actual source/artifacts ⇒ `BLOCKED` until reconciled.

**Initial action:** perform M0 reconnaissance and establish an actual runtime budget epoch. No real training is approved merely by opening this ledger.
