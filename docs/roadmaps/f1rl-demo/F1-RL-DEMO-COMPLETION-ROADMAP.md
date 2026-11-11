# F1-RL — Demo Completion Master Roadmap

> **Document version:** 1.9 (2026-11-11)
> **Overall state (2026-11-11):** `M0_PASS / M1_PASS / M2_PASS / M3_PASS / M4_ENGINEERING_PASS / M5_CYCLE1_DIAG_COMPLETE_CANDIDATE_NOT_ACCEPTED / NOT_ACCEPTED` — Cycle 1's non-training stochastic-variance ablation raised Ramp V1 from 1/10 to 10/10, but std×0.5 scored 9/10 on Slalom versus its historical 10/10, and std×0.65 scored 6/10. No PPO fine-tuning, model freeze or final acceptance occurred.
> **Owner:** F1-RL project owner. **Execution:** autonomous coding agent, gate-driven.  
> **Related:** [Evidence Ledger](F1-RL-DEMO-EVIDENCE-LEDGER.md) · [Agent Handoff](F1-RL-DEMO-AGENT-HANDOFF.md).

**Audit note (2026-11-10):** PR #60 and PR #61 merged; main baseline is `ab8951a9c68f009d7b2a14127b13c07426a2460a`. M0 PASS remains as recorded. V5 Auto Drive feasibility passes on execution commit `b578d83` in branch `feat/f1rl-demo-completion`: one layered crossing route completes and relevant seam/clearance regressions pass. Independent audit found the registered TRAIN/development distribution still lacks intersection/overpass coverage, so M1 full acceptance is BLOCKED and M2 remains pending. V5 checkpoint checks establish action/observation interface compatibility only, not physics-semantic compatibility or policy performance. V5 remains custom Auto Drive development only; training and smoke admission remain closed, and V5 crash/roll terminal acceptance remains fail-closed pending a separately versioned contract. No M2 work, final acceptance tracks, held-out access or main merge occurred. See ledger for exact evidence and limits.

**Independent audit follow-up (2026-11-11, source commit `afcc26e`):** P1-01 is resolved by separating `V5 feasibility PASS` from `M1 full acceptance BLOCKED`; missing TRAIN/development intersection/overpass coverage remains an explicit unchecked gate. P1-02 is fixed: validation distinguishes observation/action, physics-profile, and Pure PPO runtime compatibility, and layered tracks no longer claim full PPO compatibility. P2-01/P2-02 API-to-WebSocket route evidence passes with a barrier and cone. P2-03 profile points to simulator physics stepping as the bottleneck, but the reported full-suite timing variance is not explained; profiling was affinity-pinned while another CPU workload was active. M2 remains pending; no held-out access or main merge occurred.

**M1 final hardening (2026-11-11, execution base `8e7b99c`):** versioned `TRAIN_DEVELOPMENT_ONLY` geometry fixtures cover a grade-separated intersection/overpass and a combined left/right/S-turn course with cone, barrier, ramp/elevation and checkpoints. Geometry/Physics assertions and reference Auto Drive completion pass; the API/WebSocket E2E uses the versioned crossing fixture, has bounded event/wall waits, progress/status diagnostics and session cleanup. Controlled V5 benchmark, Physics V4/PPO contract regressions, Python regression, web tests and build pass. M1 is `PASS / READY_FOR_M2`; this is engineering readiness only. At this M1 checkpoint, M2 was pending and outside scope. Subsequent M2/M3 work is tracked separately in the continuation sections below and in the handoff. Evidence and hashes are in the ledger.

**M1 scope closeout audit (2026-11-11):** the `8e7b99c` execution base and M1 evidence inventory were rechecked in the feature worktree. All 16 inventoried source/config files and 12 output artifacts match their SHA-256 manifest. The M1 implementation and its archived test/build logs are unchanged. This closeout did not start or advance any M2/PPO work; later M2/M3 branch activity is outside this M1 acceptance statement.

## 0. Mission / definition of done

Finish an **interactive, live, Pure-PPO F1-RL demo** for REL301m (~10-minute presentation). User builds a track with Track Builder, presses **Run**, and sees the car move based on *current* PPO inference and Physics V4 state. A prerecorded trajectory, replay, Auto Drive or Hybrid fallback is **not** an acceptable substitute.

The model must drive autonomously, finish the route, turn left/right smoothly, steer stably, brake before demanding sections, accelerate when appropriate, and generalize to new custom tracks **without training again**.

**Engineering demo acceptance — all required:**
1. Four registered TRAIN tracks: `straight-sprint-v1`, `slalom-v1`, `right-turn-v1`, `ramp-v1`. Additional left-turn development coverage is required (these four do not establish left-turn competence alone).
2. At least **three newly generated custom acceptance tracks**, generated **only after final model freeze**; as a suite they must exercise left/right turns, successive/S turns, cones/barriers, ramps, bridges, intersections and elevation changes. No acceptance track may enter training, tuning, curriculum or model selection.
3. **Exactly 10 pre-registered evaluation attempts per track** × at least 7 tracks = **at least 70 attempts**. **Each track individually must finish ≥8/10** via valid directed finish/checkpoint crossing. Neither pooled averages nor dropped failures count. If more tracks are added, each must satisfy the same gate.
4. Evaluation in **Pure PPO mode only**: policy predicts steering, throttle and brake; no Auto/Hybrid intervention, fallback or hidden precomputed trajectories. Log raw policy actions, applied actions and any override; override invalidates that attempt for Pure-PPO acceptance.
5. Live simulation **1× real-time target**, frontend **≥30 FPS target**, backend-state-to-visible-render **p95 ≤150 ms target** with actual instrumentation. Rendering interpolation between authoritative states is permitted; gameplay, physics and action decisions must never be interpolated/fabricated.
6. Reproducible evidence, model/source/contract compatibility, crash/timeout handling, and an end-to-end live browser check. A successful recorded run alone is not acceptance.
7. Total **active execution budget ≤24 hours** for smoke, real benchmark, train, evaluation and runtime acceptance; maximum **two improvement cycles** after first baseline training. If unmet at cap, preserve best development checkpoint and report `NOT_ACCEPTED`. No automatic extension.

**Scientific boundary:** WP7/Gate 2 remain BLOCKED; registered scientific `HELD-OUT` stays SEALED. The new custom demo acceptance suite is **not** the registered HELD-OUT set and is not a substitute for scientific acceptance.

## 1. Eighteen recorded owner decisions

| # | Binding decision |
|---|---|
| 01 | Finish all four TRAIN tracks; smooth left/right turns, stable steering, speed control; prioritize finishing arbitrary custom routes. |
| 02 | New custom route must be driven without retraining after freeze. |
| 03 | Demo acceptance ≥80% = ≥8/10 **on each** of 4 TRAIN + ≥3 unseen custom tracks. |
| 04 | Agent can autonomously train/evaluate/improve inside agreed budget; must stop on violations. |
| 05 | 24h active workload execution, max 2 improvement rounds; initially 2 workers, 4 GiB process-tree RSS/job, ≥8 GiB free RAM; 4 workers only after benchmark + admission PASS. |
| 06 | Full RL engineering allowed (PPO, reward, curriculum, observation/action, vehicle/physics) with impact analysis, regression and compatibility review. |
| 07 | Only Pure PPO steering/throttle/brake may count toward finish-rate acceptance. |
| 08 | Agent generates 3 fresh acceptance tracks **after** model freeze; may not learn/tune from those outcomes. |
| 09 | Custom suite covers full Track Builder: turns, obstacles, ramp, bridge, intersection, elevation. |
| 10 | At budget cap stop training, retain best development model, prepare honest demo, report `NOT_ACCEPTED` if any gate fails. |
| 11 | Press Run = live, step-driven inference/physics/web state; no recording or replay. |
| 12 | Agent may commit, push and open PRs. **Merge into main requires owner approval.** |
| 13 | Real-time target 1×; ≥30 frontend FPS; p95 state-to-render ≤150ms; interpolated rendering only. |
| 14 | Agent advances between milestones only with verified acceptance gates/evidence; stop at BLOCKED, budget limit, new permission or merge. |
| 15 | Local versioned checkpoint/optimizer/RNG/artifact recovery; never pretend a weights-only checkpoint is a resumable training session. |
| 16 | Compatibility gate + selective fine-tune/retrain; old checkpoints must be reaccepted when contract changes. |
| 17 | Exactly three coordinated Markdown documents: roadmap, evidence ledger, handoff. |
| 18 | Store docs in `docs/roadmaps/f1rl-demo/` under Git; large artifacts outside repo at `D:\Projects\.archives\F1-RL`. |

## 2. Non-negotiable execution rules

- **Fail closed.** Status vocabulary: `PENDING`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`, `NOT_ACCEPTED`. Never mark PASS from a test name, author's statement, mock or missing evidence.
- **Source admission first.** Before any real training, record current branch, clean committed HEAD, tree/source hashes, configuration hashes, checkpoint provenance and available disk/RAM. Do not train from `main` if LOCAL_DEMO admission forbids it; do not weaken admission.
- **Do not mutate frozen v3 or scientific registration.** Historical `configs/research/wp3-ppo-tuning-campaign-v3.json` remains immutable (SHA-256 `8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4` at 2026-11-10). New workloads require new versioned config/protocol, independent output roots and honest scope labels.
- **Budget is cumulative ACTIVE execution time**, not a promise of 24h wall-clock continuous operation; measure actual training/benchmark/smoke/eval/runtime-test durations per attempt, including failed attempts. Never reset the ledger after restart. Docs/code authoring time is not charged unless running measured workload. Cancel tasks before reaching cap, allowing bounded cleanup within the safety reserve.
- **Hardware envelope:** start ≤2 independent PPO jobs, `n_envs=1`/CPU, process-tree RSS ≤4 GiB/job and ≥8 GiB host available-RAM reserve during admission/monitoring. Existing worker config uses 8 Torch intra-op + 1 inter-op threads/job; changing that is a new versioned runtime contract. Four workers only after 1-vs-2-vs-4 real benchmark supports it AND host memory/CPU/process limits allow it. RSS polling is a watchdog, not a guaranteed hard cap; abort early if unsafe.
- **No automatic retries or overwrites.** One attempt = isolated root, logs, immutable manifests, checkpoint hashes, explicit status; resume successful work only after independent validation. Partial jobs preserved. True training resume only if model, optimizer, RNG, sampler/environment and step state are sufficient and verified; otherwise create a new attempt.
- **Data isolation:** TRAIN/development seeds and tracks are allowed for iteration. Fresh custom acceptance tracks generated after freeze are evaluation-only, may not be redrawn or reused to optimize the same model. Final scientific HELD-OUT must never be read. No Auto/Hybrid success relabelled PPO.
- **Agent permissions:** create isolated worktrees/branches, commit, push and open PRs; **never merge main**, force push, delete old evidence, clean unrelated worktrees or silently change protected contracts. Source code can evolve; migration is explicit, backward compatibility tested.
- **Evidence:** for each gate, put executable commands, exact absolute artifact paths, SHA-256, source commit/tree, environment version, measured metrics, PASS/FAIL rationale and time charged in the ledger. Update handoff on every stop/transition.
- **Source-of-truth separation:** current `main` may not contain the local parallel implementation. Do not assume a branch has been merged or pushed; check refs and select the implementation worktree explicitly. Roadmap/docs may live on a separate docs PR before owner merges it.

## 3. Dependency graph and canonical states

`M0 Source & simulator admission → M1 Track/physics coverage → M2 Real PPO smoke → M3 Real concurrency benchmark → M4 Full training → M5 Development evaluation & ≤2 improvement cycles → M6 Model freeze → M7 Post-freeze custom suite → M8 70-attempt Pure PPO acceptance → M9 Live web runtime acceptance → M10 Demo packaging & review`

Some frontend engineering in M9 may be developed alongside M1–M5 but **M9 PASS** still requires the final frozen model and live acceptance test. Do **not** use final custom acceptance outcomes for another development cycle.

| Gate | State at document creation | Required predecessor | Key outcome |
|---|---|---|---|
| M0 | PASS | None | Clean source, runtime/test paths and budget provenance verified on 2026-11-10 |
| M1 V5 feasibility | PASS | M0 | One custom layered crossing route completes under Auto Drive; no PPO claim |
| M1 full coverage | PASS | M0 | Versioned development geometry fixtures, Physics/reference-controller finishes, bounded API/WebSocket E2E, regressions and controlled V5 benchmark |
| M2 | IN_PROGRESS | M0–M1 | Fresh actual PPO smoke required on current clean source |
| M3 | BLOCKED_ON_M2 | M2 | Fresh controlled real benchmark required due historical profile hash mismatch and source-admission prerequisite |
| M4 | PENDING | M3 | Authenticated real PPO training and validated artifacts |
| M5 | PENDING | M4 | Development-evidenced Pure PPO candidate selection |
| M6 | PENDING | M5 | Immutable model/source/contract freeze |
| M7 | PENDING | M6 | Three newly generated custom eval tracks sealed |
| M8 | PENDING | M7 | ≥8/10 each of ≥7 tracks, no PPO fallback |
| M9 | PENDING | M8 | Causal live simulation 1×, ≥30 FPS, p95 ≤150ms |
| M10 | PENDING | M9 | Reproducible demo handoff + PR, owner merge pending |

### M0 — Repository, authority and environment gate

- [x] Recheck `main`, feature branch, worktrees, local changes, upstream sync, source admission and source commit/tree (2026-11-10 audit; main `ab8951a9c68f009d7b2a14127b13c07426a2460a`, clean execution worktree).
- [x] Verify current Physics V4/SAT/Track Builder/PPO implementation & tests; classify any missing external WP3 evidence separately rather than silently skipping failures. Seven scoped WP7 nodes are BLOCKED_EXTERNAL_EVIDENCE; scientific Gate 2 remains BLOCKED/SEALED.
- [x] Find/verify real runtime commands from `--help` and source; confirm the documented parallel-v1 branch exists locally/remote and review committed diff. `wp3_campaign.py` exposes sequential run/resume; the real benchmark CLI exposes smoke/sequential/parallel; source/docs differences were recorded. No run was started.
- [x] Verify Python/venv, Torch, CPU thread policy, ≥8 GiB free RAM reserve, disk space and archive root permissions; verify output roots outside repo. Python 3.12.15 / Torch 2.15 CPU; 24.8 GiB free RAM at snapshot; outputs outside repo.
- [x] Import verified prior runtime charges; ledger starts at 1,263.031 seconds used / 86,400. M1 feasibility wall-time charge is not captured and remains an accounting caveat.
- [x] Run focused non-training regression, record failures by source/environment/scientific policy. 337 passed, 1 skipped; local API/frontend health 200; live browser rendering not run. Seven external WP7 engineering evidence nodes remain BLOCKED_EXTERNAL_EVIDENCE and scientific Gate 2 remains BLOCKED/SEALED.
- **PASS (2026-11-10):** clean admissible source and executable test/inference paths; immutable baseline and budget provenance recorded. Detailed evidence is in the ledger.

### M1 — Simulator, Track Builder and training coverage gate

- [x] Inspect route graph, directed checkpoint/finish, lane geometry, left/right steering convention, road width/boundaries, collision contacts, object placement, height/seams/overpasses and bridge intersections. Geometry tests cover route/world adaptation, ramp and bridge heights, obstacle placement, collision/contact and explicit rejection of unsupported crossings.
- [x] Demonstrate one complete custom V5 grade-separated crossing/intersection route under Auto Drive. This is feasibility evidence only.
- [x] Add versioned development fixtures for a grade-separated intersection/overpass and a combined left/right/S-turn route with cone, barrier, ramp/elevation and checkpoints. Keep these TRAIN/development-only; do not add final acceptance tracks or access HELD-OUT.
- [x] Validate route reachability by geometry/physics constraints, spawn position, directed finish crossing and reset reproducibility; use a reference controller *only for track feasibility*, **never as PPO success evidence**. Left/right TRAIN-v2 each finished in 497 steps with ordered checkpoints and exact replay.
- [x] Add separate versioned **TRAIN/development** fixtures without changing the official TRAIN roster or producing custom final acceptance tracks. The fixture schema is `f1rl_m1_development_geometry_v1`, version 1, scope `TRAIN_DEVELOPMENT_ONLY`.
- [x] Verify geometry, Physics profile, ordered checkpoints, collision-free reference-controller finishes; harden API/WebSocket E2E with bounded receive/wall timeout, progress diagnostics and cleanup. Run affected Python and Physics V4/PPO contract regressions, web tests and production build.
- [x] Benchmark V5 on a low-load host; capture wall/CPU, Physics-step mean/p95 and realtime factor. Physics stepping is the measured bottleneck; no optimization was justified or made.
- **M1 PASS / READY_FOR_M2 (2026-11-11):** all M1 engineering acceptance checks pass. V5 remains custom Auto Drive only; its crash/roll validators remain fail-closed, and checkpoint probes establish interface shape/action bounds only. This status is not final demo acceptance.

**M2/M3 continuation audit (2026-11-11):** the historical M2 smoke hashes and configuration were verified, but its source is `6452517e3e887ee83f9d98b3323e3e5089d4c187`, not the current source. A fresh current-source M2 attempt (`smoke-20261110T230720Z-524ac611`, source `1404e8e`) failed before learning: process-count watchdog termination; `state.json` reports 0 timesteps and 0 optimizer updates. Its `result.json` SHA-256 is `76a5663f7596e3613e94cb983b338d6147002dda50121ed0eb4bf8b244b5493b`. Diagnosis reproduced the process-count excess in dry-run and traced it to nested Python/Git launcher processes. A no-training dry-run using direct base Python plus direct Git executable reached `dry_run_pass` with max process count 2 under the unchanged profile. The next M2 attempt will use that explicit runtime environment and a new immutable attempt directory. The initial M3 preflight also started no PPO child and exited 1 after 2.268 seconds; raw record SHA-256 `50ef77509963ee872aa5df7eae30e1dd7705f5493b6da2bfd9c90935e5e91284`. Historical M3 attempts bind profile SHA `1b9f…`, unreconciled with their claimed source/config; their timings remain historical only.

### M2 — One-model **real PPO smoke** gate

- [x] Run LOCAL_DEMO smoke for `C-seed0` on the current clean committed source, **256 timesteps / one rollout / CPU / n_envs=1**. Current-source attempt `smoke-20261110T234203Z-2e3f4744` passed at source `50c76513`.
- [x] Confirm `PPO.learn()` execution: 10 optimizer updates, finite nonzero gradients, changed weights, reloadable final checkpoint, current smoke-profile SHA, exact checkpoint/telemetry SHA matches, cleanup verified and HELD-OUT access false.
- [x] Do NOT call frozen full `parallel-run` for a 256- or 4,096-step smoke: it enforces full **65,536-step C0 engineering gate**.
- **PASS:** current-source smoke `smoke-20261110T234203Z-2e3f4744` completed 256 transitions and 10 optimizer updates, wall 23.375s, profile SHA `3d078a4bc45718407fb8a58cdb2f2dd856df101510deb6c9327ebff61c43ee36`. Full attempt hashes and independent validation are recorded in the evidence ledger. Earlier failed/historical attempts remain preserved.

### M3 — Controlled **real PPO concurrency** benchmark

- [x] Use the separate versioned engineering benchmark profile of **4,096 timesteps/job**, rollout=256, CPU, `n_envs=1`; this is not frozen v3 campaign completion.
- [x] Run the same two jobs/seeds/configs sequentially (1 worker) versus concurrently (2 workers) under current committed source with isolated artifacts. Both were source-bound; sequential 2×4096 completed in 668.718s and parallel 2×4096 in 340.625s, with process/resource cleanup and integrity checks passing.
- [ ] Test 4 workers **only after** 2-worker safety/throughput evidence and fresh admission; use comparable workloads, declare any thread-budget changes, assess oversubscription and memory reserve.
- [ ] Choose fastest **safe measured** configuration, not highest count; reject performance claims based on synthetic/mock subprocess timing or vectorized-env benchmark (different experiment).
- [ ] Reserve sufficient remaining 24h budget for M4–M9; if estimated training cannot fit, prefer a versioned smaller development plan or stop for owner review—do not silently start full campaign.
- **PASS (2-worker scope only):** current-source sequential and parallel results use profile SHA `f3f3cad13eb6b8f717a93584b4cd5551717405a15584a26c9eb62fb34821ec5f` and source `50c76513`. Four-worker benchmarking was not performed and remains unadmitted. Historical mismatched-profile attempts remain excluded.

### M4 — Real PPO training campaign gate

- [x] Choose a versioned M1–M3-informed training plan: `m4-m5-development-v2`, CPU, 65,536 steps each, final and periodic checkpoints, C-seed0 then A-seed0. This is explicitly a bounded development plan, not the full 12-job v3 campaign.
- [x] Complete and independently validate the full C0 engineering gate before A0; both attempts met source, timestep, optimizer, checkpoint, telemetry and resource checks.
- [x] Use one worker, isolated attempts, resource watchdogs, no retries; preserve the failed pre-training v1 harness attempt.
- [x] Capture actual learning telemetry/checkpoints over the four registered TRAIN tracks; no acceptance tracks or HELD-OUT data used.
- [x] Include `left-turn-train-v2` in the source-bound M5 development evaluation roster. It was evaluation-only; it was not part of PPO training transitions.
- **ENGINEERING PASS (bounded scope):** two real PPO models completed the declared development plan. This is not completion of frozen v3's 12-job campaign and does not imply policy competence.

### M5 — Development evaluation and improvement gate (maximum 2 cycles)

- [x] Evaluate Pure PPO stochastically on predeclared four TRAIN tracks plus `left-turn-train-v2` (`TRAIN_DEVELOPMENT_ONLY`), 10 fixed-seed episodes per model/track; report per-track finish rate, route progress, collisions, timeout, steering and timing metrics. No intersection variant was included in this evaluation.
- [x] Diagnose behavior from observations, raw/applied actions, rewards, collisions, Physics and terminal telemetry using paired deterministic/stochastic probes; evidence and limits are in the M5 diagnosis report.
- [x] Audit Right Turn V1/V2 geometry, road edges, navigation, spawn, gates, steering convention and Physics V4 runtime behavior; preserve both assets. A separate Right Turn V3 design proposal is recorded; no asset or contract was changed.
- [ ] Agent may edit PPO hyperparameters, reward, curriculum, observation/action mapping, vehicle control or physics; **before each edit** write impact graph, contract/version delta, checkpoint compatibility classification, targeted regression and migration/retrain decision.
- [ ] Run at most **two complete improvement cycles** (change → smoke/regressions → real train/fine-tune → dev eval). Each consumes remaining active-runtime budget. Never call model compatible solely because the .zip loads.
- [ ] Choose a candidate meeting declared development criteria including left turns and multi-feature development routes. Neither candidate met the evidence bar: C-seed0 mean finish rate 0.62 over five cells, with 0/10 right-turn and 1/10 ramp; A-seed0 mean finish rate 0.00 across all five cells.
- **FAIL (candidate gate; diagnosis complete):** 100/100 scheduled M5 episodes completed with valid harness records, but neither candidate meets development criteria. Paired diagnosis explains the main A0 and C0 failures; it does not change the predeclared scores. No improvement cycle, model freeze or custom acceptance track creation has occurred. Remain `NOT_ACCEPTED`; review the Cycle 1 proposal before any training.

### M6 — Model + runtime contract freeze

- [ ] Freeze exact model checkpoint SHA-256, source commit/tree, PPO hyperparameters, reward, feature names/order and normalization, action units/ranges, Physics V4/SAT settings, track-builder semantics, dependency versions and deterministic inference settings.
- [ ] Run compatibility gates (architecture + semantic + checkpoint IO + authoritative source/track hashes); final frozen policy must drive steering/throttle/brake end-to-end.
- [ ] Write `MODEL-FREEZE.json` under versioned external artifacts with immutable artifact inventory, timestamps and all reference hashes. Ledger links to it; handoff records freeze identity.
- [ ] From now through acceptance **no code/model/config tuning** using custom acceptance outcomes. If model or inference semantics change, this freeze is invalid and a **new** acceptance suite must be generated.
- **PASS:** independently re-loadable frozen model and signed/hash-checked runtime contract. This is the **temporal boundary** before generating evaluation tracks.

### M7 — Post-freeze custom demo acceptance suite

- [ ] **After M6 PASS**, generate ≥3 new tracks via the production Track Builder or equivalent same serialization and runtime validation. Fixed generator revision and predeclared seeds; no reuse from any TRAIN/dev trace or track library.
- [ ] Pre-register feature matrix across ≥3: left/right/S, cone/barrier, ramp, bridge, intersection, elevation; preserve all feature occurrences, geometry, spawn/checkpoints/finish and validation outcomes.
- [ ] Validate physically realizable geometry **without running the frozen PPO on it**. If a track is geometrically invalid, discard with reason **before seeing any model performance**; log generation attempts and deterministic replacement rule.
- [ ] Freeze exact track JSON/content hashes, intended episode reset seeds and exactly 10 scheduled attempts per track in `DEMO-ACCEPTANCE-PLAN.json`. Do not change roster after viewing PPO results.
- **PASS:** three sealed, feasible, new evaluation-only custom tracks created **after** model freeze, fully hash-bound to model and plan. Never access scientific HELD-OUT.

### M8 — Final Pure PPO acceptance (TRAIN + custom)

- [ ] Pre-register **10 varied, deterministic reset seeds per track** for all 4 TRAIN + 3 custom; verify randomness/perturbation actually creates distinct trials where supported. Inference policy deterministic/stochastic choice is fixed **before** evaluation and logged; no cherry-picked seeds.
- [ ] Run **all 70 attempts** (more if more tracks), every attempt with current inference/physics state and Pure PPO actions; no reset-on-failure to erase a FAIL, no Hybrid/Auto fallback, no track redesign after seeing outcomes.
- [ ] Store per-attempt result including seed, timestamps, model/track/physics SHA, checkpoint/finish crossing, whether any assist/override occurred, collision/off-road/timeout, actual steering/throttle/brake and telemetry.
- [ ] Require **≥8 valid finishes out of 10 on *each* track**; invalid attempts must remain visible and be treated conservatively (not silently removed/replaced). Show per-track exact numerator/denominator plus total counts.
- [ ] A FAIL/NOT_ACCEPTED **does not authorize** training on these acceptance tracks. Preserve results, report failure. Later model revision requires a newly frozen model **and independent newly generated acceptance suite**, with owner-approved continuation if 24h exhausted.
- **PASS:** exact per-track 8/10 threshold, all provenance and integrity checks, no inference overrides, no acceptance leakage.

### M9 — Live frontend/backend **real-time** demo gate

- [ ] Track Builder → validate/save → click **Run** → spawn on correct starting surface → initialize frozen PPO, observation, controller, Physics V4 → step authoritative simulation continuously → stream authoritative sequence-numbered car/state snapshots to frontend.
- [ ] For every authoritative decision step log `track_id`, `step_id`, `sim_time`, observation hash, PPO action, applied action, physics result, checkpoint/finish/terminal state, monotonic emission order. Do not use prerecorded path, replay buffers or Auto Drive to steer the car.
- [ ] The browser renders new state as it arrives. May interpolate **display only** between real snapshots; cannot fabricate physics, checkpoint progression or steering. Keep UI visually stable (no map jitter, no wrong-lane anchoring) and show speed/steer/brake/telemetry.
- [ ] Verify pause/restart, new custom track loading, crash/off-road/timeout, finish, post-finish state, backend disconnect/reconnect, frontend stale-packet handling, monotonic state and no ghost car/duplicate simulation loop.
- [ ] On target presentation hardware, instrument **1× real-time** pace (simulated dt / wall time), **frontend ≥30 FPS**, **p95 backend-state-to-painted-frame ≤150ms** over a representative ≥10-minute interactive soak/demo rehearsal, with synchronized timestamps/clock offsets and raw per-sample evidence. Report average/p95/max and dropped/out-of-order updates. No prerecorded video backup or replay accepted as the main demo.
- [ ] Reconcile browser-visible position with authoritative backend telemetry; prove policy action-to-next-state causality on new Track Builder route. Browser/integration tests required, not only unit tests.
- **PASS:** live Pure PPO response, correct 1× state evolution, latency/FPS goals, zero replay/fallback and resilient interactive state transitions.

### M10 — Completion packaging, independent review and Git delivery

- [ ] Make repeatable local demo launch/stop instructions; verify clean startup, freeze checkpoint load, Track Builder→Run→finish, fallback **error reporting** (not fallback controller), and short 10-minute presentation script.
- [ ] Produce `DEMO-ACCEPTANCE-RESULT.json`, timing/benchmark tables, per-track confusion/failure breakdown, frozen model/track manifests and live session log. Freeze final hashes.
- [ ] Check all gates with evidence ledger links; refresh this roadmap and handoff. Label final `DEMO_READY` **only if M0–M9 PASS** and 7/7 per-track acceptance PASS; else `NOT_ACCEPTED` and list deficits.
- [ ] Agent may commit, push and open focused PR(s) with reproducible evidence and red/green test summary. **No merge into main until explicit owner approval.**
- [ ] Conduct independent read-only review after agent self-tests. This review may accept the engineering demo but **cannot authorize WP7/Gate 2 or scientific HELD-OUT**.
- **PASS:** independently inspectable finished demo package, no unsupported claims, owner decision pending on PR merge.

## 4. Gate transition / failure algorithm

1. Start every session by reading **all three documents**, fetching current source/evidence refs, checking pending PRs and validating handoff against actual repository and hashes.
2. Identify the lowest numbered non-PASS milestone with satisfied dependencies. Create isolated branch/worktree for its source changes; log any cross-branch integration prerequisites.
3. Before execution, write `IN_PROGRESS`, acceptance plan and remaining budget to handoff. Obtain measured preflight host resources and training authorization boundaries.
4. Run bounded tests/workload. Collect real logs/artifacts. Set `PASS` **only** if all specific assertions are independently verified, source SHA and evidence manifest exist. Update ledger/handoff together in a Git commit.
5. If failure is fixable within scope/budget, create versioned attempt and retry according to explicit policy; no destructive reset. If contract changes, run compatibility gate and regressions first.
6. On `FAIL/BLOCKED`, cap breach, source discrepancy, evidence corruption or new permissions requirement: **stop**, preserve state and artifacts, document exact next command and owner action needed. Do not advance milestone.
7. On workload budget exhaustion: stop all new train jobs, gracefully terminate active ones, preserve best dev-approved checkpoint, label `NOT_ACCEPTED` if not all required gates pass; package demo *with disclosed limitations*, not fake PASS.
8. Owner approval is required for any `main` merge, budget extension, scientific protocol change or opening HELD-OUT.

## 5. Mandatory evidence contract

For **each milestone** the evidence ledger must contain:
`milestone_id`, `attempt_id`, `status`, `scope`, `source_branch`, `source_commit`, `source_tree_hash`, `config_hashes`, `model_hash`, `track_hashes`, `host_snapshot`, `commands`, `exit_codes`, `tests/results`, `metric_definitions`, `measured_values`, `artifact_paths`, `artifact_SHA256`, `time_charged_seconds`, `cumulative_seconds`, `blockers`, `reviewer`, `verified_at`.

Tests are not interchangeable with training: tag every item `STATIC`, `MOCK`, `REAL_SMOKE`, `REAL_TRAIN`, `REAL_EVAL`, `LIVE_BROWSER` or `INDEPENDENT_AUDIT`. Never claim mock timings as PPO speedups.

The ledger stores only lightweight metadata/links. Actual multi-GB artifacts belong in `D:\Projects\.archives\F1-RL\demo-completion\<campaign-id>\<milestone>\<attempt>\` with immutable hashes. No large binaries, sensitive credentials or scientific HELD-OUT data in Git.

## 6. Stop conditions and honest terminal verdict

- `DEMO_READY`: M0–M9 PASS, ≥8/10 on **each** 7+ tracks, Pure PPO end-to-end live runtime targets PASS, evidence complete.
- `NOT_ACCEPTED`: budget exhausted, no valid model, any per-track finish threshold misses, live FPS/latency below target or provenance incomplete. Preserve highest-ranked development model and explain visible demo limitations.
- `BLOCKED`: external missing evidence/permission, invalid source compatibility, unsafe memory/disk, orphan child, disputed scientific isolation, or unresolved engineering regression. Explain and do not silently bypass.
- `READY_FOR_REVIEW`: agent implementation/tests finished, independent reviewer has **not** independently signed off.
- **Neither** `DEMO_READY` nor `READY_FOR_REVIEW` means permission to merge main; **neither** opens scientific HELD-OUT.

## 7. Agent bootstrap

Read in this order:
1. `docs/roadmaps/f1rl-demo/F1-RL-DEMO-COMPLETION-ROADMAP.md`
2. `docs/roadmaps/f1rl-demo/F1-RL-DEMO-EVIDENCE-LEDGER.md`
3. `docs/roadmaps/f1rl-demo/F1-RL-DEMO-AGENT-HANDOFF.md`

Then independently verify repository HEAD/worktrees, existing source admission policy, protected research registration, relevant tests and external artifact inventory. Do **not** assume docs-branch availability in other worktrees before the owner merges the documentation PR. Source and GitHub refs must be re-resolved at the start of each session.

**Current baseline note (documentation creation only):** remote `main` reported `3683981d99f70e1dca2bf600a7d0db93a2280ab9` on 2026-11-10. A separate, not-yet-merged parallel implementation was previously audited at local commit `4c8808f39e1cfdbd71c38b26ac5892c7a0719a81`, with 63 targeted and 3 evaluation/report tests passing. This is **historical code/mock evidence only**. No PPO real smoke, concurrency training, 70-run acceptance or live-performance pass is claimed.

## 8. M5 final hardening & improvement readiness (2026-11-11)

- **Gate:** engineering fixes and source/evidence checks PASS; candidate remains `M5 CANDIDATE NOT_ACCEPTED`. Do not label `DEMO_READY`.
- **Source-lock:** portable source revision V1 and historical registration are preserved. Revision V2 is non-authorizing and binds 44/44 source Git blobs at `d34a58341c55540d37d8743621be5cc2b4154bec`. Four restored STEP3 witness files match their restore manifest at source commit `6e30ebd78b010d3f20d120b936c0283eab6ca78a`.
- **C0 evaluation:** Pure PPO Right Turn V2 passes 10/10 in both deterministic and stochastic diagnostic modes. Ramp deterministic is 10/10; registered stochastic is 1/10, with nine 500-step simulation limits and no collisions or wall timeouts. The 10 deterministic Ramp episodes yield identical actions/outcomes/metrics, so they do not establish seed robustness.
- **Ramp telemetry:** stochastic applied controls have mean throttle 0.660, brake 0.079 and steering −0.013; step-level population SD is 0.406, 0.216 and 0.677 respectively. Throttle and brake are simultaneously above 0.05 on 13.4% of steps. Mean speed is 2.445m/s versus 7.682m/s deterministic. This provides evidence consistent with stochastic pedal conflict and insufficient route progress, not proof of one exclusive cause.
- **Evaluation contract:** the original M5 plan registered stochastic Pure PPO only. Deterministic paired probes are diagnostic. Proposal thresholds explicitly marked unapproved were not retroactively used. The frozen historical registration is not rewritten.
- **Right lane fix:** Auto Drive right offset is capped at 3.5m inside the V2 4m hard corridor; integration and focused unit regressions pass. This is not PPO evidence.
- **Improvement Cycle 1 proposal (not started):** create a new versioned development roster preserving V1 as a legacy regression and evaluating V2 as the development turn; first run a non-training action/reward ablation for throttle-brake overlap and ramp progress. If a new training run is separately authorized, change one supported cause, preserve the original stochastic seed roster for comparison, and apply the existing per-track 8/10 demo bar where applicable. Deterministic runs remain supplementary. Risks include suppressing necessary braking, reward/curriculum overfit and invalid roster changes.
- **Evidence package:** `D:\Projects\.archives\F1-RL\demo-completion\m5-engineering-fix-20261111\final-hardening`; see Evidence Ledger §0.6 for exact hashes and commands.
- **Restrictions honored:** no training, M2/M3 rerun, M6+, U-turn, HELD-OUT access, frozen asset/checkpoint edits, Physics V4 changes or main merge.

## 9. M5 Improvement Cycle 1 — Ramp stochastic ablation (2026-11-11)

- **Status:** development diagnosis complete; `M5 CANDIDATE NOT_ACCEPTED`; do not label `DEMO_READY`. Cycle 1 changed only the loaded PPO policy's learned `log_std` for ablation. No gradient update or PPO fine-tuning was run, and C0's original checkpoint remains unchanged.
- **Root cause evidence:** baseline Ramp V1 stochastic Pure PPO finished 1/10 on the fixed M5 roster; 9/10 reached the unchanged 500-step simulation limit with no collision. The raw/applied action mapping was exact. Throttle/brake overlap and large sampled steering variation were present. Changing only the policy sampling standard deviation changed the result on the same seeds, isolating stochastic policy variance as a causal contributor; it does not prove that variance is the only contributing factor.
- **Candidate std×0.5:** Ramp 10/10 stochastic (1/10 baseline), 10/10 diagnostic deterministic; Right Turn V2 10/10, Straight 10/10, Left Turn V2 10/10 stochastic. Slalom stochastic was 9/10 versus the original 10/10 on the same roster. Slalom deterministic was 0/10 in the diagnostic run; the original registered M5 contract is stochastic-only, so deterministic results are reported separately and do not replace that score.
- **Second ablation std×0.65:** same Ramp/Slalom seed roster, no training; Ramp 10/10 stochastic and Slalom 6/10 stochastic. This did not restore Slalom and was not expanded to other tracks.
- **Acceptance interpretation:** the existing final-demo bar is ≥8/10 per track, but the Cycle 1 candidate is not declared accepted on that basis: Slalom regressed from its fixed-roster historical 10/10, the proposal threshold does not redefine the registered M5 contract, and the candidate is not frozen. Preserve both results and require owner review for any next model selection step.
- **Fine-tuning decision:** not run. The non-training ablation resolves the Ramp failure under the fixed contract, and further training is not justified by this evidence alone given the Slalom trade-off. No additional training cycle was opened.
- **Validation:** std×0.5 evaluated 100/100 cases (five development tracks × ten seeds × deterministic/stochastic); std×0.65 evaluated 40/40 cases (Ramp and Slalom × ten seeds × deterministic/stochastic). Both used Pure PPO, Physics V4, original action/reward/curriculum/step limit, no Auto/Hybrid/action override, no wall timeout and no worker failure.
- **Evidence:** `D:\Projects\.archives\F1-RL\demo-completion\m5-improvement-cycle1-20261111`; report/plan/checkpoint/raw trace hashes are inventoried in `SHA256-MANIFEST.json` (file SHA-256 `7ce4e93f948f7115857fb8aadf732be383e1dc6018469783224548c060c09e75`; 308 artifacts). See Evidence Ledger §0.7.
- **Restrictions honored:** no HELD-OUT access, M2/M3 rerun, M6+, U-turn, Physics V4 or contract changes, historical asset/checkpoint edits, or `main` change/merge.
