# F1-RL — Demo Evidence Ledger

> **Document version:** 1.1 (2026-11-10). **Role:** inspectable factual evidence register, NOT a success report.
> **Master contract:** [Roadmap](F1-RL-DEMO-COMPLETION-ROADMAP.md). **Operational resume:** [Handoff](F1-RL-DEMO-AGENT-HANDOFF.md).  
> **Initial global verdict:** `NOT_ACCEPTED / NOT_EVALUATED`. No real PPO training, live demo acceptance or scientific HELD-OUT proof is claimed here.

## 0. M0 reconciled source and engineering evidence (2026-11-10)

- Both documentation PR #60 and parallel-supervisor PR #61 have been merged into `main`. Local and origin main: `ab8951a9c68f009d7b2a14127b13c07426a2460a`; execution worktree: `feat/f1rl-demo-completion` at that baseline.
- **REAL_SMOKE / REAL_BENCHMARK historical attempts were independently sampled and 5/5 model-final.zip SHA-256 digests matched**. Actual PPO source for those attempts: `8582d9d08826074dbd2b6d32609d9d58639f771d`; current merged source differs by a benchmark attempt metadata-only fix (do not relabel old evidence).
- Historical result totals: real smoke 256 steps / 10 optimizer updates; sequential 2 x 4,096 steps, wall 696.250s; parallel 2 x 4,096 steps, wall 367.500s, measured speedup 1.8946x. Single pair only; no full training or final PPO finish evidence.
- Frozen WP3 v3 SHA-256 verified unchanged: `8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4`.
- Fresh M0 preflight: Python 3.12.15, 32 logical CPUs (historical verified), free RAM 27,105,672 KiB, D: free 611,822,104,576 B; no active matching PPO jobs. Measurements are a snapshot, not a guarantee for later training.
- Targeted regression independently rerun after the metadata fix: **68 PASS, 1 SKIPPED**; pytest option `-o addopts=` was required because optional pytest-xdist is absent in that venv. No full-system regression or scientific admission acceptance is claimed.
- Current M0 state: **IN_PROGRESS**, pending additional cross-layer verification and explicit external-evidence classification. M1-M10 remain PENDING.

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

## 3. Milestone evidence register (append/update only with provenance)

| Gate | State | Minimum evidence to link | Actual source SHA | Attempt(s) | Verified by |
|---|---|---|---|---|---|
| M0 Source admission | IN_PROGRESS | git/ref/dirty-state snapshot, tool versions, baseline tests, host/RAM/disk preflight | — | — | — |
| M1 Track/physics | PENDING | feature matrix + valid geometric paths, spawn/finish/bridge/seam/contact regressions | — | — | — |
| M2 Real smoke | PENDING | run/state, weight delta, optimizer/gradients, final checkpoint SHA, 256-step telemetry | — | — | — |
| M3 PPO benchmark | PENDING | same-job 1 vs 2 workers (optionally 4), real CPU/RSS/wall/throughput raw data and cleanup | — | — | — |
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

* **Charge breakdown** (seconds from immutable result.json): smoke attempts 0.016 + 37.453 + 22.328 + 26.438 + 22.328 = **108.563**; sequential **696.250**; parallel **367.500**. Combined **1,172.313 s** (~19m32.313s). All prior attempts, including failed smoke, count; the budget must never reset between sessions.
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
