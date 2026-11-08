# Physics v4 — final main-task closeout (Task 0–6)

**Date:** 2026-11-07
**Repository:** `onsra520/F1-RL`
**Branch:** `handoff/physics-v4-khanh-bang-20261106`
**Verified source anchor:** `f1b9ae4d2a2dc16a7ecd9698ff31705f01c97932`
**Original historical baseline:** `2417570f32f6879c2651bce37d88fd6ac83a2ea4`
**Decision:** **Main engineering Task 0–6 CLOSED on the versioned paired TRAIN v2 replacement, with the original v1 mirror deviation explicitly disclosed.**

This is a **new, current append-only report**. `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\FINAL-REPORT.md` is the 2026-11-06 historical report and **must not be overwritten**. This closeout covers Physics v4 diagnosis, bilateral executable feasibility, paired frozen PPO smoke and direction-specific retraining recommendations. It does **not** certify scientific PPO training, physical hardware validity or held-out generalization.

## 1. Main task checklist

| Task from `PHYSICS-V4-LEFT-TURN-PLAN.md` | Verdict | Exact meaning |
|---|---|---|
| 0. Preflight and nine-case PPO diagnosis | **PASS** | All nine historic Right/Ramp/Slalom episodes reviewed; provenance and uncertainty recorded in `driving-nine-cases.json` and `driving-diagnosis.md` |
| 1. Historical four-track behavior frozen | **PASS** | 3 original config/manifest inputs, 4 original TRAIN assets and 12 model ZIPs (19 Git objects) unchanged against the base commit; 30 archived handoff SHA-256 receipts verified |
| 2. Separate five-track TRAIN selection | **PASS for separately versioned v2; NOT literal v1** | Exact v1 X-reflection was rejected for HELD-OUT world/content fingerprint collision; unique Right/Left TRAIN v2 source assets and opt-in five-track candidate supersede v1 without modifying frozen TRAIN-4 |
| 3. Independent Left and Right Physics v4 feasibility | **PASS for v2 pair** | Two actual non-PPO control executions and exact action-tape replays: both valid finish at tick 497 with crossings at ticks 113, 311 and 497 |
| 4. Physics v4 and historical compatibility | **PASS for required and inspected scoped tests** | Current original 73/73 PASS, v4-focused 61/61 PASS, optional original WP7 profile tests 24/24 PASS; two old missing-file failures no longer reproduce |
| 5. Paired deterministic PPO smoke | **PASS for v2 pair only** | Original frozen C/B/A policy × Right/Left TRAIN v2: 6/6 FINISH, 1,117 steps; source-locked logs and model hashes; no new smoke replay |
| 6. Final report and direction-specific retraining decision | **PASS — this report** | Separate decisions below; no training is performed or authorized within main task |

**Precise completion accounting:** Seven engineering workstreams are dispositioned under the later source-bound **paired v2 replacement**. The original draft's literal `left-turn-train-v1` exact-mirror acceptance is **NOT 100% PASS**. It was rejected correctly because it conflicts with the HELD-OUT identity. Do not describe `v1` as successfully implemented and admitted; the later `v2` alternative has separate provenance and acceptance.

## 2. Frozen historical evidence and nine-case diagnosis

The historical four-track source and outcomes are preserved. Historical FULL-06 remains **4 TRAIN tracks, 96 cells, 5,760 episodes**. No FULL-06 rerun was performed in this closeout. The previous 30 archived SHA256SUMS entries remained valid.

| Historical TRAIN-4 track | Frozen PPO finish count | Diagnosis |
|---|---:|---|
| `right-turn-v1` | **0/3** | Three certified bounded-road-body crash terminals after checkpoint-0. Existing checkpoint-1/finish local +Z conflicts with the exit route tangent +X. This does **not** show that PPO alone caused failure |
| `ramp-v1` | **1/3** | `C-seed0` finished, `B-seed1` and `A-seed2` ended at certified wheel-hard-stop crashes; no missing wheel details are fabricated |
| `slalom-v1` | **0/3** | Three certified physical crashes with distinct trajectories; cause is not assigned solely to PPO |

Evidence: `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\driving-nine-cases.json` and `driving-diagnosis.md`, including the original JSONL file/line references, step/action/terminal summaries and Phase 2A/2B/2C comparison limitations. The 2026-11-06 `FINAL-GATE-VERDICT.json` remains a historical snapshot, not current status.

## 3. Why v1 was blocked and v2 is the actual accepted implementation

The originally requested exact-mirror `left-turn-train-v1` has **the same world and content SHA-256 fingerprints as the registered HELD-OUT `left-turn-v1` metadata**. This is a genuine independence violation; the v1 opt-in must reject it. HELD-OUT geometry was not opened to discover the collision.

The subsequent independently authored Left TRAIN v2 and paired Right TRAIN v2 introduce distinct registered TRAIN-only geometry and correctly directed exit gates. The old Right TRAIN v1, historical TRAIN-4 configs, original 12 PPO models, Physics engine and terminal/collision thresholds remain unchanged. The v2 evidence applies only to the separate `right-turn-train-v2` and `left-turn-train-v2` tracks.

## 4. Task 3 — independent non-PPO existence witnesses

| Actual v4 controller | Checkpoint-0 | Checkpoint-1 | Finish | Replay |
|---|---:|---:|---:|---|
| `right-turn-train-v2` | 113 | 311 | 497 | Source-bound float32 action tape **exact** |
| `left-turn-train-v2` | 113 | 311 | 497 | Source-bound float32 action tape **exact** |

Each independently generated valid terminal `successful_finish` using the actual simulator step function, Physics `spatial_road_edge_partial_lifecycle_v4`, race `wp7_directed_lifecycle_crash_contract_v4`, crash contract `wp3_physical_crash_terminal_v1`, tilt cap 0.7 rad. Both runs have grounded state, no collision, ordered directed crossings and matched replays. These prove **nominal executable existence**, not arbitrary speed/controller/initialization feasibility.

Immutable witness: `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\STEP3-PHYSICS-V4-DUAL-TURN-ACCEPTANCE.json` and `docs/.superpowers/PHYSICS-V4-DUAL-TURN-FEASIBILITY-20261106.md`.

## 5. Task 4 — exact-source regression

| Test group | PASS | FAIL | ERROR | SKIP |
|---|---:|---:|---:|---:|
| Original 73 required handoff tests | **73** | 0 | 0 | 0 |
| Targeted Physics v4 tests (airborne, contact, hard stop, crash prefix) | **61** | 0 | 0 | 0 |
| Old WP7 profile/seed compatibility tests | **24** | 0 | 0 | 0 |
| Latest scoped integration regression on source anchor | **217** | 0 | 0 | 0 |
| GitHub CI of source anchor (run `42363990745`) | **108** | 0 | 0 | 0 |

New same-anchor JUnit receipts: `HANDOFF-F1B9AE4-PHYSICS-V4-ORIGINAL-73-FINAL-20261107.xml` and `HANDOFF-F1B9AE4-PHYSICS-V4-TARGETED-AND-LEGACY-85-FINAL-20261107.xml`, in the external evidence directory. Previous 22-PASS/2-FAIL compatibility report was historical; on this commit the missing source files are now present, and all 24 tests PASS. These are **scoped** regressions, not a claim that the full repository test suite has been run.

The plan additionally names `tests/research/test_wp3_v4_technical_acceptance.py`: it was **inspected but deliberately NOT RUN** in this closeout. That test binds to machine-specific historical WP3 evidence at `D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1` and assumes a historical dirty source tree. Its distinct owner-scientific signoff is not inferred from the 73+85 current PASS results. This precise exclusion is consistent with the source plan's NOT RUN instruction for missing or unsuitable historical data.

## 6. Task 5 — frozen policy paired deterministic PPO smoke

Original single-use inference source: `01ef1e3e75de8df60fc2159ed70b38d0ee777253`. CPU deterministic, reset seed `20261106`, per-case cap 2,500 steps. **No episodes were rerun for this closeout.**

| Frozen PPO checkpoint | Right TRAIN v2 | Left TRAIN v2 |
|---|---|---|
| `C-seed0` | FINISH, 203 steps | FINISH, 205 steps |
| `B-seed1` | FINISH, 177 steps | FINISH, 177 steps |
| `A-seed2` | FINISH, 179 steps | FINISH, 176 steps |

**6/6 FINISH, 1,117 total steps.** All six have `checkpoint-0`, `checkpoint-1` and valid terminal; original logs, three model ZIP SHA-256 and source commit were independently verified. Left TRAIN v2 is **zero-shot** for these previously frozen policies. The v2 Right finishes are not direct stepwise performance comparisons with historical Right v1 because geometry/gates changed. A single reset seed is **not** multi-seed robustness evidence.

Original sealed smoke: `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\HANDOFF-01EF1E3-TRAIN5-PPO-SMOKE-LOCKED-ACCEPTANCE-20261107.json` and corresponding immutable ZIP.

## 7. Task 6 — per-direction retraining decisions

- **Right TRAIN v2 — NO automatic retraining proposal.** Independent control and 3/3 frozen PPO runs finish at one nominal reset. There is no failure-triggered recommendation here; generalization remains unmeasured.
- **Left TRAIN v2 — NO automatic retraining proposal.** Independent control and 3/3 zero-shot frozen PPO runs finish at one nominal reset. This does not certify trained left-turn ability or future robustness.
- **Historical Right TRAIN v1 — BLOCKED / INCONCLUSIVE causal retraining decision.** Historical 0/3 finishes coexist with gate orientation mismatch, and the original v1 lacks the independent ordered finish witness needed by the source plan. Do not blame PPO alone.
- **Mirror Left TRAIN v1 — NOT ADMISSIBLE.** HELD-OUT metadata fingerprint collision; no accepted v1 smoke/feasibility.
- **Ramp and Slalom historical — retained as diagnosis**; further optimization would be a separately scoped scientific task, not part of this turn-smoke closeout.

## 8. Explicit boundaries and handoff

**Original engineering main task:** `CLOSED_WITH_DISCLOSED_VERSIONED_V2_REPLACEMENT`.

**Literal original v1 draft:** `NOT_FULLY_PASS / SUPERSEDED_WITH_RECORDED_EXCEPTION`; report this explicitly instead of claiming an unconditional 100% on the original text.

**PPO training, fine-tuning, 18 reserved training seeds, 3,600 development evaluation, HELD-OUT, scientific FULL-06 and release:** `NOT EXECUTED / BLOCKED` — separately scoped after Task 6. No hardware validation is claimed.

The old draft also requested no commit while preparing its evidence. The owner subsequently gave explicit instructions to implement and commit on this named handoff branch; this final closeout adds documentation/verification only, not new experimental activity.

**Machine-readable verdict:** `docs/.superpowers/PHYSICS-V4-MAIN-TASK-FINAL-VERDICT-20261107.json`; supporting evidence hashes are stored there. No historical report, control tape, model ZIP or PPO result was overwritten.
