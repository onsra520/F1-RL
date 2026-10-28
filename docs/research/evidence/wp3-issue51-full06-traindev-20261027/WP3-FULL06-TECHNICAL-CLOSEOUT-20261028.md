# WP3 Issue #51 - FULL-06 technical closeout (2026-10-28)

## Reproducible technical acceptance

- Owner-authorized TRAIN-development run: WP3-OWNER-TRAINDEV-20261027-FULL-06
- Evaluation source: 6c479540134d98d7d27331879c926c8ba82c2965 (frozen technical tag wp3-physics-v4-airborne-road-6c47954-20261027).
- Frozen TRAIN provenance: fe83f517479f3b990028d6409304fae83b8199a9; NO retraining, original source and FULL-05 unchanged.
- Same original TRAIN seed/track/stratum roster: 12 checkpoint models x 4 tracks x 2 strata x 60 episodes.
- Completed on 2026-10-27T23:53:10.2357118+07:00 (Asia/Ho_Chi_Minh); controller exit 0.
- **PASS: 96/96 cells; 5760/5760 valid episodes; 0 invalid; 0 unattempted; 0 in-flight.**
- Full repository regression on frozen source: 2947 PASS / 3 SKIP / 0 FAIL, exit 0. Historical seeded replay R3 7/7, R4 1/1, R5 1/1 valid.
- Output and evidence: 7/7 registered output hashes and 96/96 JSONL cell hashes verified; 110/110 original run files copied with matching SHA-256. Git committed source/result blob hashes sampled and matched original bytes.
- Original input manifest SHA-256: 4c62ee924951f8b61d85f86c98090f95b77026bb2bdedb23958ad7c80b895701
- Original final output manifest file SHA-256: 1bc69c491e369cd07d2b464e8d2f4bb51914081b0825aee82fc4e5418c961e13
- Engineering source PR: https://github.com/onsra520/F1-RL/pull/55 ; stacked on previous WP3 PRs; **main not merged**.

## All terminal outcome categories

| Category | Count | Fraction of 5760 |
|---|---:|---:|
| finish | 1550 | 26.91% |
| physical_crash | 1253 | 21.75% |
| hard_route_violation | 1256 | 21.81% |
| step_limit | 1700 | 29.51% |
| tilt_envelope_exit | 1 | 0.02% |

IMPORTANT: 5760 formally valid episode records do NOT imply 5760 successful driving finishes; total finishes are 1550 (26.91%).

## Descriptive performance by PPO candidate

| Candidate | Finish /1440 | Finish rate | Seed rates | Seed mean | Seed SD (n=3) |
|---|---:|---:|---|---:|---:|
| A | 611/1440 | 42.43% | 0.00%, 58.54%, 68.75% | 42.43% | 37.10 pp |
| B | 427/1440 | 29.65% | 0.00%, 60.00%, 28.96% | 29.65% | 30.01 pp |
| C | 512/1440 | 35.56% | 48.96%, 57.71%, 0.00% | 35.56% | 31.10 pp |
| D | 0/1440 | 0.00% | 0.00%, 0.00%, 0.00% | 0.00% | 0.00 pp |

All candidate statistics are **descriptive only** and use just 3 independently trained seeds per candidate. Repeated evaluation episodes do not constitute 60 independent train seeds. Do not assert statistical significance, causal reward effects, or robust superiority.

## Descriptive performance by track and stratum

Each row pools 12 models x 60 paired evaluation episodes = 720 valid episodes.

| Track | Stratum | Valid | Finishes | Finish rate | Crash | Hard-route | Step limit | Tilt | Mean route progress | Mean checkpoints |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| straight-sprint-v1 | nominal | 720 | 360 | 50.00% | 0 | 194 | 166 | 0 | 0.561 | 1.017 |
| straight-sprint-v1 | randomized_train | 720 | 358 | 49.72% | 0 | 196 | 166 | 0 | 0.559 | 1.011 |
| slalom-v1 | nominal | 720 | 168 | 23.33% | 277 | 48 | 227 | 0 | 0.474 | 1.215 |
| slalom-v1 | randomized_train | 720 | 186 | 25.83% | 264 | 50 | 220 | 0 | 0.483 | 1.258 |
| right-turn-v1 | nominal | 720 | 0 | 0.00% | 354 | 193 | 173 | 0 | 0.304 | 0.517 |
| right-turn-v1 | randomized_train | 720 | 0 | 0.00% | 356 | 194 | 169 | 1 | 0.303 | 0.508 |
| ramp-v1 | nominal | 720 | 239 | 33.19% | 1 | 197 | 283 | 0 | 0.489 | 0.854 |
| ramp-v1 | randomized_train | 720 | 239 | 33.19% | 1 | 184 | 296 | 0 | 0.491 | 0.854 |

Per-model 12-seed finish rates and further secondary metrics (duration, steering RMS, collision, hard-route, timeouts, route progress and checkpoint counts) are recorded without aggregation loss in WP3-TRAIN-EVALUATION.json and WP3-CANDIDATE-COMPARISON.md, with 96 source-bound individual cell JSONL files.

## Issue #51 scope resolution and active holds

- Evaluator reporting NameError, reporter writer regressions, fail-closed terminal metadata, crash and impact physics, and all new test/replay work: **technical verification COMPLETE** in stacked source PRs.
- Original canonical campaign recovery: **BLOCKED** on unavailable historical smoke/preflight prerequisites; do not fabricate, override or retroactively rewrite evidence.
- FULL-06 original-seed TRAIN-development engineering Eval: **COMPLETE, 5760/5760, zero invalid**. Fresh versioned run, not a recovery/relabel of FULL-05.
- Independent Physics v4 scientific signoff: **NOT PROVIDED**. Scientific registered Eval and HELDOUT/WP7/Gate 2: **BLOCKED_PENDING_INDEPENDENT_REVIEW**. No scientific certification or inferential superiority claims.
- Leave GitHub Issue #51 OPEN until independent provenance and Physics review and any scientifically registered Eval requirements are explicitly authorized and resolved. This closeout does not replace reviewer signoff.

The complete original FINAL-STATUS, STATISTICAL-READINESS, TRAIN-EVALUATION, 96 cell JSONL, console, controller receipts and source/output hash manifests are stored adjacent to this file.
