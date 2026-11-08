# FINAL-REPORT — Physics v4: engineering Task 0–6

**Closeout type:** Owner-authorized paired TRAIN v2 scope amendment; engineering only.
**Date:** 2026-11-07
**Repository/branch:** onsra520/F1-RL / handoff/physics-v4-khanh-bang-20261106
**Tested parent source commit:** `d0e61d54232c6e4d92a7ab2484ebf7a8f1c5040d`
**Historical baseline:** `2417570f32f6879c2651bce37d88fd6ac83a2ea4`
**Exact closing Git commit and CI:** recorded in a separately sealed external
  acceptance receipt after commit (avoid circular commit-hash self-reference).

## Executive decision

**Engineering Task 0–6: CLOSED WITH OWNER-APPROVED VERSIONED TRAIN v2
REPLACEMENT AND DISCLOSED EVIDENCE LIMITS.** This is not an unqualified
100% PASS of the original literal v1 mirror plan. Owner direction authorizes
the new v2 engineering scope in chat; no cryptographic owner signature,
independent reviewer signature, or scientific training gate is asserted.

The exact Left TRAIN v1 reflection shares source-registered world/content
fingerprints with HELD-OUT metadata and remains blocked. The independently
authored paired Right/Left TRAIN v2 is the accepted technical substitute.
Historical TRAIN-4 inputs, checkpoint models and the 2026-11-06
FINAL-REPORT.md are preserved.

## Task 0–6 checklist

| Task | Owner-amended outcome | Evidence or limitation |
|---|---|---|
| Task 0: nine episode diagnosis | PASS | 9/9 frozen Right/Ramp/Slalom records, actions/reset/terminals, physical crash vs policy uncertainty |
| Task 1: frozen four-track baseline | PASS | 19 source/config/asset/model objects unchanged; 30 historic handoff SHA records verified; FULL-06 historical only |
| Task 2: separate five-track TRAIN | ACCEPTED V2 REPLACEMENT | Right/Left v2 independent, source-bound; literal mirrored v1 BLOCKED due HELD-OUT identity |
| Task 3: independent v4 turn feasibility | PASS ON V2 ONLY | non-PPO real simulator controls and exact action tape replays, two 497-step valid finishes, each gate at 113/311/497 |
| Task 4: v4 and regression compatibility | PASS IN VERIFIED SCOPE | Original 73/73, targeted 61/61, legacy 24/24, current deep 1,570 unique testcase PASS; site-specific technical test NOT RUN |
| Task 5: paired frozen PPO smoke | 6/6 PASS ON V2 WITH DURATION GAP | original one-shot CPU deterministic seed 20261106, 1,117 steps; missing per-case elapsed wall-clock time disclosed below |
| Task 6: report and retraining decision | COMPLETE WITH EXCEPTIONS | this dated canonical report, direction-specific conclusions and owner scope amendment; no training authorized |

### Task 0: historical nine-case diagnosis

- Right TURN v1: 0/3 finish; accepted bounded-road-body crash outcomes,
  but gate orientation mismatch prevents blaming PPO alone.
- Ramp v1: 1/3 finish, two wheel hard-stop crashes; missing wheel
  measurements are not reconstructed.
- Slalom v1: 0/3 finish, three certified crashes with distinct trajectories.
- Historical FULL-06 remains 4 TRAIN tracks, 96 cells, 5,760 episodes.
  It was not rerun. Phase 2A/2B/2C eight-case terminal outcomes and steps
  agree 8/8 at **episode level**, not necessarily physics substep level.

### Task 3: independently replayed controls

| Paired TRAIN v2 direction | Checkpoint-0 | Checkpoint-1 | Finish | Replay |
|---|---:|---:|---:|---|
| Right | 113 | 311 | 497 | exact |
| Left | 113 | 311 | 497 | exact |

Both witness actual Physics v4 step execution and valid terminal without
PPO, crash or invalid transition. Evidence is one nominal executable
existence witness per direction; statistical generalization is **not**
certified.

### Task 5: six original frozen PPO smoke cases (no rerun)

| Frozen policy | TRAIN track | Steps | Terminal | Checkpoints | Elapsed wall-clock duration |
|---|---|---:|---|---|---|
| `A-seed2` | `left-turn-train-v2` | 176 | FINISH | CP0 → CP1 → finish | `NOT_RECORDED` |
| `A-seed2` | `right-turn-train-v2` | 179 | FINISH | CP0 → CP1 → finish | `NOT_RECORDED` |
| `B-seed1` | `left-turn-train-v2` | 177 | FINISH | CP0 → CP1 → finish | `NOT_RECORDED` |
| `B-seed1` | `right-turn-train-v2` | 177 | FINISH | CP0 → CP1 → finish | `NOT_RECORDED` |
| `C-seed0` | `left-turn-train-v2` | 205 | FINISH | CP0 → CP1 → finish | `NOT_RECORDED` |
| `C-seed0` | `right-turn-train-v2` | 203 | FINISH | CP0 → CP1 → finish | `NOT_RECORDED` |

All six case-level original JSONL SHA-256 values and source identities
were independently rechecked before this closeout. The logged fields
contain per-step action, reward, position, speed, checkpoint and terminal,
but **no elapsed wall-clock timestamp/start/end/duration**. Therefore,
duration is `null`/`NOT_RECORDED` for each episode, not
zero seconds, not simulation steps converted to elapsed seconds, and not
a retrospective guess. See SMOKE-DURATION-DISCLOSURE.json. No original
one-shot smoke was reexecuted.

### Task 4: current verification and source integrity

Previous exact-source source-only CI for parent `d0e61d54232c6e4d92a7ab2484ebf7a8f1c5040d`:
**123/123 PASS**, GitHub run **42371926116**. Previously reproduced
28 FAIL cases were repaired with original WP3 bytes; 6 historical
sources verified byte-for-byte against Git origin. Deep regression
second check: **1,575 test executions, 1,570 distinct IDs, 0 FAIL/ERROR/SKIP**,
with five repeated tests excluded from unique coverage. The original
historical 73, v4-targeted 61 and legacy 24 groups all passed.

The resource-intensive **expanded 32-module WP7 execution/feasibility**
test group was interrupted and has no finished JUnit; it is **NOT PASS**
and is not included in the counts. This report does not claim complete
repository regression, site-specific technical fixture acceptance,
independent physical calibration or real-world performance certification.

### Task 6: retraining recommendation by direction

- Paired Right TRAIN v2: **no automatic retraining recommendation** after
  3/3 successful frozen-policy smoke at a single seed and independent
  engineering feasibility. Generalization remains unproven.
- Paired Left TRAIN v2: **no automatic retraining recommendation** after
  3/3 zero-shot frozen-policy successes and independent non-PPO finish.
  The frozen models were not previously trained on v2 Left.
- Historical Right TRAIN v1: **BLOCKED / INCONCLUSIVE** for causal
  retraining attribution (historical 0/3 with gate mismatch).
- Mirrored Left TRAIN v1: **NOT ADMISSIBLE** due the HELD-OUT metadata
  collision; no accepted v1 independent finish exists.

Any new PPO training, held-out evaluation, performance benchmark or
scientific release requires its own independently approved protocol.
This owner-approved engineering scope amendment is **not** such an
authorization.

## Evidence, final receipt and non-destructive output location

This exact `FINAL-REPORT.md` is published in a unique dated
closeout directory inside the repository and a separate mirrored
evidence output directory. Older `FINAL-REPORT.md` remains
immutable. Owner amendment and six-case normalized smoke manifest are
stored beside this report. After the closing documentation commit and
new exact-commit CI succeed, the independent receipt will record the
final commit SHA and new CI run without altering these already-hashed
source artifacts.

**Truthful final status:** Owner-amended v2 main task technically closed
with documented v1 and missing-duration exceptions; original unamended
plan **not** 100% satisfied; full training/scientific release still closed.
