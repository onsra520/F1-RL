# WP7 extended engineering regression — follow-up to Physics v4 main-task closeout

**Date:** 2026-11-07
**Parent source:** `ab19584fe51b91ea0dcf7ae470ae44cbc4e43039`
**Scope:** TRAIN-only software/engineering regression. No PPO training, held-out geometry, FULL-06 evaluation or hardware calibration.

## Previously unverified test coverage

The extended WP7 tests were run **to completion** in an isolated clean checkout. Candidate files were compared byte-for-byte with the uncommitted files found on the owner's working tree; the owner's working tree was not overwritten.

| New test coverage | PASS | FAIL | ERROR | SKIP |
|---|---:|---:|---:|---:|
| Readiness, admission, execution binding, manifests and CLI compatibility | 66 | 0 | 0 | 0 |
| Continuous Arc Trajectory (all cases) | 15 | 0 | 0 | 0 |
| Center-gate, executed feasibility, spatial execution, ramp and replay | 77 | 0 | 0 | 0 |
| Historical provenance, owner profile, reference benchmarks and robustness | 37 | 0 | 0 | 0 |
| **New WP7 unique testcase IDs** | **195** | **0** | **0** | **0** |

The nine-module execution group took 1037.66 seconds because real simulator/replay cases are resource-intensive. These tests were not skipped or relabeled as PASS; all 77 produced successful pytest results.

### Missing source dependencies corrected

The previous WP7 readiness failures were caused by missing `configs/research/wp7-core-readiness-matrix-v1.json`; four test modules also referenced CLI programs that were no longer located in `scripts/`. The fix adds the versioned matrix, makes those CLI utilities available under `devtools/legacy_f1rl_cli/`, and updates the affected test references. No Physics implementation, track assets, PPO checkpoints, criteria or seed registrations are altered.

The matrix has **nine subsystems, 27 capabilities**; 24 are documented as implemented, three remain explicitly blocked and all nine compatibility gates remain `in_progress`. The new tests check that this state remains **fail-closed**. A green software regression is **not** acceptance of those open capabilities or scientific release.

The fast WP7 tests validating the added matrix/CLI paths are now included in the source-only GitHub CI workflow; its exact-commit result must be obtained independently after committing.

### Evidence references

The original complete JUnit reports are stored outside the repo under `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`:

- `PHYSICS-V4-AB19584-WP7-UNCOVERED-FAST-WITH-CANDIDATE-FIXTURES-20261107.xml`
- `PHYSICS-V4-AB19584-WP7-ARC-NEW-EXHAUSTIVE-20261107.xml`
- `PHYSICS-V4-AB19584-WP7-EXTENDED-EXECUTION-SAFE-20261107.xml`
- `PHYSICS-V4-AB19584-WP7-REMAINING-FIVE-MODULES-20261107.xml`

The accepted Physics v4 main task remains the **owner-approved paired TRAIN v2 engineering scope**. The literal mirrored Left TRAIN v1 remains HELD-OUT-inadmissible, and the original six PPO smoke durations remain `NOT_RECORDED`. Neither historical issue is silently presented as an observed PASS.

**Engineering result:** All 39 WP7 test modules are now covered by executed suites, including the bounded TRAIN-5 tests previously reported. This does not mean all underlying WP7 scientific/compatibility gates are complete.
