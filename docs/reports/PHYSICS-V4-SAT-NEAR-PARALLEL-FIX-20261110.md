# Physics V4 SAT Near-Parallel Fix — 2026-11-10

## Verdict

`READY_FOR_REVIEW`

The supplied counterexample is fixed by using the existing Decimal SAT fallback when the float kernel encounters a near-parallel cross axis. No physics thresholds or collision contracts changed. Verification stopped at L2 as requested.

## Root cause and change

`obb_separation_lower_bound` skipped every axis with magnitude below `1e-12`, including nonzero cross-product axes. Such axes can still be decisive after normalization. In the supplied case, the skipped axis caused the reported gap to be `2.221933248726449 m`, incorrectly within the `2.4 m` threshold, while an independent vertex oracle gives `2.5312532935515336 m` (clear).

The kernel still skips exact zero cross axes for identical rotations. For nonzero cross axes below `1e-12`, and zero cross axes where the quaternion rotations are not exactly identical up to sign, it now computes the full Decimal SAT gap and returns that value. Normal queries retain the float fast path; Decimal is not used for every query. The cutoff remains a fallback trigger rather than being removed.

## Files changed

- `f1rl/sandbox/body_collision.py` — near-parallel cross-axis fallback trigger and exact same-rotation check.
- `tests/sandbox/test_body_collision.py` — fixed counterexample regression and seeded near-parallel differential.
- `tests/sandbox/sat_reference.py` — independent Decimal oracle using rotation matrices, all eight box vertices, and vertex projection intervals.

The independent oracle does not use the production support-radius SAT formula.

## Targeted verification

The required order was followed:

- **L0, before fix:** 1 expected failure. Oracle gap `2.531253293551533638... m`; production float gap `2.221933248726449 m`; comparator incorrectly returned within threshold.
- **L0, after fix:** 1 passed.
- **L1:** `python -m pytest tests/sandbox/test_body_collision.py tests/sandbox/test_bounded_patch_clearance.py -q` — **36 passed in 11.70 s**.
- **L2:** seeded vertex-oracle differential — **32 comparisons**, 4 deterministic seeds × 8 thresholds; positive and negative outcomes included, **0 false positives, 0 false negatives, 0 mismatches**, repeated decisions identical.

No Auto Drive/T05, 216-node collision suite, full regression, PPO, HELD-OUT, scientific evaluation, or performance benchmark was run.

## Performance and remaining risk

Ordinary SAT queries retain the existing float path. Only near-parallel cross-axis pairs trigger a full 100-digit Decimal recomputation, so expected cost is limited to rare geometry pairs; tracks with many nearly aligned OBB pairs may see additional CPU cost. No benchmark was run under the task’s policy.

The fallback uses finite-precision Decimal arithmetic and the L2 set is targeted, not a proof over every representable geometry. Extremely close orientations outside the tested ranges remain a numerical risk for review.

Worktree remains on `codex/sat-broadphase-workers` at `74a5147bfc8f907670b45f5fd02315505262813d`. Existing uncommitted changes were preserved. No commit, push, merge, or branch change was made.
