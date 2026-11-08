# Physics v4 — owner-directed amendment for the main engineering task

**Recorded:** 2026-11-07T20:22:39.208216+07:00
**Parent engineering source:** `d0e61d54232c6e4d92a7ab2484ebf7a8f1c5040d`
**Decision:** Accept the versioned paired Right/Left TRAIN v2 replacement **for Task 0–6 engineering closeout only**.

## Basis of authorization

The project owner explicitly requested this four-part closure workflow on
2026-11-07: approve v2 replacement, normalize missing smoke information,
issue a new correctly named FINAL-REPORT.md, commit and seal the outcome.
This is **in-chat authorization**, not a cryptographically signed owner
attestation or an independent scientific review. Do not represent it as
either kind of signature.

The original plan `PHYSICS-V4-LEFT-TURN-PLAN.md` is a historic
DRAFT; leave its original bytes, older outputs and failed literal v1
assertions untouched. This amendment is a **new versioned engineering
scope decision**; it does not rewrite historical evidence.

## Task-specific amendment

- **Task 2:** Accept the separately registered five-track setup with
  `right-turn-train-v2` and `left-turn-train-v2`.
  The exact reflected Left TRAIN v1 has a HELD-OUT fingerprint collision
  and remains rejected, not passed.
- **Task 3:** Accept two independently controlled actual Physics v4 simulator
  v2 completions (ticks 113, 311, 497, exact replay) as bounded engineering
  feasibility; not a v1 witness or laboratory physical validation.
- **Task 5:** Accept the original six one-shot deterministic frozen-policy
  v2 smoke episodes (6/6 FINISH, 1,117 simulation steps); record missing
  elapsed wall-clock episode duration as `NOT_RECORDED` for each.
  No zero/estimated duration or synthetic timestamp is introduced.
- **Task 6:** Publish a fresh, standalone `FINAL-REPORT.md` in a
  separate dated closeout folder, preserving the original
  OUTPUT_ROOT/FINAL-REPORT.md (2026-11-06) byte for byte.

## Explicit exclusions

No retraining, reseeding, simulated smoke rerun, additional FULL-06,
HELD-OUT evaluation, scientific release, historical evidence replacement,
physics threshold relaxation, new hardware calibration or automatic merge.
The existing scientific admission guards remain fail-closed.

**Disposition:** owner-approved v2 *engineering* amendment with disclosed
historic v1 and elapsed-duration gaps. The unamended original v1 plan is
**not** retroactively 100% PASS.
