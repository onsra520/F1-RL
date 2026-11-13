# WP3 Issue #51 — First certified roll-envelope exit (26 Oct 2026)

**Scope: 1 exact R4 invalid episode fixed, 7 old hard-stop failures regression-replayed; no Full Evaluation.**

## Frozen source, reproducibility

- Source branch: `fix/wp3-roll-domain-source-certified-20261026`
- Locked source: `fc187879d96d95303308b3607a102049a56cd783` (the correction is isolated on a new Git worktree and pushed to origin).
- R4 immutable failure: `WP3-OWNER-TRAINDEV-20261026-FULL-04`, model `C-seed1`, track `right-turn-v1`, `randomized_train`, episode `10`, environment seed **3886169428**, policy seed **1971191095**.
- The original `SuspensionResolutionError`: roll 0.704367412 rad exceeded an unchanged 0.7 rad physics-domain bound. Original R4 remains **INCOMPLETE_RANKING_WITHHELD**.
- All replay runs use the original owner-approved TRAIN pair/roster and existing frozen PPO ZIP checkpoints. Source/checkpoint provenance digests verified. HELD-OUT never accessed and no model trained.

## Diagnosis: no actual chassis/road crash

At roll overstep, the oriented rigid chassis is **still geometrically separated from every authored bounded road patch**; a road-normal impulse would therefore be physically fabricated. The *original* 0.7 rad bound remains unchanged.

The new `wp3_source_certified_tilt_envelope_terminal_v1_dev` is a distinct **noncollision valid terminal failure** in TRAIN-development, never `physical_crash`:

- Source-bisect full RK4 steps to the *last actually accepted* pose before the existing tilt boundary, with a first rejected crossing bracket.
- Verify accepted RK4 linear and angular momentum and mechanical work/dissipation energy ledger; no artificial normal/angular crash impulse, no collision energy loss.
- Geometric SAT test that source, accepted and rejected chassis do not touch a bounded road patch.
- Independent runtime transition verifier rechecks source, prior accepted leaves, event audit, chronological prefix and absorbed tick remainder.
- Route and evaluator use a separate versioned `roll_envelope_terminal` contract, valid ONLY at certified time with reward/gate prefix, never forging a finish or a crash. Anti-forgery tests reject altered times, angle, impulse, collision kind, version or source descriptor.
- Engineering validity **is not** independent physics reviewer approval or release as registered scientific Evaluation.

## Signed numerical event certificate (original paired seed)

| Quantity | Observed |
|---|---:|
| Tick-local last accepted time | 0.02910301173105836 s |
| First outside time | 0.029103011824190617 s |
| Time bracket | 9.31322553798797e-11 s |
| Fixed roll boundary | 0.7 rad |
| Last accepted roll | 0.6999999999220863 rad |
| First outside roll | 0.700000000111329 rad |
| Minimum bounded road/body separation | 0.1413601754730962 m |
| Linear impulse ledger residual | 2.03354152622082e-13 N*s |
| Angular impulse ledger residual | 6.03311189385692e-13 N*m*s |
| Energy ledger residual | 3.30050917152214e-12 J |
| Allowed energy tolerance | 0.0002327565097517405 J |
| Artificial collision impulse | **(0, 0, 0) N*s** |
| Artificial collision energy loss | **0 J** |
| Episode termination | **True, truncated False** |

`ROLL-CERTIFICATE.jsonl` preserves unrounded source values and the evaluator's registered terminal info.

## Replays and regression

- Original R4 seed baseline before correction: **0/1 valid, 1 SuspensionResolutionError** (`R4-ROLL-BASELINE-RESULTS.jsonl`).
- On clean locked commit `fc18787`: **1/1 valid `tilt_envelope_exit`, 0 invalid**, original seed and PPO checkpoint unchanged.
- Historical R3 7 hard-stop seeds, replayed on that same clean commit: **7/7 valid `physical_crash`, 0 invalid**. No classification of those genuine collisions changed.
- **180/180 focused regression PASS on clean source** (receipt and stdout included). Additional earlier suites: 125 PASS and 61 PASS, overlapping the 180-test set.
- **Whole-repository regression PASS: 2,943 passed, 3 skipped, 0 failed**, in 1,332.56 seconds; full exit receipt = 0, source clean. Completed 26 October 2026 at 16:54:32 ICT. The exact full stdout and verified exit receipt are included below.
- Engineering Git tag **`wp3-physics-v4-roll-domain-fc18787-20261026`** resolves to the same locked source commit. Independent scientific Physics v4 signoff is still PENDING.

**Do NOT conflate engineering `tilt_envelope_exit` with a road contact.** The goal here was to stop at the actual numerical physics domain boundary while preserving honest momentum/energy and evaluator semantics, not to extend the small-angle model or to infer a collision. Scientific signoff stays PENDING.

## Exclusions

- FULL-05 **not created**.
- No 5,760-episode fresh Evaluation and no new ranking or scientific statistical claim.
- Existing R3/R4 evaluation data, original TRAIN sources and HELD-OUT untouched.
