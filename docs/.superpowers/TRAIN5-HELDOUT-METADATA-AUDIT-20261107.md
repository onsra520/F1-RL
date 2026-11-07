# F1-RL TRAIN-5 v2 — HELD-OUT failure audit and TRAIN-only metadata correction

**Date:** 2026-11-07. **Scope:** Steps 1–2 only; not Full Regression release, not official HELD-OUT testing.

## 1. Audit of 30 failures on committed e3ed864

Source: `TRAIN5-FINAL-GROUPED-FULL-REGRESSION-E3ED864.xml` from source commit
`e3ed864e67f4f380375a8c91673d6e58e3725258` (3,150 PASS, 30 FAIL,
3 SKIP). Each of the 30 failed tests was classified by exact JUnit node ID;
the classification receipt is stored outside the source checkout in
`D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\TRAIN5-30-HELDOUT-FAILURES-CALLSITE-CLASSIFICATION-20261107.json`.

| Group | Count | Decision |
|---|---:|---|
| TRAIN-only pilot preflight and MOCK milestone/checkpoint tests | 15 | Validate existing four TRAIN asset contents only |
| Provenance for explicitly selected TRAIN assets | 6 | Validate selected assets, preserve full manifest's metadata and exact missing-ID rejection message |
| TRAIN-only Slalom geometry regression | 1 | Verify only Slalom's original track geometry and SHA, not unrelated tracks |
| Tests that explicitly read/compare actual HELD-OUT world geometry | 8 | **Leave unchanged. Separate scientific authority required.** |

Total 30, of which **22 can be repaired without access to HELD-OUT geometry**
and **8 require authorized access** for their original assertions.

The 8 retained geometry tests are: three in
`test_experiment_track_split_integration.py`, four in
`test_track_asset_corpus.py` and one in
`test_navigation_v2_reconstruction.py`. They check real geometry, spawn,
road physics, obstacle volumes, finish placement and/or authored routes.
It is not scientifically valid to substitute manifest hashes for these tests.

## 2. Minimal source changes

- `f1rl/learning/pilot_preflight.py`: pass `validate_asset_ids=TRAIN_IDS`
  only at the pilot's original callsite. All 4 TRAIN assets are still loaded,
  schema-validated and checked against world/content SHA-256.
- `f1rl/learning/provenance.py`: use
  `validate_asset_ids=selected_track_ids`, and retain the exact legacy
  error including the missing selected ID if the manifest rejects an
  undeclared track. No new evaluation or training capability.
- The TRAIN-only Slalom test, and the two provenance expectation fixtures,
  now pass their explicit selected TRAIN IDs. Assertions and expected
  physical measurements/hashes were not removed or loosened.
- `tests/research/test_train_only_manifest_callsites.py` is new test-first
  evidence: both old callsites failed RED, then passed GREEN. It also verifies
  that tampering with a TRAIN world SHA is rejected, and that the
  **default** manifest loader still rejects missing assets when called
  without `validate_asset_ids`.

The strict default behavior of `f1rl/track_assets/manifest.py` is **unchanged**.
No HELD-OUT file was opened, copied, loaded or used for training or model
selection. The original 12 frozen PPO ZIPs were not modified.

## 3. Source and scientific gates

Review branch:
`fix/train5-heldout-metadata-only-20261107`, created in a separate sparse
worktree from `e3ed864e67f4f380375a8c91673d6e58e3725258`:
`D:\F1-RL-PHYSICS-V4-TRAIN5-METADATA-REVIEW-20261107`.

Source registration is versioned in place **as an unapproved draft** with
**61 exact source SHA-256 bindings**. The previously omitted provenance
implementation is now explicitly bound, and the existing pilot and
registration-module digests were refreshed. All four original external Step3
evidence SHA-256 receipts are unchanged and valid. Original WP7 source
line endings were pinned to reproduce the exact locked bytes.

All scientific admission remains fail-closed:
`training_permission=false`, `smoke_permission=false`,
`heldout_permission=false`, `owner_approval=null`,
`global_seed_registry_receipt=null`.

## 4. Verification / what this does NOT prove

- Exact **22 originally failed metadata-only node IDs** re-run; result is
  recorded in `TRAIN5-METADATA-ORIGINAL-22-FINAL-20261107.xml`.
- Expanded focused unit/integration regression and Physics-v4 scoped
  regression have separate exact JUnit receipts in the external evidence
  directory.
- **No claim of 3,183/3,183 Full Regression PASS**: the 8 original
  HELD-OUT geometry tests are still legitimately unavailable.
- No official held-out test, model evaluation, PPO smoke, retraining,
  full scientific acceptance, GitHub CI, commit/push or owner-signed Global
  Seed Registry was performed under this review task.

**Verdict:** The metadata-only part of the original 30 failures is addressed;
8 true HELD-OUT geometry checks stay OPEN and require a separate authority
and read-only evaluation protocol.
