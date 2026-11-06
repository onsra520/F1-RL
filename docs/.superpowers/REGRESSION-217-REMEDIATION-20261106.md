# F1-RL Physics v4 / TRAIN-5 — 217-case Full Regression remediation

**Status: PARTIALLY REMEDIATED — release, GitHub CI approval, PPO smoke and training are CLOSED.**

## Scope and baseline

Original full regression on isolated handoff checkout:
**3,018 collected; 2,798 PASS; 217 FAIL; 0 ERROR; 3 SKIP**.
The three SKIP cases require Windows symlink privileges. That suite ran before the historical-source restoration and is not retroactively described as a PASS.

This remediation uses a **separate review candidate worktree**:
`D:\F1-RL-PHYSICS-V4-TRAIN5-RC-20261106`, branch
`rc/physics-v4-train5-v2-review-20261106`, based on original
`1de279c37e62a27531d661480d20fa6e095c4e84`.

No commit, push, merge, frozen checkpoint modification, PPO model rollout or HELD-OUT geometry read occurred.

## Reproducible Git restoration, not synthetic evidence

Restored **50 historical paths** in the review worktree from immutable Git commit blobs (not mutable source copies):

- **28 source/evidence dependencies**: configurations, source scripts, archived WP3 receipts and legacy README files; every restore has commit and SHA-256 provenance.
- **1 original WP3 Crash Terminal regression receipt** (old test record, not new certification).
- **18 historical `src/**` readiness-matrix source references**; all checked against Git HEAD from the same branch as the historical matrix. One path, `src/simulation/spatial_contact_dynamics.py`, has a different tracked blob in a second historical checkout; the chosen commit and both SHA-256 identities are preserved in the audit.
- **3 original WP3 diagnostic replay receipts** for Right Turn, Ramp and Slalom. Their original `full_5760_eval_admission=BLOCKED` and `scientific_evidence=false` remain unchanged; these are not newly successful episodes.

A versioned **CLI-only fix** was made to the restored `scripts/wp7_verify_run_evidence.py`: use its physical repository root to import `f1rl` when executed by absolute path in an uninstalled checkout. This verifier does not load models, execute evaluation, change physics, or authorize training.

The RC `.gitattributes` pins the exact Git-compatible line endings of the 44 originally source-locked files and 50 restored historical files. No original immutable model archive was rewritten. Everything is separately copied and SHA-verifiable. Original dirty project checkout remains untouched.

## Tests: exact comparison, no inflated full-regression claims

The **194 original FAIL cases** that could be retried without running protected HELD-OUT/model tests were replayed by their exact pytest node IDs:

| Retry group | PASS | FAIL |
|---|---:|---:|
| First selected replay of original failures | **160** | 34 |
| Additional historical WP3 blocked-release fixtures (same original failed nodes, after restoration) | **7** | 0 |
| CLI source-bound replay verifier (same original failed node, after fix) | **1** | 0 |
| Five documentation-only cases that were previously excluded | **5** | 0 |
| Two model-worker rejection tests that cannot load a real model or execute training | **2** | 0 |

Thus **175 distinct tests that originally FAIL are now confirmed PASS** in independent targeted reruns.

The other **42 original failed test IDs are not certified PASS**:
- **30** require sealed HELD-OUT geometry, directly or through original whole-manifest provenance; no access granted.
- **3** require original historical WP3 source commits to be genuine ancestors of the snapshot-style handoff branch; Git ancestry cannot be forged.
- **2** expect a clean and committed WP2 source tree; the review candidate remains deliberately uncommitted because release gates are OPEN.
- **7** are real PPO fixture / vector integration / generated policy archive tests deliberately not run without the separate model-test scope.

**This is not a new full-project 3,018-test PASS.** The original 217-failure receipt is preserved; an exact complete rerun is blocked until the missing legacy-test authority/fixtures/branch history are resolved. No tests were marked as passed through skipping, fake evidence, monkeypatching admission or modifying their assertions.

After these restorations, an additional focused Physics v4 / TRAIN-5 v2 suite runs **170/170 PASS, 0 FAIL/ERROR/SKIP**. The source-bound draft preflight still checks **44/44 source files plus 4 external Step3 engineering evidence receipts**. All TRAIN-5 v2 admission methods continue to unconditionally DENY training, six-case frozen PPO smoke and HELD-OUT. Original baseline 20 inputs and 30 archived SHA evidence records remain preserved.

## Scientific boundary

- **WP3 Physics release gate 7/7 PASS means the historic three invalid replay cases remain identified as invalid, not newly scored as certified crashes.** Full 5,760-episode evaluation is still explicitly BLOCKED.
- **Legacy `src/` matrix coverage only verifies referenced files exist**. Their presence does not prove that the live `f1rl/` implementation passes newer scientific validation.
- The current owner-v3/paired TRAIN-5 candidate is still a **draft**, with proposed seed audit `PROPOSED_UNVERIFIED_GLOBAL` and no global owner signature.
- GitHub Actions workflow is prepared in the RC but not run remotely. **No CI SUCCESS is claimed.**
- This work does not replace the held-out scientific policy, modify collision/reward thresholds or retrain any model.

## Raw audit receipts

All under `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`:

- `RELEASE-FULL-REGRESSION-20261106.xml` — original full-run 217 FAIL/3 SKIP.
- `RELEASE-MISSING-DEPS-HISTORIC-PROVENANCE.json`
- `RELEASE-RESTORED-28-HISTORIC-DEPS-20261106.json`
- `RELEASE-WP7-LEGACY-REFERENCE-PROVENANCE.json`
- `RELEASE-WP3-CRASH-RECEIPT-RESTORE.json`
- `RELEASE-WP3-THREE-OLD-INVALID-WITNESSES-RESTORE.json`
- `RELEASE-RECHECK-SAFE-ORIGINAL-FAILURES.xml`
- `RELEASE-RECHECK-34-REMAINING-FAILURES.json`
- `WP3-RELEASE-BLOCKED-GREEN-JUNIT.xml`
- `WP7-CLI-VERIFIER-SOURCE-FIX-JUNIT.xml`
- `RESTORED-LEGACY-README-DOCS.xml`
- `PPO-ADMISSION-DENIAL-ONLY-JUNIT.xml`
- `RELEASE-AFTER-RESTORE-SCOPED-JUNIT.xml`

## What still blocks full release

1. Explicit owner review for original HELD-OUT fixture handling — no automatic copying into training data.
2. Recreate a legitimate source ancestry/release branch based on actual historical WP3 lineage, with new source and training-protocol receipts, rather than faking a parent commit.
3. Obtain permission to commit and close the clean-source/CI gates **only after** the full regression scope is resolved.
4. Separately authorize any model-execution tests; do not conflate them with the six frozen PPO turn smoke cases.
5. Confirm Global Seed Registry across other machines/cloud/Colab and obtain owner signature.

**Verdict: Significant source regression repair, but scientific admission and full release remain BLOCKED.**
