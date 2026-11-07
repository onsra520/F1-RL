# WP7 — Stage 1 source, seed, C2 and nine-subsystem compatibility audit

**Date:** 2026-11-07
**Parent audited commit:** `f0d4f8c7b1c406d827792b3e306e86b938825194`
**Scope:** read-only/engineering evidence audit and correction of a missing authentic historical C2 draft fixture. No training, HELD-OUT, PPO smoke replay, FULL-06 or scientific release.

## Stage 1 decision

**PASS — BOUNDED ENGINEERING AUDIT, ADMISSIBLE TO STAGE 2 PROTOCOL DRAFT/SCIENTIFIC REVIEW ONLY.**

This decision is **NOT** a declaration that the nine WP7 compatibility gates, the three blocked scientific capabilities, or scientific training admission have passed. It only establishes that the documented evidence and engineering cross-module contracts have been checked with known exceptions. Stage 2 must resolve/approve the remaining scientific prerequisites separately.

## Reproduced failure and exact remediation

The initial 19-module cross-module/C2 regression on the parent source returned **175 PASS / 26 FAIL**. All 26 failures came from the missing `configs/experiments/ppo-generalization-c2-v2.yml`. The file was recovered **byte-for-byte from Git** in the original runtime project at commit `03824d52d5a26da4c4a27d7a07ec38d89ff6d0e7`, not generated from a test expectation.

- Canonical LF SHA-256: `a2cfa9bf4391069bc41ef391198da6a22dddece8bdd80b50a22a72c480ee4917`.
- Source Git blob ID: `e54fd714410734ffe6f59551d99071b1bc74fcc0`.
- Original content remains `experiment.status = draft`, `training.protocol_status = pilot_owned`; `seeds.ppo = []` and `seeds.sampling = []`. HELD-OUT tracks are names in frozen metadata only.
- After restoration: **28/28 C2 tests PASS**, including malformed configurations, status promotion, seed/version drift, and source candidate alignment.
- Complete affected cross-module group: **205/205 PASS** (20 test modules, including four fail-closed Stage 1 audit tests).

Source: `WP7-STEP1-NINE-COMPATIBILITY-C2-AUDIT-RECHECK-20261107.xml`, stored in the external, append-only evidence root. An earlier failed JUnit remains preserved; never erase the 26 FAIL history.

## Nine subsystem inspection

| WP7 subsystem | Matrix content audited | Current compatibility gate |
|---|---|---|
| simulation_runtime | fixed step, transition audit, legacy contract | IN_PROGRESS |
| vehicle_dynamics | bounded lifecycle, ramp domain, mixed-normal recovery | IN_PROGRESS |
| track_feasibility | body-aware audit, TRAIN geometry, speed feasibility | IN_PROGRESS |
| race_episode | directed gates, airborne gating, hard-route failure | IN_PROGRESS |
| gym_reset | seeded reset, Physics-to-Gym, lifecycle/reset | IN_PROGRESS |
| perception_control | observation, steering/throttle, turn control | IN_PROGRESS |
| train_ppo | vector interface, seed binding, registered training guard | IN_PROGRESS |
| evaluation | legacy metrics, seeded test-cell contract, stochastic admission | IN_PROGRESS |
| provenance_admission | manifests, closed guard, independent scientific approval | IN_PROGRESS |

The matrix's source and test paths exist for **27/27** entries; **24** capabilities are recorded as `implemented`, **3** remain `blocked`: `registered_wp7_training`, `admitted_stochastic_ppo`, and `independent_approval_restore`. Nine `compatibility.status` values remain `in_progress` by design. Each lists required cross-OS same-head CI, historical C2 replay, and cross-module seed/JSON-schema evidence. Software green tests do not establish scientific readiness.

The latest Linux source-only CI before the change was **194/194 PASS** on the parent commit, and the new C2 and Stage 1 regression was run on Windows. The closing CI must be verified **again on the exact new commit** after adding the restored artifact and tests. Historical C2 draft contract tests were completed; this is **not** a new scientific historical trajectory replay.

## Global Seed Registry and historical evidence

The 18 future TRAIN-5 seed values are distinct, and the registered proposal and owner exception manifest agree exactly. Two owner factual statements had verified Ed25519 signatures; the repository's append-only Global Seed Registry has status **`CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION`**.

**Unresolved fact:** 15 old GitHub Actions job logs are irretrievable. The owner accepted this disclosed evidence limitation; no missing job is claimed PASS and universal global seed non-use is **not** independently established. The separate TRAIN-5 proposal remains `draft_not_authorized` with `owner_approval = null`.

This audit does not change the 18 reserved seed values or consume any of them; it does not alter the original historical TRAIN-4 source, model ZIPs or Physics v4 thresholds. Source files and archived evidence stay source-hash bound.

## Stage 1 / Stage 2 boundary

**PASS for proceeding to Stage 2 drafting and scientific review**, only because the source/registry audit now has a bounded documented answer, the recovered C2 contract matches original Git provenance, the cross-module engineering tests PASS and the accepted seed exception is explicit.

**NOT PASS for training admission, confirmation, HELD-OUT or scientific release.** Before any PPO training, Stage 2 must decide simulation-only vs independently measured calibration, justify/freeze development metrics and seed protocol, review exact checkpoint-vs-scratch and budget, resolve training-specific authenticated admission at all guards, and obtain explicit owner authorization for that separate protocol. A future stage's failure must stop without opportunistic reseeding or retuning.

**No permission flag was modified and no approval was forged.** A distinct closing receipt should record the final commit SHA, source-file hash, Windows JUnit and exact-commit Linux CI after commit.
