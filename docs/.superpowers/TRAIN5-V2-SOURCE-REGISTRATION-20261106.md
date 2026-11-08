# TRAIN-5 paired v2 — source-bound registration and admission gate

**State:** Source integrity/engineering preview **PASS**. Scientific training admission **CLOSED**. PPO smoke **CLOSED**. HELD-OUT **CLOSED**.

This is a new, **non-authorizing draft** distinct from the historical four-track `wp7_protocol.build_protocol()`. The old `TRAIN_IDS`, `assets/tracks/manifest.json`, `configs/experiments/ppo-generalization-v1.yml`, historical FULL-06 (4 TRAIN/96 cells/5,760 episodes) and 12 frozen PPO checkpoint archives must remain unchanged.

## Contract files (new only)

- `configs/research/wp7-train5-paired-v2-registration.json` — canonical source-bound draft. It pins **44 source/asset/config/model files** by SHA-256, plus **4 external Step 3 engineering-witness files**.
- `f1rl/research/wp7_train5_v2_protocol.py` — fail-closed protocol verifier and preview factory, without an executable training or eval admission path.
- `tests/research/test_wp7_train5_v2_protocol.py` — tests of source hashes, exact roster, foreign/missing evidence, forged approvals, seed identity and HELD-OUT isolation.
- Previously versioned TRAIN-only assets and paired candidate:
  `assets/tracks/manifest-five-track-paired-v2.json`,
  `configs/experiments/ppo-generalization-five-track-paired-v2.yml`,
  `configs/recovery/wp7-c2-five-track-paired-v2.yml`.

## Exact scope

TRAIN-5: `straight-sprint-v1`, `slalom-v1`, `right-turn-train-v2`, `ramp-v1`, `left-turn-train-v2` in that order. Original HELD-OUT metadata: `left-turn-v1`, `offset-slalom-v1` unchanged.

- Original Right Turn v1 is excluded from this proposed *new* TRAIN-5 and stays preserved in historical TRAIN-4. Its only substitute in the new proposal is the separately versioned Right Turn TRAIN v2; Left Turn TRAIN v2 is separately authored and fingerprint-unique against original manifest metadata.
- All five candidate TRAIN factory objects can be prepared with the actual v4 Physics and race contracts, with **no HELD-OUT geometry reads and no training reset use**.
- The source-bound preview uses the existing owner-v3 physics/action/suspension parameters with a production v4 overlay. It does not silently claim that the draft C2 reconstruction is a registered scientific learning protocol; the v4 overlay and source provenance must be reviewed as a new scientific training contract.
- All actual source hashes must match the draft before touching TRAIN geometry; all four Step 3 external evidence hashes must match the original sealed receipts. The Step 3 witness additionally confirms ordered directed crossings, two successful terminal finishes and deterministic replay.
- No source path outside the known checkout may be substituted; HELD-OUT geometry is not used for the identity audit.

## Seed and evaluation proposal (NO allocation or usage)

Two future PPO choices: `train5_from_scratch` and `train5_from_frozen_checkpoint`, each proposed with **three independent seed bundles** for PPO initialization, sampling, and reset perturbation. The exact 18 proposed seed integers are in the registration JSON.

The proposal compares all six resulting policies on five development TRAIN tracks in two strata: nominal and randomized TRAIN. If later approved, the proposed grid is **60 development cells x 60 episodes = 3,600 development episodes**. These are *planning counts only*, **not experiments that have been run**. Final-only checkpoint primary scoring is proposed. HELD-OUT is explicitly excluded and remains closed; this is not a rerun or extension of historical FULL-06.

Historical owner-v3 seeds `ppo/sampling=0,1,2` and `perturbation=1000,1001,1002` are not reused for the proposed new training. The new 18 seed integers were scanned against the currently bound checked-out sources and are distinct from the old source registration, **but there is no complete cross-machine, backup, collaborator, historic-run or global seed-registry audit**. Their status is `PROPOSED_UNVERIFIED_GLOBAL`, not `RESERVED_GLOBALLY` or `UNUSED_CONFIRMED`.

## Conditions for future scientific training admission

All of the following are still unfulfilled and require a **new separately approved phase**:

1. Complete Global Seed Registry verification across the project, devices, backup archives, collaborators and training logs, with owner-signed append-only receipt. If any proposed seed is already used, revise the protocol version and lock fresh identities before running.
2. Source/CI freeze: commit reviewed branch source as a clean tree, validate fresh commit tree and hashes; resolve or explicitly disposition the two old missing-file WP7 compatibility failures; rerun required tests under the exact candidate and capture a complete source receipt.
3. Approve training procedure, initial checkpoint selection for fine-tune cohort, 200k-style compute budget/hyperparameter contract, action observation and reward equivalence under v4, seed independence and paired evaluation criteria. **No checkpoint from the original 12 may be overwritten.**
4. Validate both direction real Physics v4 witnesses and source-bound replay against the release candidate (the current Step 3 engineering proof is valid for the presently locked source, not universal future updates).
5. Separate approval of the six-case frozen-PPO smoke (if used) and train/dev scientific evaluation. Define abort-on-invalid-transition, no opportunistic retuning or repeated seeds, no held-out training/model selection, exact checkpoint/lifecycle provenance.
6. Require explicit owner sign-off and a training-specific admission function **in a new approved protocol version**. The current draft's `require_training_admission`, `require_smoke_admission`, and `require_heldout_admission` unconditionally raise `PermissionError`; changing Boolean status fields does not grant access.

Do not present the draft as a scientific approval, real training execution, new model checkpoint, new FULL-06 result or successful generalization. The 2 preexisting optional WP7 compatibility failures caused by missing files in the source baseline remain separately documented; they were not silently marked PASS.

## Technical validation

- Original step 3 real independent controller/replay: both directions passed at tick 497 with checkpoint crossings at ticks 113 and 311, v4 runtime source bound.
- Registration preflight verifies 44 source hashes and 4 external immutable receipt hashes before constructing the five TRAIN preview factories. The initial default four-track caller remains unchanged.
- 23 new focused tests and previous 169 scoped tests: **192/192 PASS**, 0 FAIL/ERROR/SKIP. This is not a full project regression and is not a global seed audit.
- All source and reports remain in the dedicated new handoff checkout; engineering receipts and JUnit stay outside the checkout under `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`.
- As required by the handoff draft, no commit/push/merge is performed at this stage.

## Decision

**Step 4 engineering registration/preflight: PASS. Scientific TRAIN-5 authorization: BLOCKED, pending the owner/global seed/CI/protocol approvals above.** The next admissible action is evidence/seed/CI audit and separately approved frozen-model paired smoke; **not retraining**.
