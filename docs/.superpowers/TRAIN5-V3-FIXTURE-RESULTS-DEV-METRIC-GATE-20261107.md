# TRAIN-5 v3 — executed fixture and development metric gate

## Actual Physics v4 numerical fixture evidence

One-shot source commit:
`9122c3dde74e7466ea0f7cc937486ac87789c078`.

The previously frozen TRAIN-only fixture was **executed once** and
independently verified (no replay):

- 5/5 TRACK identities; 3 fixture seeds × 2 reset indices = **30/30** valid
  reset identities, each draw distinct
- **360** numerical simulator steps; **5,760** contact substeps
- Each step had finite observations, action, and reward with valid
  `research_outcome`, Physics v4 SHA, numerical speed bound and
  contact dynamics
- Maximum `abs(energy_residual_j)/energy_tolerance_j` across all leaves:
  **0.0674260378669114**, below the preregistered limit 1.0
- Linear/angular impulse residuals and dissipation remain within
  preregistered numerical bounds, otherwise the one-shot script would
  have failed closed
- No PPO training, heldout evaluation, frozen PPO smoke replay or use of
  any of the 18 reserved training seeds

External locked evidence receipt:

`D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\HANDOFF-9122C3D-TRAIN5-V3-NUMERIC-FIXTURE-CI-SCIENTIFIC-GATE-20261107.json`

SHA-256:
`cdc0ce69da1a0c3b2af37af5f7da99317b74237862d2598d2bcabb3b8cd661d5`

Sealed ZIP:
`HANDOFF-9122C3D-TRAIN5-V3-NUMERIC-FIXTURE-CI-SCIENTIFIC-GATE-20261107.zip`

ZIP SHA-256:
`9453383f0ea7ff9246564096d14269108a3c41e1da57a2325d340ae3e03a778d`

Exact-source regression: **199/199 PASS**. GitHub CI:
**90/90 PASS** (run 42363293276). Neither is a physical lab
calibration certificate or universal scientific training admission.

## Functional development evaluator

`f1rl/research/wp7_train5_v3_dev_metrics.py` implements a
*strict, deterministic, pure* analysis of **future** TRAIN-only
development results. It refuses partial or incorrectly grouped input
and requires:

- Six unique final-checkpoint hashes corresponding to
  `scratch/frozen_initialization × replicates 0, 1, 2`;
- All **60** named cells:
  2 training cohorts × 3 seeds × 5 TRAIN tracks × 2 conditions;
- Exactly 60 unique episode indices (0–59) per cell, 3,600 total;
- Final 200,000-step checkpoint only, no checkpoint switching;
- No HELD-OUT or other track ID, no unknown outcomes or non-finite
  return values;
- Frozen numerical thresholds as written **before** development.

The evaluator returns metrics and named failure conditions. Even if all
3,600 synthetic or real future development episodes numerically PASS,
its `scientific_training_permission` and `heldout_permission`
remain **false**.

Tests generate complete **synthetic** 3,600-record fixtures and
exercise error paths: malformed cells, duplicate indices, HELD-OUT,
mismatched checkpoint, altered threshold, invalid Physics and crash
rates. Synthetic test data are **never** mislabeled as actual
development evaluation episodes or scientific acceptance evidence.

## Scientific / authorization boundary

The source-backed physical calibration measurement set has not been
provided and has not been inferred from simulator energy checks.
The development threshold numbers remain proposals; owner-specific
signature approving their exact values is not present. Existing
`TrainingRunner`, PPO backend and TRAIN5 source-specific admission
continue to deny training.

`NUMERICAL_PHYSICS_QA = PASS`

`RESET_SOURCE = PASS`

`DEVELOPMENT_METRIC_CODE = TESTED`

`EMPIRICAL_PHYSICAL_CALIBRATION = MISSING`

`OWNER_SIGNED_THRESHOLD_PROTOCOL = MISSING`

`TRAINING_ADMISSION = BLOCKED`

**No training, fine-tuning, 3,600 development evaluation episodes,
HELD-OUT or scientific release have been executed or admitted.**
