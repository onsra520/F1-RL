# WP7 Scientific Acceptance — Approved Unblock Protocol v1
**Date:** 2026-10-15. **Status:** ACCEPTED ROADMAP, NOT SCIENTIFIC ACCEPTANCE.
**Branch:** research/wp7-physical-validation-readiness-v1, stacked on PR #16.
**Split:** TRAIN permitted only for preregistered evaluation when gates allow it; HELD-OUT SEALED.

## Decision recorded

Proceed in order **G1 physical reference verification → G2 experimental calibration → G3 TRAIN robustness evaluation → G4 reproducibility → G5 private Drive evidence recovery → G6 independent scientific review**. Acceptance of this *plan* does not constitute approval to change acceptance thresholds, run new PPO training, access HELD-OUT, merge draft PRs, or start WP8.

## G1 — Two separate proofs, not interchangeable

**Software analytical references:** the frozen matrix `WP7_ANALYTIC_REFERENCES` in `f1rl/research/wp7_reference_benchmarks.py` contains six channel-level mathematical baselines: ballistic vertical position/velocity, principal-axis quaternion w/y, zero hub-to-plane gap and zero yaw steering moment. Unit, case ID, traceable calculation and numerical residual are checked, all required channels mandatory, extras/duplicates blocked. These synthetic tests **must pass** but cannot certify experimentally correct vehicle physics.

**Empirical physical validation still required:** pre-register and independently review externally sourced trajectories/reference measurements; include vehicle geometry and inertia provenance; compare contact-time/normal, rebound velocity, lateral yaw, suspension force and 6-DOF angular motion. No experimental error thresholds have yet been agreed; freeze of a measurement protocol and tolerances must precede comparison. Do not assert reference benchmark PASS as physical G1 PASS.

**Unresolved time-of-impact:** the exact Euler ODE enclosure in PR #16 proves *clearance only*; near-surface contact gives an uncertainty interval, not a solved impact event. The true 6-DOF event-time impulse/recovery solver is still a separate engineering blocker.

## G2 — Calibration evidence acquisition, no fabricated constants

Request actual measurement/evidence for **all 12 named values** in `REQUIRED_WP7_MEASUREMENTS`:
`mass_kg`, `pitch_inertia_kg_m2`, `yaw_inertia_kg_m2`, `roll_inertia_kg_m2`, `wheel_radius_m`, `wheel_width_m`, `spring_stiffness_npm`, `damper_nspm`, `front_cornering_stiffness_nprad`, `rear_cornering_stiffness_nprad`, `tire_friction_mu`, `tire_restitution`.

Every measurement needs a correct SI unit, uncertainty, method, evidence ID, original bytes with matching lowercase SHA-256 and an independent reviewer evidence artifact. The evaluator in `wp7_calibration_readiness.py` checks **structural integrity only**. A value and a metadata label cannot establish experimental accuracy, independent review or a scientifically calibrated vehicle. Experimental source and method selection remains to be agreed.

## G3 — Preregistered TRAIN robustness (later; no training in this PR)

Once G1/G2 and experiment registration are approved, evaluate exactly the registered policy × TRAIN track × stratum matrix, with **>=60 stochastic episodes per required cell**, **>=3 matched paired seeds**, **one-sided 95% Clopper–Pearson lower bound >=0.90** for each cell. The existing `wp7_robustness_gate.py` can check *provided* results without evaluating agents. Missing cells and underperforming strata must fail. Neither synthetic successes nor unit-test fixtures qualify.

## G4 — Reproducibility

Pin exact Git tree SHA, build dependencies/OS, vehicle geometry, inertia and solver policies, track world hashes, TRAIN seeds/strata, evaluator version and source data SHA-256. Independent reviewer must be able to reproduce each result without modifying HELD-OUT, historical model checkpoints or accepted evidence.

## G5 — Private Google Drive recovery (later)

Manifest each artifact with SHA-256; upload approved evidence only to its authorized private Drive destination. Independently re-download/restore to an isolated location and compare bytes/digests with the original. `compare_restored_evidence` verifies bytes but **does not attest a Drive provider, backup permission or actual upload**. The final gate additionally requires a verified Drive source identifier, access controls, and independent restore receipt. No upload is claimed or performed in this PR.

## G6 — Scientific signoff (last)

Only after G1–G5 have external evidence and passed review can the owner consider scientific acceptance and issue a documented freeze decision. A green GitHub Actions run, a mathematically complete reference matrix, or a locally matching evidence archive cannot confer scientific approval.

## Current unblock status

| Gate | Software preparation | Scientific gate |
|---|---|---|
| G1 reference | Frozen synthetic analytic matrix and conservative Euler clearance | **BLOCKED:** measured reference and 6-DOF TOI/recovery |
| G2 calibration | Coverage, units, provenance, SHA-256 verifier | **BLOCKED:** authentic measurements and lab validation |
| G3 robustness | Exact CP numerical checker | **BLOCKED:** preregistered real TRAIN outcomes |
| G4 repeatability | Existing deterministic CI + evidence contracts | **BLOCKED:** independent reproduction |
| G5 Drive recovery | Offline bytewise restore validator | **BLOCKED:** real private Drive backup/restore |
| G6 signoff | Required gate sequence documented | **BLOCKED:** independent reviewer and owner signoff |

### How to run the currently authorized verification

```bash
python -m pytest tests/research/test_wp7_reference_benchmarks.py tests/research/test_wp7_calibration_readiness.py tests/research/test_wp7_robustness_gate.py -q
python scripts/wp7_review_owner_acceptance.py
python -m pytest -q
```

Both OS and the exact PR HEAD are checked by the existing WP7 GitHub Actions workflow. No automatic evaluation or PPO training is introduced.
