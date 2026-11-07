# Stage 2 — critical scientific desk review (non-independent, non-authorizing)

**Review date:** 2026-11-07
**Reviewed source:** `64b22bcf90d684836da30d2237ad1cefc94150ce`; reviewed proposal is `configs/research/wp7-train5-stage2-science-proposal-v1.json`.
**Reviewer status:** Assistant technical review; not an independent reviewer signature, not an owner-signed protocol.

## Verdict

**PASS DOCUMENT COMPLETENESS; NOT ACCEPTED FOR TRAINING.** Stage 1's cross-module and seed-source findings justify preparation, but real training admission requires scientific choices and additional engineering. This review explicitly avoids reclassifying an unmeasured quantity as a verified PASS.

## Major findings (must resolve before training)

1. **M1 — Timesteps/checkpoint contradiction:** configured `n_envs=8`, `n_steps=256` imply 2,048-step full-rollout granularity. Targeting 200,000 implies 200,704 full-rollout transitions. Existing development metrics hard reject a final checkpoint other than 200,000. The 25,000 checkpoint interval is not rollout-aligned. A new versioned, test-first budget/checkpoint/evaluation contract and explicit owner selection are mandatory.
2. **M2 — Evidence domain:** 30/30 reset/physics fixtures PASS numerically but are simulator-only (360 steps; 5,760 substeps). They do not calibrate suspension, tire forces, sensor noise, contact/energy measurement in the physical world. Restrict scientific claims to simulation-only *if* owner approves.
3. **M3 — Thresholds/statistical power:** 90% nominal, 80% per-track, 2% hard-route, 1% certified crash and zero invalid are existing proposals. Clustered policy episodes and rare failures mean aggregate point estimates alone do not demonstrate generality; require preregistered uncertainty, sensitivity and failure taxonomy.
4. **M4 — Missing actual admission:** source v2 draft and both TrainingRunner/PPO backend intentionally fail closed; no approved or authenticated training-specific scientific permission exists. Do not toggle permission flags or bypass tests.
5. **M5 — Development confounding:** 3 distinct frozen initialization candidates differ from 3 scratch replicas. Difference in outcomes cannot be interpreted as the pure effect of fine-tuning without a justified design or narrower descriptive claim.
6. **M6 — Global seed limitation:** 18 reserved seeds and signed owner facts exist, but 15 old CI job logs were lost. Owner-accepted exception does not certify universal non-use; carry risk into every admission receipt.
7. **M7 — Compatibility scope:** 24/27 WP7 capabilities are recorded implemented, three scientific capabilities blocked, 9/9 compatibility gates still in_progress; a current 226-test CI PASS is not proof those formal gates are closed.

## Decision on scientific route

Recommend a **bounded simulation-only protocol** with exact finite sampling and stop-on-failure, preserving original TRAIN/HELD-OUT isolation. Empirical calibration becomes an explicit excluded claim rather than a silent PASS; owner must approve this narrower claim and limitations. This does not waive numerical consistency, source/replay contracts, registered statistics or authentic training admission.

Two options remain open: (A) preserve PPO rollout settings and register 200,704 as the actual final checkpoint in a *new* protocol/evaluator version, or (B) redesign rollout/batch schedule to achieve exact 200,000 and test changes independently. No option is accepted in this review; the previous evaluator and source remain unchanged.

## Scientific owner-decision receipt needed

The future decision must individually approve: simulation-only domain; timestep/checkpoint solution; thresholds and analysis plan; reset perturbation assumptions; six model initializations and method comparison; seed registry exception in this protocol; no re-run/retune rules; and implementation/testing of an authenticated training admission mechanism. Only a newly locked, signed protocol after those choices can be proposed for Stage 3 admission verification. HELD-OUT and Scientific Release are *not* a consequence of this desk review.

**Final review status: `REVIEW_COMPLETE_DECISION_PENDING`; no PPO training, fine-tuning, smoke replay, development rollout or HELD-OUT executed.**
