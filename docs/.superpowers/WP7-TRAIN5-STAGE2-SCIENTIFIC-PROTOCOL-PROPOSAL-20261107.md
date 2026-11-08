# TRAIN-5 Stage 2 — source-bound simulation-only scientific protocol proposal

**Date:** 2026-11-07
**Source anchor:** `64b22bcf90d684836da30d2237ad1cefc94150ce`
**Decision state:** PROPOSAL and independent desk review completed; **no owner scientific approval, no training permission**.
**Target:** permit only simulation-domain research on five source-bound TRAIN tracks after a *separate* admitted protocol. This document does not approve physical real-world safety, transfer, HELD-OUT or final scientific release.

## Evidence inherited without rerunning experiments

- Stage 1: original missing C2 draft restored from authenticated historical Git and cross-module 205 PASS; exact-commit Linux CI 226 PASS. The Stage 1 receipt permits drafting this Stage 2 document only.
- Frozen TRAIN-5 paired-v2 registration includes five TRAIN tracks, HELD-OUT metadata only, 44 original Git source bindings and external witnesses.
- 18 registered potential training seeds are distinct and **unused** in completed experiments, with signed owner facts. A 15-job GitHub-log evidence exception was accepted; absolute global non-use was **not** independently proven.
- Original engineering fixture: 5 tracks × 3 independent fixture seeds × 2 reset indices = **30/30**; 360 simulator steps, 5,760 physics substeps, no invalid transitions. This is **not** external physical calibration. Fixture SHA-256 is bound in companion JSON.

## Proposed science scope

**Recommend a simulation-only study.** Scientific question: on five registered TRAIN tracks, how do final PPO policies from scratch differ descriptively from policies initialized from 3 pre-existing frozen model ZIPs under nominal and randomized TRAIN reset conditions? This choice needs owner approval. The 3 heterogeneous initialization models confound a simple causal “fine-tuning effect”; report model-by-model and within-cohort variation without overclaiming.

No empirical physics calibration is available. The reset adapter uses lateral jitter ±0.5 m, heading ±5°, up to 32 rejection attempts; these are source-bound engineering assumptions, not measured road/vehicle uncertainty. Any sim-to-real/hardware claim is explicitly out of scope.

## Candidate design and fixed counts

| Item | Proposal |
|---|---|
| Models | 2 cohorts × 3 replicates = **6** |
| PPO | CPU, 8 vector envs, n_steps 256, batch_size 64, gamma .99, learning rate 0.0003; other parameters source-bound |
| Nominal budget | 200,000 timesteps per model, 1,200,000 requested total |
| Checkpoint policy | final checkpoint only; no validation-driven checkpoint search |
| Development | 6 models × 5 TRAIN tracks × 2 conditions × 60 episodes = **3,600** |
| HELD-OUT | blocked, never used for checkpoint or threshold selection |

**Critical hard stop — timestep budget:** 8 × 256 = **2,048 env transitions per PPO rollout**. A standard full-rollout PPO run targeting 200,000 advances through 98 rollouts to **200,704** per policy (an overshoot of 704). The existing development metric code mandates `checkpoint_step == 200000` and therefore cannot honestly accept the real full-rollout final checkpoint. The proposed 25,000-step intermediate checkpoint schedule is likewise not an exact multiple of 2,048. Before admission the owner must approve and source-version *one* consistent design: either register final 200,704 plus a harmonized evaluator/checkpoint schedule, or separately test a modified rollout/batch configuration that produces exactly 200,000. No metric test, checkpoint number or protocol is modified in this Stage 2 draft.

## Development metrics — proposals, not accepted thresholds

Preexisting numeric thresholds are preserved unchanged:

| Metric | Threshold | Exact count under proposed design |
|---|---|---|
| Nominal finish | ≥90% | ≥1,620 / 1,800 nominal |
| Per-track finish | ≥80% | ≥576 / 720 per track |
| Hard-route violation | ≤2% | ≤72 / 3,600 |
| Certified physics crash | ≤1% | ≤36 / 3,600 |
| Uncertified/invalid physics | 0 events | 0 / 3,600 |

The existing metric-gate code runs against **synthetic** 3,600-row fixtures and rejects malformed episodes, unknown tracks and threshold modification; no genuine PPO development evaluation has occurred. These thresholds still lack owner signature and documented statistical power. Repeated episodes on six fixed policies are *clustered*, not 3,600 independent learned policies. Before science acceptance, pre-register per-policy, cohort, track, perturbation and per-cell reporting; uncertainty intervals respecting the dependence, plus a risk rationale for rare crashes. No after-the-fact threshold optimization.

## Seed independence and source integrity

Six training identities use exactly the 18 reserved PPO/sampling/perturbation seed values already registered (3 per run); **zero consumed** in Stage 2. Define domain-separated evaluation seed derivation keyed to the final signed protocol SHA and policy identity, with collision checks against the 18 reserved training values before execution. The current registration's evaluation seed expression is a *proposal*, not a seed-generation implementation. No HELD-OUT geometry, PPO train/smoke/finetune, or Full-06 is accessed.

## Scientific admission checklist

1. Approve simulation-only research claims or provide independent empirical calibration. **Pending.**
2. Reconcile 200k versus real rollout granularity and version all affected checkpoint/dev-metric contracts. **Pending.**
3. Review/freeze thresholds, baseline choice, uncertainty analysis, cohort confounding and stop rules. **Pending.**
4. Accept the 18-seed historical GitHub-log exception for *this specific* future training protocol; original owner fact signatures do not grant training. **Pending protocol-specific sign-off.**
5. Implement and test a new *explicitly admitted* runner + PPO backend + protocol guard under a new version. Existing unconditional fail-closed guards must not be disabled. **Pending Stage 3 review.**
6. Revalidate source/manifest hash, reset sampling, reviewer authentication, CI same-commit, separate owner authorization, and one-shot run receipt before any real training. **Pending.**
7. Resolve applicable 9 WP7 compatibility gates and 3 blocked scientific capabilities under their own evidence rules. **Pending.**

**Stage 2 disposition: protocol prepared and critically reviewed, but scientific training admission is BLOCKED.** The user authorization “lượt 2” approves preparing this proposal, *not* silently signing the listed scientific choices. See the structured companion JSON for hard-stop risk register and source hashes.
