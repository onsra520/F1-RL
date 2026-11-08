# TRAIN-5 Global Seed Registry — owner-approved exception acceptance

**Acceptance date:** 2026-11-07

**Branch:** `handoff/physics-v4-khanh-bang-20261106`

**Locked source anchor:** `e8c15965431465e8c97b73bfe2be30fd261875ff`

**Registry disposition:** `CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION`.

## Owner decision and scope

The project owner explicitly approved in the current conversation the
documented evidence exception for **15 historical GitHub Actions job logs**
which cannot be downloaded, having reviewed the bounded audit, the
cryptographically signed 18-seed nonuse facts and the archived GitHub
workflow source. The user stated that these 15 inaccessible job logs must
**NOT** be represented as PASS, and that separate approvals are required
for PPO smoke, training, HELD-OUT and scientific release.

The approval is preserved verbatim and SHA-256 bound in
`configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json`.
It was a *direct chat authorization*, not an independently verified SSH
signature over the exception decision itself. This distinction is
intentional.

## Evidence accepted with disclosed uncertainty

- The 18 proposed seed integers match the frozen source-bound TRAIN-5 v2
  registration; no collision or prior consumption was found in reachable,
  readable project histories within the pre-declared scan boundaries.
- The owner signed a factual statement that these 18 seed values had not
  been consumed on the current machine for PPO training, smoke or
  evaluation. That statement's exact byte payload and ED25519 SSHSIG were
  cryptographically verified using the owner public key.
- A separate signed owner statement confirms no prior F1-RL execution on
  Google Colab, other personal computers or removable drives.
- Google Drive's 297 Colab notebook metadata entries and 18 indexed seed
  queries were inspected; no F1-RL-related title or indexed seed hit was
  found. Search index results are not exhaustive session logs.
- Of 22 selected old GitHub job-log requests, seven were accessible with
  zero proposed seed mentions. Fifteen were unavailable. Archived Git
  objects identify 8 source commits (October 18–19, 2026) and 2 distinct
  workflow definitions for those 15; those YAML definitions contain zero
  candidate-seed literals. They **do not reconstruct the missing logs**.
- The owner explicitly accepted this bounded, clearly stated residual
  uncertainty; no global mathematical or forensic proof of nonuse in every
  possible deleted execution is asserted.

**Exception record:** `F1RL-TRAIN5-GSR-EX-001`, accepted by owner.
The registry is *closed with an accepted evidence exception*, not an
unqualified proof of historical seed nonuse.

## Frozen source and permissions

The original signed source input
`configs/research/wp7-train5-paired-v2-registration.json` remains
`draft_not_authorized`. Its canonical source hashes, previous
`preflight_digest`, seed allocation, existing checkpoints and asset
manifests were **not** changed. This is required to preserve historical
source provenance and fail-closed behavior.

The registry acceptance is a separate immutable record. It is **not**
permission to run any PPO smoke, TRAIN-5 training/fine-tuning, HELD-OUT
policy evaluation, FULL-06 scientific evaluation or release.

These require explicit later protocol-specific owner approval, plus
evidence-backed execution and scientific evaluation. Gate status must not
be inferred from this acceptance record.

## Exact evidence references

Read-only evidence folder:
`D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`.

- Accepted registry receipt:
  `HANDOFF-E8C1596-GLOBAL-SEED-REGISTRY-OWNER-APPROVED-EXCEPTION-FINAL-20261107.json`
- Signed final 18-seed facts:
  `HANDOFF-E8C1596-F1RL-FINAL-18-SEED-OWNER-FACTS-20261107.json`
  and `.json.sig`
- Signed owner no-offmachine facts:
  `HANDOFF-E8C1596-F1RL-OWNER-NO-OFFMACHINE-STATEMENT-20261107.json`
  and `.json.sig`
- Historical source review:
  `HANDOFF-E8C1596-MISSING-GHA-LOG-WORKFLOW-SOURCE-REVIEW-20261107.json`
- Sealed evidence archive:
  `HANDOFF-E8C1596-GLOBAL-SEED-REGISTRY-OWNER-APPROVED-EXCEPTION-FINAL-20261107.zip`

External signed source artifacts are **not** included in the public repo.
The committed registry JSON records their SHA-256 hashes and references.
CI checks only the public audit decision structure and fail-closed gates;
the original owner SSHSIGs were verified locally against exact file bytes.

## Decision summary

`GLOBAL_SEED_REGISTRY = CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION`

`TRAINING = BLOCKED`

`PPO_SMOKE = BLOCKED`

`HELD_OUT = BLOCKED`

`FULL06 = BLOCKED`

`SCIENTIFIC_RELEASE = BLOCKED`
