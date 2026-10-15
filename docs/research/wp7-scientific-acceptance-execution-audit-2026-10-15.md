# WP7 — scientific acceptance execution audit (2026-10-15)

**Decision:** user authorized the full unblock *workflow*, but the existing
preregistered scientific admission controls remain binding. This document is
the evidence trail for the first read-only execution on the Windows research
worktree; it does **not** authorize TRAIN PPO, HELD-OUT, WP8, or freeze.

## Research boundary and exact repository source

- Base: Draft PR #17, commit `dd0f4e97d4ba9710c30a54b4d4f50652dcccb89b`.
- New isolated branch: `research/wp7-scientific-acceptance-evidence-v1`.
- Separate Python 3.12.15 environment; no changes to historical model
  checkpoints, exploration archives, or sealed evidence.
- Owner preregistration: decisions **1B/2B/3B/4B/5A/6B/7B/8B** recognized.
  Its reported status is `design_locked_implementation_pending`; current
  `training_permitted=false`, `heldout_permitted=false`,
  `scientific_freeze_permitted=false`, `wp8_permitted=false`.
- Original pilot preflight exits **2 / review_required** with specific blockers:
  slalom 1.94 m point-to-obstacle clearance needs body-swept feasibility;
  right-turn authored 90 degree bend vs ~5.49 m minimum bicycle radius;
  runner/evaluator end-to-end provenance not yet reviewed.
- Local Python verification prior to this increment:
  **74 focused PASS** and **1748 full regression PASS / 2 SKIPPED** (Windows).
  Platform-specific skips are **not** failed tests.

## Historical PPO evidence — what exists and what it means

Two previous real exploratory PPO runs are present in a local separate archive,
dated 2026-10-11 and 2026-10-12, each with independently checksum-verified
**8 milestones**, `heldout_rollouts=0` and
`scientific_evidence=false`. The existing read-only evidence checker labels
each `verified_exploratory_evidence_not_scientific`. The previous historical
freeze-readiness reports also list unregistered physical body geometry, point-
only collision scoring, right-turn corridor, missing training-time runtime
environment provenance, one deterministic result per track and lack of
independent off-device backup as blockers.

The older `wp7_freeze_readiness.py` on the **current** PR #17 checkout
rejected the old evidence as `archived source fingerprint declarations
drifted`. This was NOT evidence of trained-model corruption. Two config files
were updated after training. Separately, six other archived Windows
source-fingerprint SHA-256 values correspond to **CRLF checkout bytes** of
the original commit's LF Git blobs. The new
`wp7_historical_source_audit.py` verifies the exact declared source
file hashes against commit `1478cf6b9f671a08c5ace3878c8b19f34da951cc`,
allowing **only** byte-exact Git blob or explicitly logged LF→CRLF
transformation. It reports **8/8 historical source paths verified** for
both runs, while flagging the **two current-worktree config differences**.

The audit *never* asserts that the Git commit certifies the actual training
runtime/dependencies, original seed replay, episode robustness, calibration,
or scientific acceptance. `training_runtime_provenance_verified=false`
and `scientific_acceptance_permitted=false` remain fixed in the report.
No source fingerprint or historical model ZIP is rewritten.

## Read-only execution

```powershell
python scripts/wp7_review_owner_acceptance.py
python scripts/wp7_pilot_preflight.py
python scripts/wp7_verify_pilot_evidence.py "PATH_TO_RUN_0"
python scripts/wp7_verify_pilot_evidence.py "PATH_TO_RUN_1"

python scripts/wp7_historical_source_audit.py `
  --archive-run "PATH_TO_RUN_0" `
  --archive-run "PATH_TO_RUN_1" `
  --output "OUTSIDE_GIT_WORKTREE\fresh-audit.json"
```

The audit output is **create-only**, and it must reside outside the Git
worktree. Store/report its SHA-256 in evidence records. The output is **not**
itself proof of a private Google Drive restore. Run tests:

```powershell
python -m pytest tests/research/test_wp7_historical_provenance.py -q
python -m pytest -q
```

## Precise remaining unblock dependencies

1. **G1/G2:** certified 6-DOF contact time and impulse/grounded recovery for
   registered applicable road/tire regimes; preregistered physical reference
   datasets, calibration measurements, traceable uncertainty, independent
   comparison and signoff. Software/analytic baselines alone do not suffice.
2. **G3:** after admission is granted, generate genuine evaluated TRAIN
   samples under preregistered conditions: >=60 episodes per policy x track x
   stratum, >=3 paired seeds, and one-sided 95% exact Clopper-Pearson lower
   bound >=0.90 in **each** required cell. Old one-episode-per-track
   exploratory pilots cannot satisfy this statistical requirement.
3. **G4:** repeat from independently reconstructible exact training source
   and runtime dependencies; historical Git bytes alone are insufficient.
4. **G5:** authorized private Google Drive archival and independently
   reproduced off-device restore with bytewise/hash attestation. A local
   archive is not equivalent to remote recovery evidence.
5. **G6:** independent scientific review and explicit owner acceptance
   decision. A green regression pipeline does not sign the protocol.

### Hard exclusions

No new PPO training while the preregistered owner-admission gate is false;
no HELD-OUT access, WP8, legacy archive rewrite, historical baseline
relaxation or main/recovery merge. Keep this branch **Draft**.
