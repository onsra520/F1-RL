# Physics v4 / TRAIN-5 handoff — source branch integration record

> **Historical record (07/11):** For current terminal-penalty implementation, interrupted batch, evidence and working branch, see [TRAIN-5 terminal penalty handoff (11/11)](TRAIN5-TERMINAL-PENALTY-HANDOFF-20261111.md). Current working branch is `fix/physics-v4-handoff-audit`; the target below describes the earlier handoff only.

**Target branch:** `handoff/physics-v4-khanh-bang-20261106`

**Scope:** Implement source-only technical checks and GitHub CI directly on the
user-selected handoff branch. This record is not approval to run PPO training,
smoke, HELD-OUT geometry/policy evaluation or scientific FULL-06.

## Reviewed input and frozen boundaries

- The complete `docs/` corpus in the handoff checkout (40 files;
  3,298,629 bytes) was read bytewise and SHA-256 inventoried separately under
  `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\HANDOFF-7469053-DOCS-CORPUS-INVENTORY-20261107.json`.
- The plan, spec, Phase 2A/2B/2C handoff reports, Step-3 bilateral v4
  executable-feasibility report, TRAIN-5 registration and seed/source audit
  were inspected for execution boundaries and scientific scope.
- Historic Physics-v4 handoff baseline `2417570f32f6879c2651bce37d88fd6ac83a2ea4`
  is an ancestor of this branch; the draft registration source anchor is
  `1de279c37e62a27531d661480d20fa6e095c4e84`.
- The 31-file handoff engineering implementation was already committed as
  `74690534949c657bde9fa18b7cf4c895862739b7` before this integration.
- The frozen historical four-track roster, original train assets/configs,
  12 PPO checkpoints, HELD-OUT geometry and the original Step-3 evidence
  were not edited.

## New implementation on this branch

- `scripts/wp7_train5_handoff_preflight.py`: portable, read-only source
  integrity and fail-closed admission check. Validates 44 registered source
  inputs, 18 **proposed and globally unverified** seeds, frozen roster,
  exact source Git history and draft permission fields. Optionally verifies
  the four external historical Step-3 receipts when explicitly supplied.
- `configs/research/wp7-train5-paired-v2-git-blob-lock.json`: per-source
  Windows receipt SHA / canonical Git blob SHA bridge. Locally verified
  15 byte-identical and 29 CRLF/LF-only representations. Git checkout on
  Linux must verify all 44 pinned Git blob SHA identities; never pretend
  Windows raw byte hashes are directly reproduced for normalized LF files.
  The original draft registration was not modified.
- `tests/research/test_wp7_train5_handoff_preflight_portable.py`:
  negative-path tests for forged approval and missing external witnesses,
  positive cross-platform source-lock test and HELD-OUT path isolation.
- `.github/workflows/handoff-train5-engineering-preflight.yml`:
  no-training GitHub Actions workflow on **this exact handoff branch**,
  checking history, source lock, and hermetic TRAIN-only tests. It cannot
  verify external Step-3 evidence absent from the runner.

## Evidence boundaries and remaining work

- The earlier 3,176-pass scoped regression and two successful GitHub CI runs
  are on *another, non-ancestral source history*, commit `c6c2c731…`.
  **They do not automatically establish PASS for this branch.**
- Independent handoff-branch original focused tests: 60 PASS on
  `7469053` (raw JUnit retained externally).
- The new portable preflight tests: 12 PASS before adding the CI workflow
  (JUnit retained externally). This handoff commit must receive its own
  local test and remote CI outcome before a handoff-specific engineering PASS.
- GitHub Actions old run IDs and a technical owner signature for the other
  release candidate are not transferable to this branch.
- Global Seed Registry remains
  `OPEN_OWNER_OFF_MACHINE_ATTESTATION_REQUIRED`: Colab execution history,
  other machines/offline media, deleted logs and old inaccessible GitHub logs
  have unresolved provenance. Zero observed hits does **not** certify global
  seed non-use.
- PPO turn smoke, training/fine-tuning, HELD-OUT geometry/policy evaluation,
  scientific FULL-06 and scientific release remain **BLOCKED**. No approval
  flag or source lock may be edited to bypass them.
- The approved handoff plan's remaining scientific work requires a separate
  source-bound training protocol review, owner-signed global seed attestation
  and subsequent explicit authorization. The current work is engineering
  integration only.

### How to verify safely

Local, with the existing Python 3.12 environment (no PPO execution):

```powershell
python -B scripts/wp7_train5_handoff_preflight.py --root .
python -B scripts/wp7_train5_handoff_preflight.py --root . --evidence-root "D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106"
python -B -m pytest tests/research/test_wp7_train5_handoff_preflight_portable.py tests/research/test_wp7_five_track_opt_in.py tests/research/test_wp7_train_only_audit_no_heldout.py tests/research/test_wp7_train_turn_gate_audit.py tests/research/test_left_turn_train_v2_design.py tests/research/test_right_turn_train_v2_design.py -q -p no:cacheprovider
```

Final gate verdict: **ENGINEERING INTEGRATION — subject to same-commit
source test + CI; SCIENTIFIC TRAIN-5 RELEASE — OPEN/BLOCKED.**
