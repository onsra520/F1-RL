# WP3 TRAIN handoff — Khánh Băng (@BANGPC123)

**Date:** 2026-10-21 (Asia/Ho_Chi_Minh)
**Repo:** private `onsra520/F1-RL`
**Scientific status:** **development-only TRAIN experiment; WP7/HELD-OUT not authorized**
**Frozen executable source:** `research/wp3-ppo-tuning-v2` at **`fe83f517479f3b990028d6409304fae83b8199a9`**
**Handoff branch:** `handoff/wp3-khanh-bang-20261021` (checkpoints + metadata + recovery helper; *not* a valid source HEAD for resuming the existing campaign)

## What's transferred

- WP3 full source branch (27 commits on top of main): `scripts/wp3_campaign.py`, `scripts/local_demo.py`, `f1rl/local_demo/*`, configs, physics/perception/environment, validation and tests.
- **48 verified PPO checkpoints:** 16 for each `C-seed0`, `A-seed0`, `B-seed0`, at every 4,096 steps from 4,096 through 65,536.
- **3 final trained models** (`model-final.zip`), plus training/run manifests and engineering validation. All three independently validated `ENGINEERING_PASS`, with all 48 checkpoint hashes matching their original manifests.
- **Full portable TRAIN evidence** as a private GitHub prerelease asset `WP3-attempt-v8-full-evidence.zip`, including the three original large `training-telemetry.jsonl` files, `training-integrity.json`, original event log and campaign state. This archive is essential to *validate and reuse* the three completed models without retraining; checkpoint ZIPs alone are insufficient.

## Current campaign state and recovery caution

The orchestrator exited unexpectedly at **17:40 local time** after finishing `B-seed0`. Both B0 final weights and `engineering-validation.json` are valid, but its PASS had not been committed to the authoritative `campaign-state.json`. A stale `campaign-state.json.tmp` remains in the evidence. The unmodified campaign displays 2/12 complete and `current_model_id=B-seed0`; the true completed TRAIN runs are **C0, A0, B0 (3/12)**.

**Do not start a new campaign, rerun C0/A0/B0, modify immutable manifests, combine runs under a new source SHA, or read HELD-OUT data.** Avoid copying only the checkpoints and claiming the original full campaign can be resumed.

## Băng's setup (Windows PowerShell, Python 3.12 + uv)

```powershell
git clone https://github.com/onsra520/F1-RL.git F1-RL
cd F1-RL
git fetch origin
git switch research/wp3-ppo-tuning-v2
# This MUST equal fe83f517479f3b990028d6409304fae83b8199a9
git rev-parse HEAD

uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e '.[dev]'

# Fetch snapshot from the private GitHub prerelease.
gh release download wp3-khanh-bang-20261021 -R onsra520/F1-RL -p WP3-attempt-v8-full-evidence.zip
# Verify against SHA listed below and in release notes before extracting:
(Get-FileHash .\WP3-attempt-v8-full-evidence.zip -Algorithm SHA256).Hash.ToLower()
Expand-Archive .\WP3-attempt-v8-full-evidence.zip -DestinationPath .\wp3-evidence
$evidence = (Resolve-Path .\wp3-evidence\attempt-v8).Path
```

**Expected full evidence archive SHA-256:** `fd39193a8689db1ba529910851435a46402981372b1dd893c7fd523b72363edf`

From a separate handoff worktree, or by extracting just this recovery helper:

```powershell
git worktree add ..\F1-RL-wp3-handoff origin/handoff/wp3-khanh-bang-20261021
$helper = '..\F1-RL-wp3-handoff\handoff\wp3-khanh-bang-20261021\recover_b0_state.py'
$source = (Get-Location).Path

# First, independent full validation, no authoritative campaign-state mutation.
.\.venv\Scripts\python.exe -B $helper --source $source --evidence $evidence
# Only after successful validation:
.\.venv\Scripts\python.exe -B $helper --source $source --evidence $evidence --apply

# Resume the SAME immutable attempt; this trains the 9 pending models.
.\.venv\Scripts\python.exe -B scripts\wp3_campaign.py resume --output $evidence
```

The recovery tool checks the frozen source, clean Git checkout, campaign identities, lock exclusivity and all three full training runs before reconciling B0. It creates a state backup and append-only recovery event. **Dry-run first; never bypass a failed verification.** Record the resulting state and hashes before proceeding.

## Frozen campaign protocol

- Sequence: `C0,A0,B0,D0,A1,B1,C1,D1,A2,B2,C2,D2`.
- 4 candidates (penalty 2/4 × training corridor 2/3m), 3 independent seeds per candidate; 12 models, 65,536 steps/model, 16 checkpoints/model.
- `PPO MlpPolicy`, 16 observations, 3 actions; **CPU** only, `n_envs=1`; changing GPU, parallelism, reward, physics or seed invalidates this existing campaign's frozen identity.
- After 12 verified models, run registered **TRAIN** evaluation (5,760 episodes), report finish rates and failure reasons, retain error cells; **never unseal HELD-OUT or claim WP7 scientific acceptance**.
- Do not claim `ENGINEERING_PASS` means the policy drives successfully. It only means integrity, coverage and learning steps passed.

## Hashes / upload design

- `HANDOFF-SHA256.json` hashes all transferred checkpoint/model and metadata files in this Git branch. No `training-telemetry.jsonl` is committed to Git because each is ~257–258 MiB.
- The **full evidence archive** (GitHub Release asset) preserves original telemetry so each completed run can be revalidated.
- Original source commit `fe83f51` remains available as a dedicated remote branch. **Do not execute resume while checked out at the handoff branch HEAD**, because `_load_campaign` correctly rejects source-identity drift.
- GitHub repository is private; @BANGPC123 has write access to it.

## Deliverables requested from Khánh Băng

1. Verify archive SHA-256, original manifests, all 48 checkpoints and 3 final models.
2. Run B0 recovery **dry-run**, reconcile only after checks pass and preserve backup.
3. Complete 9 remaining models without altering frozen config/provenance.
4. Run TRAIN evaluation and compare the 4 candidates across seeds.
5. Report evidence integrity, throughput, failures, winning candidate and scientific limitations. Keep Gate 2 and HELD-OUT blocked unless a separate approved process authorizes them.
