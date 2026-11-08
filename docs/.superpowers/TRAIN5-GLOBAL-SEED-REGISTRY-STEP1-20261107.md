# TRAIN-5 Global Seed Registry — Step 1 bounded scientific audit (2026-11-07)

**Exact source anchor:** `40f1d080da8ab74a4b0bbc51d4b8f6a1b6a34c64`

**Branch:** `handoff/physics-v4-khanh-bang-20261106`

**Engineering source state:** 44 pinned source hashes and 4 external Step-3
receipts verified on the local machine; this branch's GitHub CI was SUCCESS
at the source anchor, workflow run `42355849823`.

**Step 1 verdict: PASS within the explicitly inspected, readable scope.
Global Seed Registry: OPEN / NOT SCIENTIFICALLY CERTIFIED.**

## Verified scoped facts

- The TRAIN-5 v2 proposed seed roster contains exactly 18 distinct uint32
  integers; the proposed roster is identical to the seven earlier local
  history-audit receipt rosters. No seed was allocated or consumed by this
  read-only review.
- At source anchor `40f1d08`, 531 readable tracked source/config/text files
  were scanned. Exactly 36 literal seed occurrences were observed: 18 in
  `configs/research/wp7-train5-paired-v2-registration.json` and 18 in
  `f1rl/research/wp7_train5_v2_protocol.py`. Both are proposal declarations,
  not execution logs. No observed consumption witness outside these files.
- Seven independently archived C:/D:/WP3/Git/source-history audit receipts
  have matching SHA-256 and report zero observed usage in their bounded scopes.
  Original receipts and historical source material are unchanged.
- Google Drive: all 297 accessible items in the Colab Notebooks folder
  inventoried (297 notebooks). No file *name* matched F1-RL / TRAIN-5 / PPO
  / Physics v4. All 18 Drive fullText indexed queries completed with zero
  indexed results. This is not proof that notebook cells were never run.
- GitHub Actions: 100 most recent metadata items previously enumerated;
  7 selected available logs read with no exact proposed-seed literals, while
  15 selected old logs returned unavailable. Historical run coverage is
  incomplete.
- Three denied folders in the C: user directory were inspected and found
  to be hidden/system Windows reparse points (My Videos / My Pictures /
  My Music); this does not mean the full C: volume was exhaustively scanned.

## Evidence — kept outside the checkout

Base evidence directory:
`D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`.

- `HANDOFF-40F1D08-GLOBAL-SEED-BOUNDED-AUDIT-20261107.json`
  SHA-256: `dea206a83bd4d2ecf1cb229a652e323fa760ef98c16627fee0cb52e370d93dbb`
- `HANDOFF-40F1D08-GLOBAL-SEED-OWNER-PENDING-EVIDENCE-20261107.zip`
  SHA-256: `9f74fbeb613cfa1ef7b08a661101ee6bfd3454f0bae4ca803a62e640c4def425`
- `HANDOFF-40F1D08-GLOBAL-SEED-OWNER-FACT-DECLARATION-UNSIGNED-20261107.md`
  is a **blank factual declaration**, not a signature or approval.
- Raw Google Drive metadata receipts and the seven earlier local receipt
  files are bundled with the exact source CI and hash references.

## Blockers to true Global Seed Registry PASS

1. Owner must enumerate all F1-RL execution contexts historically used,
   including other PCs, USB/offline disks, private cloud accounts, Google
   Colab sessions, deleted notebooks and any collaborator-owned copies.
2. For each inaccessible or deleted source, owner must either supply
   trustworthy prior records or explicitly disclose that proof is unavailable.
   Unknown histories must not be relabeled PASS.
3. Independently determine that all 18 seeds have no prior consumption
   witness and sign the factual declaration using the owner's controlled key.
   The earlier signed *engineering technical acknowledgment* is different
   and cannot be reused as a Global Seed Registry attestation.
4. If any material history cannot be certified, leave Global Seed Registry
   OPEN; consider a separately versioned, pre-registered fresh-seed protocol
   with separate scientific approval instead of fabricating prior evidence.

## Fail-closed restrictions

The registered proposal `configs/research/wp7-train5-paired-v2-registration.json`
is intentionally still `draft_not_authorized`, with `owner_approval=null`,
`global_seed_registry_receipt=null`, `seed_audit_status=PROPOSED_UNVERIFIED_GLOBAL`,
and `training_permission=false`, `smoke_permission=false`,
`heldout_permission=false`.

**No PPO smoke, training, held-out policy evaluation, FULL-06 scientific
evaluation or scientific release has been authorized by this step.**

### Final disposition

`PASS_BOUNDED_READONLY_EVIDENCE_ONLY` — scoped audit record completed;
`OPEN_OWNER_OFF_MACHINE_ATTESTATION_REQUIRED` — global owner signoff not complete.
