# WP3 issue #51 — historical evaluation evidence snapshot

**Scope:** frozen TRAIN-development evidence copied byte-for-byte from the
local `wp3-eval-issue51-v1` archive. All **202 regular source files**
(4,013,135 bytes) were SHA-256 matched against the original, with per-file hashes
in `SHA256SUMS`. The original local archive was not modified.

**Scientific status:** `PASS_WITH_INVALID_CELLS`, **31/96 complete cells**,
**65 invalid cells**. Evaluation attempted 2,153 valid episodes; not a complete
5,760-episode evaluation. This is NOT model selection, scientific acceptance,
TRAIN-5 permission or HELD-OUT generalization proof.

**Important integrity discrepancy:** `WP3-EVAL-SOURCE-PROVENANCE.json` records
`evaluation_roster_sha256 = f2f9d15954cea48345a8932773ed945f6ed250ad1435b4e675b97e4a6b94f319`
but SHA-256 of the supplied `evaluation/evaluation-roster.json` bytes is
`1530a54a636f80cacc096f2ed86d9e89fd5e7c7a7429873deb2fe366bcd859dc`. The recorded provenance and roster have
**not** been reconciled. No historical source artifact was rewritten.

**Linked artifacts:** `runs/` in the original is made of 12 junctions to
a separate tree with 300 files and approximately 3.29 GB of model/run artifacts.
Those external objects are **not** included here; see `runs/README.md`.
This snapshot is a transparent data-and-report archive, not a complete training
checkpoint archive.

**Boundaries:** original provenance reports
`original_authoritative_campaign_recovered=false`, `heldout_access=false`
and `scientific_evidence=false`. Git inclusion changes none of these facts.
