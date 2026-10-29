# Research and immutable evidence

This directory contains historical experiments, scientific safety/authorization protocols, handoffs and canonical frozen results. File names containing `wp3`, `wp7`, `v1`, `v4`, track IDs and dates are **part of recorded identity**; they are not active API naming that should be mechanically made versionless.

At source baseline `b11a501`, there were 260 Git-tracked items under `docs/research/`, including 153 under `evidence/`. Their original paths and Git blob identities are recorded in [the snapshot inventory](../BASELINE-DOCUMENT-INVENTORY.json).

For the dated initial demonstration and early PPO improvement, see [demo history](DEMO-HISTORY.md). Its source identities are distinct from the later frozen research suite.

Read these first:

- [Final consolidation audit](FINAL-CONSOLIDATION-AUDIT.md) — frozen assets, archive hashes, historical restoration.
- [Historical provenance](../architecture/HISTORICAL-PROVENANCE.md) — why old records and current-source attestation are separate.
- [Final refactor audit](../architecture/FINAL-REFACTOR-AUDIT.md) — engineering validation limits.
- [Current operations](../operations/README.md) — implemented/not implemented, scientific gate boundaries.

**Do not rename, edit, reserialize, merge or delete source-bound artifacts, manifests, JSONL cell logs, frozen models, receipts, registered scientific IDs, or historical handoffs** without independent provenance review. Changing a path or even whitespace may invalidate recorded identities. Research documents not classified as immutable still represent history and must not be silently rewritten as current instruction.
