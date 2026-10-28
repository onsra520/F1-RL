# Architecture decisions

- The requested Python package root is `src`, not `src/f1rl`; do not create an implicit `f1rl` alias.
- Preserve model ZIPs, evidence and historical protocol IDs byte-for-byte. Keep the scientific gate fail-closed and retain the active absorbing roll-envelope behavior.
- Historical source-bound checks need a separately verified migration strategy: `wp3_v4_technical_acceptance.FILES` names 21 paths under `f1rl/`, hashes the current files at those names, and emits those paths in `source_bindings_sha256`. The legacy replay and receipts are immutable; merely substituting paths can change the source-binding record. The safe design must distinguish historical verification from new source attestation without relabeling or weakening the existing contract.
- The React/Three.js sandbox and Python runtime are separate; no bridge is implemented or authorized in this phase.
