# Architecture decisions

- `src/` is the active Python package root. There is no `src/f1rl` nesting or compatibility alias.
- Domain responsibilities and module moves are listed in `MIGRATION-MAP.md`. The deterministic simulation layer does not depend on track persistence or RL/training/evaluation; import scans test both absolute and relative forms and verify the directories exist.
- Historical WP3 records and their original `f1rl/` source identities are immutable. Historical integrity verification and new source engineering attestations are separate operations. The legacy builders refuse to reissue historical receipts, including when matching source hashes are supplied. The WP7 historical source-diff allow-list remains unchanged and rejects migrated physics paths.
- The software-only WP7 readiness matrix and current source-provenance references use the moved paths. Historical scientific IDs, evaluator identity, RNG descriptors, models, evidence and scientific manifests retain their original values.
- Installed-wheel inference can use `F1RL_PROJECT_ROOT` to locate the independent checkout's configuration and track assets; the owner contract and parent SHA-256 are still verified. `F1RL_WP3_EVIDENCE` separately identifies restored historical telemetry.
- Shared YAML input validation lives in `src/training/config_loading.py`; vector and training configuration preserve their separate path-resolution and section contracts.
- The browser and Python runtime remain separate; no bridge was implemented.
