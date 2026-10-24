# WP3 Issue #51 — Owner-authorized TRAIN-development full run registration

**Status of this package: RUN REGISTRATION ONLY — not final evaluation results.**

- Run ID: `WP3-OWNER-TRAINDEV-20261024-FULL-01`
- Locked execution source: `bbefa1b2e0261b34cd826f896be3cfb838ecdc90`
- Source branch: `fix/wp3-eval-physical-completion`
- Frozen TRAIN source: `fe83f517479f3b990028d6409304fae83b8199a9`
- Physics: `spatial_road_edge_partial_lifecycle_v4`
- Crash Terminal contract: `wp3_physical_crash_terminal_v1`
- Original paired TRAIN roster: exactly **96 cells / 5,760 episodes**; 12 PPO seed-final models, original seed pairs.
- Registered owner-development plan content SHA-256: `0de1e88811fd8651d97c205466fefb4113826a9b053c2ce951a6a200a88c71c5`
- Immutable file `RUN-MANIFEST.json` records all 5,760 original seed pairs plus original model file hashes and source.
- Host-local new run evidence directory: `D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261024`
- New run was started with three constrained CPU workers, with per-cell append-only JSONL containing `episode_started` and `episode_completed` or `episode_invalid`.
- Scientific Physics v4 independent review is still **PENDING**. This is **OWNER-APPROVED TRAIN DEVELOPMENT** only; do not claim a scientific registered result, and do not unseal HELD-OUT.

## Inspect while running on the authorized Windows machine

```powershell
cd D:\F1-RL-WP3-EVAL-PHYSICAL-COMPLETION
& 'D:\F1-RL-WP3\.venv\Scripts\python.exe' scripts\wp3_owner_train_dev_eval.py status --out D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261024
Get-Content D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261024\RUN-CONSOLE.txt -Tail 20
```

Progress is not a final candidate ranking. A successful, completely run controller writes the six reports/charts and output SHA manifest; interrupted attempts must be preserved as invalid, not retried.
