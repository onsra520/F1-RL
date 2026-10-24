# WP3 Issue #51 — Owner-authorized TRAIN-development Evaluation

## Authorization and boundary

The project owner expressly approved proceeding with a TRAIN Evaluation on 2026-10-24. The **owner development** authorization is not a reviewer signature and does not satisfy the separate scientific release gate.

- Scientific full Evaluation gate: **BLOCKED_PENDING_INDEPENDENT_REVIEW**, unchanged
- Owner-authorized full TRAIN-development evaluation: independently versioned, **permitted with explicit --owner-approved**
- New model training: NO; HELD-OUT: NO; original historic evaluation: IMMUTABLE
- Physics `spatial_road_edge_partial_lifecycle_v4`, crash contract `wp3_physical_crash_terminal_v1`
- Final PPO models: 12, 65,536 timesteps each; 4 original TRAIN tracks, 2 strata, 60 paired seed slots, original 4m boundary
- The 96-cell plan preserves **exact original frozen roster ordering**, including model order

## CLI commands

Run from a clean committed checkout on the Windows machine, with `D:\F1-RL-WP3\.venv\Scripts\python.exe`:

```powershell
python scripts/wp3_owner_train_dev_eval.py prepare --out D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261024 --run-id WP3-OWNER-TRAINDEV-20261024-FULL-01 --owner-approved
python scripts/wp3_owner_train_dev_eval.py status --out D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261024
python scripts/wp3_owner_train_dev_eval.py run --out D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261024 --workers 2
# Only if the previous run stopped and no controller is running:
python scripts/wp3_owner_train_dev_eval.py resume --out D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261024 --workers 2
```

## Evidence contract

Preparation exclusively creates `RUN-MANIFEST.json` and `RUN-STATE-INITIAL.json` in a new root (no overwrite), verifies SHA of all 12 final checkpoint ZIPs and the complete source-bound original roster. Each cell gets an append-only JSONL (started then completed or invalid) with the *original frozen paired environment/policy seeds*, no redraw and no run on HELD-OUT. A failed runtime is a retained invalid record that aborts its own cell; legitimate crash/hard-route/timeout events are complete **failed driving episodes**, not finish.

Workers are restricted to 1–3 because the current Windows host has 16GB total RAM and only ~4GB free at preflight. Source must remain clean. Control lock prevents duplicate evaluator workers. When a controller stops normally, it writes summary, per-cell breakdown, candidate comparison **only if all 96 cells have 60 valid episodes**, statistical readiness, 3 SVGs and a SHA report. An interrupted run can be resumed *without resampling a started slot*; dangling started records become invalid and abort the cell. A stale controller lock after machine failure must be checked and cleared only after confirming no evaluator process is active.

## Scientific limitations

No independent reviewer signature, experimental material/impact calibration, or formal WP7/Gate 2 approval is asserted. The existing registered scientific Eval admission remains fail-closed. Development evaluation cannot substitute a signed scientific result. Three independent PPO training seeds per candidate are insufficient for confirmatory model superiority or causal reward/curriculum claims.
