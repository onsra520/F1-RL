# Historical evidence recovery

This describes the historical technical preflight, **not** authorization for a new registered evaluation.

The two immutable release archives and their pinned SHA-256 checksums are documented in [FINAL-CONSOLIDATION-AUDIT.md](../research/FINAL-CONSOLIDATION-AUDIT.md). Supply genuine local archive paths (do not substitute a junction to a temporary reconstruction).

```powershell
python -m scripts.workflows evidence restore-history --full <full-zip> --continuation <continuation-zip> --destination <new-directory>
$env:F1RL_WP3_EVIDENCE = "<new-directory>"
python -m scripts.workflows evidence verify-historical-records --history "<new-directory>"
python -m scripts.workflows evidence verify-frozen-assets
```

Restore destination must be new; verify available disk space, archive hashes, original roster, model hashes and telemetry before calling research preflight. A full checkout contains small historical inputs and checkpoints, **not** the 3.29 GB raw telemetry.

The expected scientific-gate result remains **BLOCKED**. Historical record integrity does not reconstruct the exact uncommitted authoring source. Current engineering attestation must be generated independently against the current clean source checkout.
