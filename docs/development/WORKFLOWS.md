# Script workflows

The versionless interface dispatches grouped Python modules and preserves the existing command aliases, forwarded arguments, and exit codes. Each module retains its own scientific, training, and evidence checks.

From the repository root:

```powershell
python -m scripts.workflows list --json
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows evidence verify-historical-records --history D:/restored-history
python -m scripts.workflows diagnostics core-readiness --help
python -m scripts.workflows inference smoke-frozen-policy
```

`--help` after a command belongs to that module; use `list` to browse without executing a workflow.

## Execution scopes

- **read-only:** Does not intentionally retrain; some commands may inspect evidence or write to explicitly supplied paths.
- **development-execution:** Can create scratch rollouts or simulated episodes; these are not scientific evidence.
- **training-explicit:** Can request training; existing permit/admission logic and owner authorization remain mandatory.
- **restricted-execution:** Evaluation or validation workflows with hard scientific gates.
- **internal-only:** Trusted-pipe worker, unavailable through the dispatcher.

The router invokes the selected module from the repository root, forwards arguments and returns its exit code. It does not change project/evidence environment variables or grant training or HELD-OUT authorization.

## Shared implementation modules

Reusable behavior lives in `src/`. The package executables do not import sibling executable modules and do not modify `sys.path`. The trusted PPO worker uses isolated module invocation while disabling user-site and environment-path imports.

For current diagnostic and comparison module owners see [Development tools](DEVELOPMENT-TOOLS.md). The complete entrypoint inventory is in the [migration map](SCRIPT-MIGRATION-MAP.md); former root paths in that map are migration origins only. Historical research documents may retain original command spellings as provenance.
