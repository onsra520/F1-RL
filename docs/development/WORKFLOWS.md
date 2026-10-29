# Script workflows

The versionless interface calls unchanged compatibility scripts and preserves their CLI arguments/exit codes. This is **not** permission to start training or scientific evaluation.

From the repository root:

```powershell
python -m scripts.workflows list --json
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows evidence verify-historical-records --history D:/restored-history
python -m scripts.workflows diagnostics core-readiness --help
python -m scripts.workflows inference smoke-frozen-policy
```

`--help` after a command belongs to the original script; some historical entrypoints do not use `argparse` or may perform work at import time. Use `list` to browse without executing any script.

## Execution scopes

- **read-only:** Does not intentionally retrain; some commands still load/inspect evidence and may write new outputs to explicitly supplied paths. Check each command's own arguments.
- **development-execution:** Can create scratch rollouts or simulated episodes; not scientific evidence.
- **training-explicit:** Can request training; original permit/admission logic and human authorization remain mandatory.
- **restricted-execution:** Evaluation or validation workflows with hard scientific gates. Never use the router to infer that these gates are open.
- **internal-only:** Trusted-pipe worker. Intentionally unavailable through the dispatcher.

The router delegates to the existing Python interpreter with `subprocess.call`, does not change cwd, does not create a hidden process/session and does not modify `F1RL_PROJECT_ROOT` or `F1RL_WP3_EVIDENCE`. It cannot approve training or HELD-OUT.

## Compatibility

Original `python scripts/<legacy-file>.py` calls remain supported as before. Renaming all physical files is **deferred** until exact internal/external references, immutable source identity and regression tests are checked. Some script names are part of historical records, not current software version numbers.

See [complete alias inventory](SCRIPT-MIGRATION-MAP.md) and [historical provenance](../architecture/HISTORICAL-PROVENANCE.md).
