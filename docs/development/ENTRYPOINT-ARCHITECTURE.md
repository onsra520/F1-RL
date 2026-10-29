# Script entrypoint architecture

The public versionless CLI is:

```powershell
python -m scripts.workflows list
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows inference run-policy --help
```

The registry routes 48 stable commands to modules under functional packages. Package names and filenames are versionless except where a scientific contract identifier requires its version. The router forwards arguments and exit status; each command owns its existing admission and evidence behavior.

## Layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| CLI/router | `scripts/workflows/` | Stable aliases and module dispatch |
| Executable modules | `scripts/{diagnostics,evaluation,evidence,inference,training,internal}/` | Command-line parsing and orchestration |
| Domain/application | `src/` | Reusable implementation and validation |
| Research source records | `docs/research/` | Historical scientific identity and immutable receipts |

Executables are invoked with `python -m scripts.<package>.<module>`. No executable modifies `sys.path` or imports a sibling executable. The isolated trusted PPO worker uses module invocation with user-site and environment-path loading disabled.

Historical documentation may quote former root paths as provenance. Those recorded command lines are not active callers; current launchers, tests, configs, subprocesses, and handoffs use the grouped modules. Frozen evidence and admission records remain unchanged.
