# Executable workflows

The 48 existing Python files in the top-level `scripts/` are compatibility entrypoints, **not** a clean, versionless public API. Their historical names and locations are intentionally preserved pending source-binding and subprocess migration review. They have not been renamed or physically moved in this stage.

A stable, module-based CLI now lives in `scripts/workflows/`:

```powershell
python -m scripts.workflows list
python -m scripts.workflows list --json
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows evidence verify-historical-records --history <restored-history>
python -m scripts.workflows evidence attest-source --root . --output <new-path-outside-repo>
python -m scripts.workflows inference smoke-frozen-policy
```

The registry classifies commands into `diagnostics`, `evidence`, `evaluation`, `inference`, `training`, and `internal`. See the [full mapping](../docs/development/SCRIPT-MIGRATION-MAP.md) and [implementation-completion inventory](../docs/development/SCRIPT-STATUS.json). Names are versionless; historical file paths, registered schemas and recorded IDs are not.

## Implementation boundaries

- `scripts/workflows/catalog.py` is a declarative inventory; importing it never runs a command.
- `scripts/workflows/__main__.py` dispatches explicitly requested commands in a new Python subprocess, with the current interpreter and unchanged argument order. It propagates the existing exit status; it does not open scientific gates or change the working directory.
- `internal/ppo-fixture-worker` is catalogued but **cannot** be invoked through the public CLI: its trusted-pipe pickle protocol is not suitable for manual inputs.
- Existing scripts continue to import `src.*` and retain their own project root, evidence path, admission and output safety logic. Centralising their aliases is not proof that all legacy import/bootstrap code is already modularised.
- Do not switch historical receipt or provenance paths to new names mechanically. The old receipt belongs to old source identity.

For allowed scripts and command examples use [development workflows](../docs/development/WORKFLOWS.md). There is currently no `scripts/train.py`, `scripts/evaluate.py` or `scripts/serve.py` implied by this documentation.
