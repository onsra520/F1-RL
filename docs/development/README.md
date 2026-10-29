# Development

The supported Python import root is `src`. Use Python 3.11+ and install the project from the repository root using the declared `pyproject.toml` extras needed for your workflow.

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q tests/scripts/test_workflows.py
python -m scripts.workflows list
python -m scripts.workflows evidence verify-frozen-assets
```

A wheel can be installed in a separate environment. Workflows requiring registered project configuration and track assets may need `F1RL_PROJECT_ROOT` pointing to the source checkout; `F1RL_WP3_EVIDENCE` is a **different** variable for restored historical telemetry.

See [Workflows](WORKFLOWS.md) for safety and invocation, [script migration map](SCRIPT-MIGRATION-MAP.md) for old entrypoints, [script import audit](SCRIPT-IMPORT-AUDIT.md) for the shared runtime-selection module and historical boundaries, and [change report](SCRIPTS-DOCS-CHANGE-REPORT.md) for tested/untested scope.

For developer-only comparisons, profiling and hardware probing, see [Development tools](DEVELOPMENT-TOOLS.md). The [script status inventory](SCRIPT-STATUS.json) distinguishes fully extracted implementations from aliases that still point at legacy code.

For frozen/model-evidence and historical source checks, see [Evidence verification](../operations/EVIDENCE-VERIFICATION.md).

Frontend is built in `web/`:

```powershell
cd web
npm ci
npm test
npm run typecheck
npm run build
```

The UI currently does not drive the authoritative Python Physics/PPO runtime. It is not appropriate to claim backend integration from a successful frontend build.
