# Executable scripts

All 48 command entrypoints are grouped by responsibility:

```text
scripts/
  diagnostics/   environment and runtime diagnostics
  evaluation/    development evaluation and replay tools
  evidence/      frozen assets, manifests, and read-only audits
  inference/     saved-policy execution and demo handoff
  training/      explicit local training and preflight tools
  internal/      trusted internal workers; not public workflows
  workflows/     stable CLI registry and dispatcher
```

Run commands as modules so imports resolve from the checkout without `sys.path` changes:

```powershell
python -m scripts.workflows list
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows inference run-policy --help
```

Every command keeps its existing scientific, training, and evidence checks. The internal trusted-pipe worker is not exposed by the public dispatcher.

See the [migration map](../docs/development/SCRIPT-MIGRATION-MAP.md), [entrypoint inventory](../docs/development/SCRIPT-STATUS.json), and [architecture notes](../docs/development/ENTRYPOINT-ARCHITECTURE.md).
