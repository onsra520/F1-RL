# Executable scripts

All 46 active command entrypoints are grouped by responsibility:

```text
scripts/
  diagnostics/   environment and runtime diagnostics
  evaluation/    development evaluation and replay tools
  evidence/      frozen assets, manifests, and read-only audits
  inference/     saved-policy execution
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

The current production path runs through simulation, the Gymnasium environment, registered PPO training, evaluation and saved-policy inference. The retired one-click demo launchers and lecturer handoff are not part of the workflow registry. The local simulation command remains because the registered WP3 campaign invokes it as its training worker.
