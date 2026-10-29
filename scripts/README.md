# Executable scripts — primary functional modules

The supported primary executables are organised by responsibility:

~~~text
scripts/
  diagnostics/       benchmark, diagnostic traces, hardware probe
  evaluation/        TRAIN-development driving comparison
  evidence/          frozen assets, SHA-256 manifests, read-only audits
  inference/         saved policy and demo handoff
  training/          scoped local simulation
  internal/          reserved for non-public workers (no public entrypoint)
  workflows/         stable CLI registry and explicit dispatcher
  *.py               historical compatibility paths
~~~

All 14 extracted implementation commands now have a **versionless primary
entrypoint** inside those packages. Their actual application/validation logic
lives in `src/`. The router calls the new grouped path.

~~~powershell
python -m scripts.workflows list
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows inference run-policy --help
~~~

The **48 root-level filenames remain** for compatibility with recorded research,
direct legacy commands, external subprocesses, handoffs and tests. Only 14 are
newly grouped primary modules; the remaining 34 still execute their existing
source-bound or already-delegating entrypoints. Do not claim 48/48 physically
migrated, and do not delete historical paths solely to unclutter the root.

No grouped package performs training, inference or validation merely from import.
The internal trusted-pipe PPO worker is not exposed through the public dispatcher.

See [the migration map](../docs/development/SCRIPT-MIGRATION-MAP.md),
[implementation ownership](../docs/development/SCRIPT-STATUS.json), and
[architecture notes](../docs/development/ENTRYPOINT-ARCHITECTURE.md).
