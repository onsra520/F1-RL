# Script import and dependency audit

The 48 workflow commands now live in functional packages under `scripts/`. Reusable application, validation, and parsing logic belongs in `src/`; commands are started with `python -m scripts.<package>.<module>`.

## Current findings

- The public registry has 48 unique commands and routes each to an existing grouped module.
- The 14 former root compatibility shims were removed after their grouped implementation became the sole command entrypoint. The other 34 command implementations were moved into their functional packages.
- Executable modules do not write to `sys.path` and do not import sibling executable modules. The private PPO fixture worker and source-pinned feasibility worker use module invocation with user-site and environment-path loading disabled.
- Demo handoff manifests, Windows launchers, source subprocesses, tests, and current operator documentation use grouped module paths.
- Package filenames omit work-package and software-version prefixes. Contract identifiers such as `crash_terminal_v1` remain where they identify registered scientific evidence.
- Scientific admission, permit checks, frozen-model bytes, evidence receipts, historical source records, and exit behavior remain governed by their existing `src/` implementations.

## Historical provenance

Archived research notes, dated plans, generated architecture snapshots, and immutable evidence may quote commands such as `scripts/wp3_...py` or `scripts/wp7_...py`. These are historical command transcripts, not active path dependencies. Their contents were not rewritten. The [migration map](SCRIPT-MIGRATION-MAP.md) records each former path and its current module, plus the GitHub Actions reference that remains blocked by the task's explicit no-Actions constraint.

The repository does not include external handoff consumers; the supported replacement is the stable `python -m scripts.workflows <group> <action>` interface. The trusted internal worker remains inaccessible through that public registry.
