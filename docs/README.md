# Documentation index

This is the **current navigation layer**. The repository also contains historical plans, science registrations, read-only research evidence and earlier handoff documents; these must not be mass-renamed to conform to today's naming scheme.

| Source of truth | Location | Scope |
| --- | --- | --- |
| Current setup and CLI | [Development](development/README.md) | Supported project layout and versionless workflow commands |
| Current operational status | [Operations](operations/README.md) | Working components, non-implemented UI bridge, historical restore |
| Python architecture | [Architecture](architecture/README.md) | Domain map, decisions, migration and validated limitations |
| Frozen research and provenance | [Research](research/README.md) | Scientific gates, immutable evidence, historical IDs |
| Historic proposals/specifications | [Design history](superpowers/README.md) | Snapshots, not up-to-date implementation instructions |
| Baseline document classification | [Snapshot inventory](BASELINE-DOCUMENT-INVENTORY.json) | 324 Git-tracked docs from `b11a501`; Git blob SHAs (not SHA-256) |
| Script rename/compatibility | [Script migration map](development/SCRIPT-MIGRATION-MAP.md) | 48 legacy entrypoints → stable versionless command aliases |
| Script refactor completion | [Script status inventory](development/SCRIPT-STATUS.json) | 5 extracted / 3 shared dependencies / 40 legacy commands |

## Document classification

- **Current:** development, operations and current architecture indexes; maintain against real source paths.
- **Historical design:** `docs/superpowers/`, dated top-level project plans; retain chronology and superseded assumptions.
- **Historical research:** `docs/research/`; records and identifiers refer to original source/evaluation procedures.
- **Immutable evidence:** `docs/research/evidence/`; never rewrite, rename or rehash casually.
- **Baseline maps:** earlier diagrams, inventories and acceptance notes are reference snapshots, not current application behavior.

The inventory is a **path-and-blob snapshot** for safe reviews. Classification is a navigation aid, not permission to delete a document. Search external references and research manifests before any eventual move.
