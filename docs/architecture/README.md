# Architecture documentation

The active Python package is `src/`. The [source-backed domain map](domain-map.html), [dependency map](DEPENDENCY-MAP.md), [migration map](MIGRATION-MAP.md) and [architecture decisions](ARCHITECTURE-DECISIONS.md) describe the completed package migration and remaining coupling. Earlier baseline diagrams are historical snapshots.

- [Final refactor audit](FINAL-REFACTOR-AUDIT.md): engineering acceptance of package migration at its recorded source revision; current script cleanup status is in the [change report](../development/SCRIPTS-DOCS-CHANGE-REPORT.md).
- [Refactor validation](REFACTOR-VALIDATION.md): Python, models, wheel, frontend and historical evidence checks.
- [Deep source-code audit](DEEP-SOURCE-CODE-AUDIT.md): current Phase 1–5 findings, verification state and remaining debt.
- [Deep source refactor final audit](FINAL-DEEP-SOURCE-REFACTOR-AUDIT.md): current Phase 3–5 gate results, provenance blocker and remaining merge-readiness work.
- [Historical provenance](HISTORICAL-PROVENANCE.md): immutable source identity vs current engineering attestation.
- [Current operations](../operations/README.md): UI bridge still not implemented, scientific admission blocked.
- [Script migration map](../development/SCRIPT-MIGRATION-MAP.md): stable versionless command aliases and historical path mappings.

Note: this navigation update does not itself re-run the original full Python suite or claim scientific acceptance.
