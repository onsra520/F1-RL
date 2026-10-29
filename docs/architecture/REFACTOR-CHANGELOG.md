# Refactor changelog

- Baseline `a909a25`: engineering-accepted consolidation.
- `b86379f`: source-backed architecture map, baseline file/hash inventory and migration decisions. No application behavior changed.
- `1336538`: completed the package migration from `f1rl/` to the active `src/` domain packages. The source move and import updates were regression-tested; historical source identities, scientific contracts and frozen evidence remain unchanged.
- `8071957` and follow-up commits: completed the grouped script entrypoint migration. All 48 active commands now resolve through functional packages and the `scripts.workflows` dispatcher; former command spellings remain only as historical provenance. See `docs/development/SCRIPTS-DOCS-CHANGE-REPORT.md` for the latest validation record.
