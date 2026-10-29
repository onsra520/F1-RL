# Scripts and documentation standardisation — change report

## Branch and scope

- Base: `refactor/architecture-modularization` at `b11a501b029680f79762388ef6b179eda97141a7`.
- Change branch: `refactor/scripts-docs-standardization` (not merged).
- **Completed in this change:** standard public CLI routing, inventory and classification, current documentation navigation, corrected project status/setup guidance, compatibility map and focused tests.
- **Not completed:** physical relocation/deletion of the 48 original scripts or historical documentation, full internal code extraction from every legacy CLI, a full Python regression run on this change branch.

## Script and import strategy

The supported active Python imports remain `src.*`. Workflow dispatch runs each original executable as a subprocess with the exact original path and argument order. It never imports historical CLI modules into the dispatcher, renames their Python serialization identity, rewrites scientific paths or interprets grants/admission flags.

Because historical scripts are referenced by provenance, external handoffs, worker subprocesses and audit allow-lists, the original 48 paths are retained byte-for-byte. Their physical relocation requires a separate referenced-call-site and scientific compatibility audit, not a mechanical rename. `internal/ppo-fixture-worker` is discoverable but blocked for manual invocation.

## Documentation

Snapshot of 324 baseline Git-tracked docs with Git blob identities:

- 153 immutable evidence records
- 107 historical research records
- 44 dated historical plans/specifications
- 11 current architecture references
- 5 historical architecture diagrams/artifacts
- 4 top-level historical/index documents

New current development, operations, research and historical-design indexes avoid rewriting original records. The root README no longer claims FastAPI or WebSocket integration is working.

## Validations and limitations

GitHub-only static checks: 48 unique aliases map to 48 existing scripts, no duplicate aliases or version suffixes, zero changed/deleted legacy scripts, zero changed/deleted historical evidence, 48 migration-map entries, new relative Markdown references resolve, no baseline files removed.

Focused tests are added at `tests/scripts/test_workflows.py`, including manifest completeness, non-executing listing, internal command denial, subprocess argument preservation and document link checks. **Runtime tests, full Python suite, frontend npm suite and scientific preflight have not been run from this GitHub-only authoring environment**. Do not represent the prior `b11a501` acceptance as a new-branch test pass.

Recommended local review before manual merge:

```powershell
python -m pytest -q tests/scripts/test_workflows.py
python -m scripts.workflows list --json
python -m scripts.workflows evidence verify-frozen-assets
python -m pytest -q
cd web
npm ci
npm test
npm run typecheck
npm run build
```

The historical restore/check and symlink tests require the same external inputs/platform capability as the previously accepted baseline. No GitHub Actions, model training, scientific evaluation or HELD-OUT access was executed.
