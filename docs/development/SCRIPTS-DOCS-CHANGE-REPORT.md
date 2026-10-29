# Scripts and documentation standardisation — change report

## Branch and scope

- Base: `refactor/architecture-modularization` at `b11a501b029680f79762388ef6b179eda97141a7`.
- Change branch: `refactor/scripts-docs-standardization` (not merged).
- **Completed in this change:** standard public CLI routing, inventory and classification, current documentation navigation, corrected project status/setup guidance, compatibility map and focused tests.
- **Not completed:** physical relocation/deletion of the 48 original scripts or historical documentation, full internal code extraction from every legacy CLI, a full Python regression run on this change branch.

## Script and import strategy

The supported active Python imports remain `src.*`. Workflow dispatch runs each existing compatibility executable as a subprocess with the exact original path and argument order. The dispatcher does not import other executable scripts, change scientific identifiers or interpret admission flags. Individual CLI files can be reduced to wrappers while retaining their original path and arguments.

Because historical scripts are referenced by provenance, external handoffs, worker subprocesses and audit allow-lists, **all 48 original paths remain present**, but selected non-frozen implementations have been extracted and their wrappers updated. This is *not* a claim that all 48 script bytes are unchanged. Physical relocation of source-bound entrypoints requires a separate call-site/provenance audit. `internal/ppo-fixture-worker` remains blocked through the public router.

## Documentation

Snapshot of 324 baseline Git-tracked docs with Git blob identities:

- 153 immutable evidence records
- 107 historical research records
- 44 dated historical plans/specifications
- 11 current architecture references
- 5 historical architecture diagrams/artifacts
- 4 top-level historical/index documents

New current development, operations, research and historical-design indexes avoid rewriting original records. The root README no longer claims FastAPI or WebSocket integration is working.

## Script import inventory

Static read of all 48 historical entrypoints found **no active `f1rl` Python import**. Legacy per-script `sys.path` bootstrap, Windows historical defaults and handoff path bundling remain for compatibility. The sibling-script import in `run_inference.py` was removed through the shared `src.interfaces.runtime_selection` adapter; see [SCRIPT-IMPORT-AUDIT.md](SCRIPT-IMPORT-AUDIT.md).

## Validations and limitations

GitHub-only static checks: 48 unique aliases map to 48 existing script **paths**, no duplicate aliases/version suffixes, no deleted legacy script, no altered immutable research evidence, 48 migration-map entries, and current navigation links resolve. Eight original script entrypoints have been updated across extraction and shared-dependency batches; this is intentional and covered by new focused tests.

Focused tests exist in `tests/scripts/`. The owner independently ran **15 passed in 8.67s** on commit `e496d44` (workflow, extraction, and historical restore tests) and verified `PASS: 12 models, 7 reports and 96 cell logs`, with a clean worktree. **The owner subsequently reported `22 passed in 4.71s` at `1724ee5`**, with frozen assets verified and a clean worktree. Full Python suite, frontend npm suite, and scientific preflight have not been rerun on this branch; do not represent migration-baseline acceptance as new-branch acceptance.

Recommended local review before manual merge:

```powershell
python -m pytest -q tests/scripts/test_workflows.py tests/scripts/test_extracted_workflows.py tests/scripts/test_runtime_selection.py tests/research/test_demo_launcher.py tests/research/test_wp3_historical_restore.py
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

## Extracted implementation modules (second batch)

- `scripts/benchmark_vector_envs.py` is now a compatibility wrapper around `src/training/vector_benchmark.py`. This removes the benchmark's heavy training/simulation implementation from the executable script while keeping the same command-line options.
- `scripts/verify_consolidated_wp3_assets.py` is now a compatibility wrapper around `src/validation/frozen_asset_integrity.py`. The engine takes an explicit project root, verifies original model/report/cell bytes without mutating evidence, and returns counts for display.
- `scripts/restore_wp3_historical_evidence.py` no longer imports `verify_consolidated_wp3_assets` as a sibling executable; it calls the same `src.validation.frozen_asset_integrity.verify_frozen_assets` API. Archive pinning and scientific fail-closed preflight remain unchanged.
- Focused regression tests were added for actual frozen evidence, both legacy and versionless CLI, benchmark help, invalid arguments, and missing manifests. **No full 3,011-test execution is claimed from GitHub.**

## Runtime selector modularisation (third batch)

- The local-demo launcher and inference script now share `src/interfaces/runtime_selection.py`; no script needs to import another executable for Python environment detection.
- The launcher preserves the original `ensure_runtime`, `_runtime_ready`, and `_interpreter_candidates` extension points, so the existing `tests/research/test_demo_launcher.py` remains meaningful.
- The selector uses the original interpreter candidates, dependency checks, original command and cwd; it does not start training unless the entrypoint would have done so after its pre-existing admission process.
- `tests/scripts/test_runtime_selection.py` covers fallback and argument preservation, fail-closed missing-runtime selection, launcher monkeypatch compatibility and inference's invalid-argument path.
- The owner reported `22 passed in 4.71s` for `1724ee5`, including this runtime-selector batch and the existing launcher test, plus verified frozen assets and a clean worktree.

## Development-tool modularisation (fourth batch)

- Extracted non-scientific comparison and diagnostic implementations from `scripts/compare_driving.py` and `scripts/diagnose_driving.py` into `src/evaluation/driving_comparison.py` and `src/evaluation/driving_diagnostics.py`, while keeping their historical script paths, exported helper names and existing CLI arguments.
- Extracted CUDA probe into `src/interfaces/hardware_probe.py` and kept the original `wp7_capture_gpu.py` path as a thin wrapper.
- Added `tests/scripts/test_development_tools.py` covering fixed seed/improvement thresholds, scripted action, `--help` parity for old and versionless CLI, and GPU JSON output parity. Tests do not train or run actual evaluation episodes.
- Created [development diagnostics reference](DEVELOPMENT-TOOLS.md) with accurate module and CLI ownership.
- **Tests introduced by this fourth batch are NOT RUN** on the new HEAD. The earlier 22/22 PASS belongs to `1724ee5`.

## Completion inventory after development-tool extraction

The [machine-readable script status](SCRIPT-STATUS.json) tracks all **48** original CLI paths: **5** have their substantive implementation extracted into `src/` and are now thin wrappers, **3** retain implementation but use a shared `src/` dependency, and **40** remain legacy compatibility commands with unchanged implementation. The 48 stable aliases and all historical paths are preserved. This prevents calling the alias-only stage a complete refactor.

The owner independently reported `22 passed in 4.71s` and immutable frozen-assets verification at `1724ee5`. **The new development-tool and status inventory tests at later commits remain NOT RUN.**
