# Import and dependency audit — existing script entrypoints

Baseline: `b11a501`. Reviewed the source text of all **48 original Python entrypoints** through the GitHub connector without executing them or rewriting historical paths.

## Findings

- **0** active `from f1rl...` or `import f1rl...` statements in the 48 legacy scripts. They already use active `src.*` imports where project modules are required.
- Most scripts inject the repository root into `sys.path` to support invocation by absolute path. This is compatibility bootstrap, not clean reusable-package architecture. The **new** workflow router is a stdlib-only module and does not need this bootstrap.
- The original cross-script import from `scripts/run_inference.py` to `launch_exploratory_demo` has been replaced by a shared, versionless `src.interfaces.runtime_selection` adapter. The launcher retains its `_runtime_ready` / `_interpreter_candidates` compatibility hooks; both entrypoints use the same runtime-selection implementation.
- `scripts/build_demo_handoff.py` bundles named script paths as handoff inputs. Moving them would change packaged artifacts and their references.
- Several WP3/WP7 files contain explicit Windows defaults (`D:/...`), archived filenames or versioned scientific protocol strings. These must be classified as historical defaults/protocol identifiers before any portable CLI rewrite.
- Existing scripted workers, scientific receipt builders, source-diff allow-lists and external user commands may refer to original `scripts/wp*` paths. A text-only import scan cannot prove that no external caller exists.

## Runtime selection separation

`src/interfaces/runtime_selection.py` only discovers an interpreter and optionally delegates the caller's original command. It does **not** issue training permits, create experiments, or open scientific gates. The launcher continues to require explicit local `TRAIN` consent, and the inference script uses its existing episode/seed validation. Dedicated tests cover interpreter fallback, argument and cwd preservation, missing-runtime failure and historical monkeypatch contracts.

## Current decision

Do not mass-change `sys.path`, `Path(__file__)`, fixed historical run IDs or versioned file names in legacy evidence-bound entrypoints until call-site and acceptance tests exist. Newly supported commands are **versionless public aliases**, catalogued under `scripts/workflows/`, with original executable paths preserved. The adapter delegates through `subprocess` so its own imports are only standard-library imports; actual Python project dependencies remain owned by the original script.

## Next physical modularisation gate

1. Inventory direct script-to-script imports, script self-path inspection, subprocess launch targets and bundles/handoffs.
2. Introduce stable `src/` application APIs for any reusable code currently inside a script, with dedicated regression tests.
3. Separate interactive developer scripts from historically source-bound read-only record verifiers and trusted-pipe workers.
4. Move/retire a legacy entrypoint **only** once all callers, old receipt verification and new-source attestation remain correct.
5. Validate the complete Python suite, frontend, installed wheel, 12 checkpoints and historical preflight before merging.

This audit is static, not a claim that a test suite has passed.
