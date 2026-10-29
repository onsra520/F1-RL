# Scripts & Docs refactor — engineering change report

Branch: refactor/scripts-docs-standardization
Base: refactor/architecture-modularization, b11a501.

## Scope completed

- Stable versionless router and 48-entry inventory; original paths retained
  for historical source binding, handoffs, subprocess and user compatibility.
- Extraction of selected implemented CLI bodies into reusable src interfaces,
  training, evaluation and validation modules; explicit repository root is
  passed by the demo, inference and audit compatibility wrappers.
- Cross-script dependency between inference and demo launcher removed.
- Independent source validity/consent gates, immutable evidence, original
  checkpoints, scientific protocol IDs and physics math are unchanged.
- Documentation indexes, active setup/operations, demo history, script owners,
  evidence verification, reference-audit decisions and compatibility contracts
  consolidated. Original dated research records remain available.

## Protected boundaries

Original science-bound script paths and schemas are not bulk-renamed; the
trusted fixture worker is internal-only. All 48 script paths remain. See
SCRIPT-STATUS.json for honest extracted/shared/preexisting/guarded counts and
ENTRYPOINT-ARCHITECTURE.md for what is intentionally not migrated.

## Validation record

- User-confirmed previous batches: 31 targeted tests PASS at 47ac14d;
  frozen models 12, reports 7, cell logs 96 PASS; clean worktree.
- Remote baseline after 01009dd: 79 focused tests PASS.
- After demo/packaging extraction: 79 focused tests PASS and 6 dedicated
  entrypoint regressions PASS (separate calls).
- Full Python suite, new audit-wrapper tests, frontend and independent wheel:
  consult the latest terminal test report and commit before claiming PASS.
- The original migration baseline 3,011 tests is **not** a test result for
  this new refactor branch.

## Review and merge policy

Do not merge until source checkout is clean, all new focused tests and full
Python suite are checked, active docs links are valid, historical/frozen
asset verification has passed, and frontend build has been rerun or explicitly
recorded as not run. Scientific HELD-OUT remains BLOCKED, independently of
the engineering result. No GitHub Actions or training are used for this audit.

Use git log for the exact source commit; no report retroactively replaces the
source SHA of a historical run.

## Follow-on root cleanup (2026-10-29; starting HEAD `8071957`)

The earlier report above records the state of the initial extraction. This follow-on moved all 48 workflow commands into functional packages, removed the 14 redundant root shims, and moved the remaining 34 executable implementations. `scripts/` now has zero root-level Python compatibility files. The registry and callers use module execution; reusable behavior is in `src/`, with no executable-to-executable imports or `sys.path` mutation.

CLI aliases, argument forwarding, admission checks, frozen assets, and historical evidence remain intact. Three scientific contract filenames retain their required `v1` identifier. Archived research, plans, and generated architecture snapshots keep their original command text as provenance.

Validation at the follow-on commit: Python compilation and 12 entrypoint/inventory/documentation tests PASS. Earlier, Windows Application Control blocked `torch\\lib\\c10.dll`; on 2026-10-29 the worktree `.venv` imported `torch 2.15.0+cpu`, so the full suite ran. The initial full run reported 80 failed, 2,983 passed, 3 skipped; the failures were concentrated in historical WP3 tests that expect the absent external `F1RL_WP3_EVIDENCE` tree. With a temporary fixture reconstructed from the committed roster, provenance receipts, and frozen model ZIPs, the focused historical subset reached 95 passed, 1 failed: one admission test requires original `training-telemetry.jsonl` bytes matching the retained SHA-256, and those telemetry files are absent from repository evidence. No telemetry was fabricated. Frozen asset verification PASS (12 models, 7 reports, 96 cell logs); historical record verification PASS (2 records). PPO frozen-checkpoint inference smoke PASS (one 500-step TRAIN episode; timed out at the configured step limit without collision). Frontend checks from the prior follow-on remain 12/12 tests and build PASS. The active GitHub workflow reference to removed `scripts/wp7_review_owner_acceptance.py` remains unresolved because this task prohibits editing GitHub Actions.

## Grouped primary command modules

The 48 active commands are placed in versionless
`scripts/{diagnostics,evaluation,evidence,inference,internal,training}/`
packages and the `scripts.workflows` dispatcher invokes these paths directly.
There are zero root-level Python compatibility files under `scripts/` (apart
from `__init__.py`). `SCRIPT-STATUS.json` keeps former paths as historical
provenance; archived research records and the protected GitHub workflow retain
references that cannot be rewritten under the current scope.
