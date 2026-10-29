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

The full Python regression on the historical evidence checkout PASS: **3,063 passed, 3 skipped, 0 failed (31m17s)**. This supersedes the earlier incomplete run recorded below, which used a temporary reconstructed fixture and failed where original telemetry was unavailable. Frozen asset verification PASS (12 models, 7 reports, 96 cell logs); historical record verification PASS (2 records). PPO frozen-checkpoint inference smoke PASS (one 500-step TRAIN episode; timed out at the configured step limit without collision). Frontend checks from the prior follow-on remain 12/12 tests and build PASS.

## Pre-merge acceptance checklist (2026-10-29)

- [x] 48 grouped workflow entrypoints, zero root-level legacy Python wrappers.
- [x] Full Python regression: 3,063 passed, 3 skipped, 0 failed (31m17s), on original historical evidence.
- [x] Frozen assets and historical records verified; frozen PPO inference smoke completed without training.
- [x] Active workflow path updated to `python -m scripts.evaluation.review_owner_acceptance`; registry, docs, launchers and configs audited for active legacy paths. Historical transcripts remain unchanged.
- [x] Targeted entrypoint and documentation regressions, static validation, active link checks, and `git diff --check` pass on the final change.
- [ ] GitHub Actions execution (intentionally not run per request).
- [ ] HELD-OUT scientific evaluation (out of scope and not run).

No scripts-refactor blocker remains in active repository references. GitHub Actions execution and HELD-OUT evaluation remain unrun by instruction; those are explicit acceptance boundaries, not unresolved path defects.

## Grouped primary command modules

The 48 active commands are placed in versionless
`scripts/{diagnostics,evaluation,evidence,inference,internal,training}/`
packages and the `scripts.workflows` dispatcher invokes these paths directly.
There are zero root-level Python compatibility files under `scripts/` (apart
from `__init__.py`). `SCRIPT-STATUS.json` keeps former paths as historical
provenance; archived research records and the protected GitHub workflow retain
references that cannot be rewritten under the current scope.
