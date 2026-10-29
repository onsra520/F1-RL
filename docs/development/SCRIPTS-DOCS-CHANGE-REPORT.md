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

## Grouped primary command modules

The 14 extracted active commands are now placed in versionless
`scripts/{diagnostics,evaluation,evidence,inference,training}/` packages and
the `scripts.workflows` dispatcher invokes these paths directly. Root-level
legacy filenames remain as compatibility shims, forwarding the original helper
interface and exhibiting the same CLI help/exit behavior.

`SCRIPT-STATUS.json` contains the primary and historical path separately.
Protected historical research and worker entries retain their exact old paths;
this is a deliberate compatibility boundary, not a falsely completed migration.
