# Deep source refactor final audit

**Audit date:** 2026-10-30
**Branch:** `refactor/architecture-modularization`
**Validation HEAD:** `ceea42375d2fa967f989c8b9af281c8e8424f355` (clean committed source and Option A documentation; this report update is documentation-only).
**Status:** **ENGINEERING REFACTOR COMPLETE (engineering acceptance).** Phase 5 is assessed as behavior-preserving with the accepted canonical-digest exception. All required regression, frontend, scoped Ruff, compile, dependency/import, CLI/YAML, inference and frozen-evidence gates passed on the validation HEAD. The supplemental isolated all-rule Ruff scan is not a configured project gate; its baseline debt and current findings are recorded below. The branch is engineering merge-ready. WP7 scientific acceptance, training authority and HELD-OUT remain separately blocked.

## Inherited baseline

Khánh Băng's WP3 handoff, 12 frozen model checkpoints and immutable evidence remain the baseline. The audited handoff SHA manifest verifies 73/73 files and the evaluation manifest verifies 203 raw Git blobs plus one Windows CRLF checkout-equivalent log. FULL-06 retains the same frozen 12 model hashes and verifies 96/96 TRAIN-development cells, 5,760 episodes and seven reports. No checkpoints were retrained and no historical evidence was rewritten. Historical provenance still reports `original_source_reconstructed=false`; this audit makes no stronger reconstruction claim.

## Phase results

| Phase | Result | Implementation and compatibility evidence |
| --- | --- | --- |
| 1 — Compatibility foundation | PASS | Campaign/evaluation golden artifacts, manifests, strict JSON, atomic/immutable writes, resume/recovery, telemetry cleanup, admission and inference behavior are covered. The clean-source-bound suite passed. |
| 2 — Campaign/evaluation split | PASS | Shared artifact/file primitives and focused campaign/evaluation modules removed the Campaign ↔ Evaluation cycle while preserving public entrypoints, schemas and resume behavior. |
| 3 — PPO lifecycle boundaries | PASS | Generic and WP3 lifecycle owners remain separate because admission, telemetry, evidence and failure-state contracts differ. Equivalent chunked artifact hashing is shared. Forty-five lifecycle/hash regressions passed. |
| 4 — Production inference API | PASS | Seeded read-only rollout API is shared by the inference CLI and exploratory pilot wrapper. Progress uses an observer rather than monkey-patching `env.step`. The frozen A-seed0 checkpoint reproduced the pre-refactor 500-step row and trace hash on TRAIN Straight; exit code was 0. |
| 5 — WP7 protocol boundaries | **BEHAVIOR-PRESERVING WITH DOCUMENTED EXCEPTION** | Registration/schema validation, scenario construction, runtime fingerprint and model artifact hashing are separated. Under Option A, `wp7_protocol.digest` remains unchanged in the compatibility facade; no extraction or canonical hashing change is required. Final WP7 regressions must pass for acceptance. |

## WP7 canonical digest — accepted architectural exception (Option A)

`src/validation/wp7_protocol.py` continues to define `digest(document)` alongside source binding, seed derivation and protocol orchestration. This is an intentional boundary exception: the public compatibility API and canonical JSON hashing implementation stay in place. No canonical hashing code, format, schema, registration, seed roster or admission guard is changed.

Historical provenance is immutable: Băng's model checkpoint bytes, historical manifests, recorded seeds, evaluation results and frozen evidence are not regenerated or rewritten. Prospective protocol identity is tied to Git HEAD: `build_protocol` captures the current commit SHA and hashes selected `src/`, top-level `scripts/`, `tests/` and project/config files. A protocol built at a newer commit may therefore carry a different prospective digest and derived evaluation seeds. This is expected source-bound identity, not a rewrite of historical evidence.

Option A closes the architectural decision by retaining this documented exception. It is not a scientific approval and does not alter the scientific admission state.

## Verification snapshot

Prior validation results below remain historical snapshots. The final execution/evidence checks below ran on clean committed validation HEAD `ceea42375d2fa967f989c8b9af281c8e8424f355`; this final audit update changes documentation only. No registered artifact is to be regenerated.

| Gate | Result |
| --- | --- |
| Full pytest with `F1RL_WP3_EVIDENCE` set to the provisioned original evidence directory | PASS — 3,079 passed, 3 skipped in 2,010.10s on clean validation HEAD `ceea423`; all skips require symlink privileges unavailable to this Windows process |
| Phase 3 lifecycle/checkpoint/admission/cleanup target | PASS — 45 passed |
| Phase 4 inference/pilot/pipeline target | PASS — 45 passed |
| Phase 5 WP7 protocol/profile/admission/scenario matrix | PASS for implemented responsibilities — focused matrix 114 passed and WP2/WP7 target 157 passed on the same source state; canonical digest remains in the facade by accepted Option A exception |
| WP2/WP7 source binding, profile, manifest, admission and seed target | PASS — 157 passed on source HEAD `27adea6`; subsequent commits through validation HEAD `ceea423` changed documentation only, and the current full suite passed |
| Inference using frozen A-seed0 checkpoint | PASS — TRAIN Straight, seed 41877, 500 steps, `valid_truncation/step_limit`, exit 0; row and trace SHA matched pre-refactor output |
| Frozen evidence verification | PASS — fresh run: 12 models, 7 reports, 96 cell logs; historical record checks verified two records and left `original_source_reconstructed=false` |
| Frontend tests / typecheck / production build | PASS — 12 tests; typecheck and build exited 0 (existing large-chunk advisory) |
| `compileall`, YAML validation and workflow registry | PASS — `compileall -q src scripts`; 12 YAML files parsed; all 46 registry entries resolve |
| Import graph / dependency check | PASS — 205 Python modules scanned, zero top-level import cycles; `uv pip check` found 26 compatible packages |
| Ruff `F` checks on the 11 refactor-touched Python files | PASS — all checks passed |
| Supplemental isolated all-rule Ruff scan (`ruff check --isolated --statistics .`) | Non-gating lint debt — 1,292 findings current vs 1,287 on baseline `a4aff02`; 45 findings among refactor-touched Python files vs 40 on baseline paths present there (scope differs because extraction introduced modules). The repository has no checked-in Ruff policy/configuration or Ruff CI command. The scoped Ruff `F` check on all 11 refactor-touched Python files passes. Findings include style rules and intentional exception/re-export seams; no bulk edits were made that could change compatibility contracts. |
| `git diff --check` | PASS — current audit report changes are clean |

The earlier 83 pytest failures were reproduced as environment/working-tree issues: evidence-dependent tests lacked `F1RL_WP3_EVIDENCE`, and source-bound tests were run on a dirty tree. During this audit, an initial `uv run pytest` invocation created an untracked `uv.lock`; three WP2 source-binding tests correctly rejected that dirty tree. The generated lock was removed, the targeted WP2 file passed 11/11 on clean HEAD, and the full suite then passed using `.venv/Scripts/python.exe -m pytest -q` with the evidence path configured. No source guard or test was changed.

The three full-suite skips were identified by `pytest -q -rs`: two symlink tests skip because Windows cannot create the requested symlinks, and the WP2 nonce-ledger test skips because Windows reports `WinError 1314` for directory symlinks. A focused rerun of the affected test files passed 66 tests and repeated only those three environment skips. They are not unexplained failures.

## Remaining technical debt

- **Accepted exception:** canonical WP7 digest remains in the compatibility facade under Option A; future protocol identities follow Git HEAD while historical records remain immutable.
- **Medium:** `wp7_scenarios.py` is a broad but cohesive integration boundary spanning registered recovery, tracks, physics policies, Gymnasium factories and reward/navigation contracts.
- **Medium:** Generic and WP3 lifecycle orchestration remain distinct, intentionally; only the equivalent artifact hash primitive is shared.
- **Low:** Active WP3 code retains `LOCAL_DEMO`/`local_demo` identifiers; renaming requires a separate compatibility migration preserving immutable identifiers and historical allow-lists.
- **Low:** PPO admission has deferred imports. Moving authorization requires an explicit neutral policy interface and regression coverage for all allow/deny paths.
- **Non-blocking Ruff debt:** the scoped `F` check on refactor-touched Python files passes. The supplemental all-rule scan is noisy at baseline and current (1,287 vs 1,292 findings) and has no checked-in project policy; import ordering, broad exception catches, legacy calls and deliberate exception/re-export compatibility seams remain candidates for separately reviewed cleanup.

## Merge readiness

**Final engineering acceptance: PASS; Engineering Refactor COMPLETE.** Option A resolves the Phase 5 boundary as a documented architectural exception. Full regression (with the three explained Windows symlink skips), frontend, scoped Ruff `F`, compile/import/dependency, workflow registry, inference and frozen-evidence gates pass on clean validation HEAD `ceea423`. The supplemental all-rule Ruff scan remains documented lint debt and is not a configured project gate. The branch is ready for engineering merge review; no merge was performed. No source, Physics/PPO behavior, scientific contract, frozen evidence, model, hash or seed was changed during final acceptance. WP7 scientific acceptance and HELD-OUT authorization remain independently blocked.
