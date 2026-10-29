# Deep source refactor final audit

**Audit date:** 2026-10-30
**Branch:** `refactor/architecture-modularization`
**Acceptance HEAD:** pending clean committed validation after this report update.
**Status:** **ENGINEERING ACCEPTANCE IN PROGRESS.** Option A is accepted: Phase 5 is assessed as behavior-preserving with a documented canonical-digest exception. Do not mark Engineering Refactor COMPLETE until every final gate in this report passes. WP7 scientific acceptance, training authority and HELD-OUT remain separately blocked.

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

Prior validation results below remain historical snapshots. Final acceptance checks are being rerun against a clean commit that contains this Option A documentation; results and the exact acceptance HEAD will be added after completion. No registered artifact is to be regenerated.

| Gate | Result |
| --- | --- |
| Full pytest with `F1RL_WP3_EVIDENCE` set to the provisioned original evidence directory | PENDING final clean-HEAD run |
| Phase 3 lifecycle/checkpoint/admission/cleanup target | PASS — 45 passed |
| Phase 4 inference/pilot/pipeline target | PASS — 45 passed |
| Phase 5 WP7 protocol/profile/admission/scenario matrix | PASS for the implemented responsibilities — 114 passed; canonical digest remains in the facade by accepted Option A exception |
| Final clean-HEAD WP2/WP7 source binding, profile, manifest, admission and seed target | PENDING final clean-HEAD run |
| Inference using frozen A-seed0 checkpoint | PASS — TRAIN Straight, seed 41877, 500 steps, `valid_truncation/step_limit`, exit 0; row and trace SHA matched pre-refactor output |
| Frozen evidence verification | PASS — fresh run: 12 models, 7 reports, 96 cell logs; historical record checks verified two records and left `original_source_reconstructed=false` |
| Frontend tests / typecheck / production build | PASS — 12 tests; typecheck and build exited 0 (existing large-chunk advisory) |
| `compileall`, YAML validation and workflow registry | PASS — compileall; 11 YAML files; all 46 registry entries resolve |
| Changed-file Ruff F / import-cycle scan / dependency check | PASS — changed Python files clean; zero top-level import cycles; `uv pip check` found 26 compatible packages |
| Repository-wide Ruff F | Existing debt — 156 findings outside changed files; no repo-wide cleanup attempted |
| `git diff --check` | PENDING final report commit |

The earlier 83 pytest failures were reproduced as environment/working-tree issues: evidence-dependent tests lacked `F1RL_WP3_EVIDENCE`, and source-bound tests were run on a dirty tree. During this audit, an initial `uv run pytest` invocation created an untracked `uv.lock`; three WP2 source-binding tests correctly rejected that dirty tree. The generated lock was removed, the targeted WP2 file passed 11/11 on clean HEAD, and the full suite then passed using `.venv/Scripts/python.exe -m pytest -q` with the evidence path configured. No source guard or test was changed.

## Remaining technical debt

- **Accepted exception:** canonical WP7 digest remains in the compatibility facade under Option A; future protocol identities follow Git HEAD while historical records remain immutable.
- **Medium:** `wp7_scenarios.py` is a broad but cohesive integration boundary spanning registered recovery, tracks, physics policies, Gymnasium factories and reward/navigation contracts.
- **Medium:** Generic and WP3 lifecycle orchestration remain distinct, intentionally; only the equivalent artifact hash primitive is shared.
- **Low:** Active WP3 code retains `LOCAL_DEMO`/`local_demo` identifiers; renaming requires a separate compatibility migration preserving immutable identifiers and historical allow-lists.
- **Low:** PPO admission has deferred imports. Moving authorization requires an explicit neutral policy interface and regression coverage for all allow/deny paths.
- **Existing repository lint debt:** 156 Ruff F diagnostics outside files changed by this work.

## Merge readiness

**Final engineering acceptance: IN PROGRESS.** Option A resolves the Phase 5 boundary as a documented architectural exception. The branch is merge-ready only if the final full regression, frontend, Ruff, compile/import/dependency, workflow registry, inference and frozen-evidence gates all pass on the clean committed validation state. If they pass, mark Engineering Refactor COMPLETE; scientific acceptance and HELD-OUT authorization remain separate blocked decisions.
