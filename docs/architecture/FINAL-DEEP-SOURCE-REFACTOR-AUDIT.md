# Deep source refactor final audit

**Audit date:** 2026-10-30
**Branch:** `refactor/architecture-modularization`
**Verified source HEAD:** `92e7e22` (the branch through audit baseline `227f7c0` contains documentation-only commits after this source revision)
**Status:** **NOT COMPLETE — Phase 5 has one open source-boundary item.** Phases 3–4 and their required regressions passed. The full regression, frontend, static and frozen-evidence gates passed on the verified source HEAD. This is an engineering status only; WP7 scientific acceptance, training authority and HELD-OUT remain blocked.

## Inherited baseline

Khánh Băng's WP3 handoff, 12 frozen model checkpoints and immutable evidence remain the baseline. The audited handoff SHA manifest verifies 73/73 files and the evaluation manifest verifies 203 raw Git blobs plus one Windows CRLF checkout-equivalent log. FULL-06 retains the same frozen 12 model hashes and verifies 96/96 TRAIN-development cells, 5,760 episodes and seven reports. No checkpoints were retrained and no historical evidence was rewritten. Historical provenance still reports `original_source_reconstructed=false`; this audit makes no stronger reconstruction claim.

## Phase results

| Phase | Result | Implementation and compatibility evidence |
| --- | --- | --- |
| 1 — Compatibility foundation | PASS | Campaign/evaluation golden artifacts, manifests, strict JSON, atomic/immutable writes, resume/recovery, telemetry cleanup, admission and inference behavior are covered. The clean-source-bound suite passed. |
| 2 — Campaign/evaluation split | PASS | Shared artifact/file primitives and focused campaign/evaluation modules removed the Campaign ↔ Evaluation cycle while preserving public entrypoints, schemas and resume behavior. |
| 3 — PPO lifecycle boundaries | PASS | Generic and WP3 lifecycle owners remain separate because admission, telemetry, evidence and failure-state contracts differ. Equivalent chunked artifact hashing is shared. Forty-five lifecycle/hash regressions passed. |
| 4 — Production inference API | PASS | Seeded read-only rollout API is shared by the inference CLI and exploratory pilot wrapper. Progress uses an observer rather than monkey-patching `env.step`. The frozen A-seed0 checkpoint reproduced the pre-refactor 500-step row and trace hash on TRAIN Straight; exit code was 0. |
| 5 — WP7 protocol boundaries | **PARTIAL / BLOCKED** | Registration/schema validation, scenario construction, runtime fingerprint and model artifact hashing are separated. `wp7_protocol.digest` still owns canonical JSON hashing; details below. |

## Phase 5 blocker: canonical digest ownership

`src/validation/wp7_protocol.py` still defines `digest(document)` alongside source binding, seed derivation and protocol orchestration. Callers import this function through the module facade, so the API can remain compatible if the implementation is extracted.

However, WP7 `build_protocol` hashes the committed bytes under `src/`, `scripts/` and `tests/`. Moving the implementation and adding/adjusting tests changes those source bytes. The digest algorithm may remain byte-equivalent, but newly built protocol digests and derived evaluation seeds are expected to change because the protocol explicitly binds the source tree. This is prospective provenance, and historical records/evidence cannot be rewritten to mask it.

The requested source change is therefore stopped under the instruction not to alter scientific contracts or source provenance without an approved plan. No code was changed in this continuation. To resume this item safely, the protocol owner must define whether the existing source binding intentionally treats this refactor as a new prospective protocol identity and how new manifests/seeds are versioned. Then extract the unchanged canonical JSON hash primitive behind the existing `wp7_protocol.digest` re-export and verify compatibility, source binding, profile identities, golden manifests, seeds and fail-closed paths. Historical manifests and immutable evidence must remain byte-for-byte unchanged.

## Verification snapshot

The recorded checks are for clean committed source `92e7e22`. Documentation-only commits after that revision, including this audit update, do not touch protocol-bound source paths.

| Gate | Result |
| --- | --- |
| Full pytest with `F1RL_WP3_EVIDENCE` set to the provisioned original evidence directory | PASS — 3,079 passed, 3 skipped, 0 failed (1,866.46 s) |
| Phase 3 lifecycle/checkpoint/admission/cleanup target | PASS — 45 passed |
| Phase 4 inference/pilot/pipeline target | PASS — 45 passed |
| Phase 5 WP7 protocol/profile/admission/scenario matrix | PASS for implemented behavior — 114 passed; it does not prove the pending module extraction |
| Inference using frozen A-seed0 checkpoint | PASS — TRAIN Straight, seed 41877, 500 steps, `valid_truncation/step_limit`, exit 0; row and trace SHA matched pre-refactor output |
| Frozen evidence verification | PASS — 12 models, 7 reports, 96 cell logs; historical record checks verified two records and left `original_source_reconstructed=false` |
| Frontend tests / typecheck / production build | PASS — 12 tests; typecheck and build exited 0 (existing large-chunk advisory) |
| `compileall`, YAML validation and workflow registry | PASS — compileall; 11 YAML files; all 46 registry entries resolve |
| Changed-file Ruff F / import-cycle scan / dependency check | PASS — changed Python files clean; zero top-level import cycles; `uv pip check` found 26 compatible packages |
| Repository-wide Ruff F | Existing debt — 156 findings outside changed files; no repo-wide cleanup attempted |
| `git diff --check` | PASS on the source acceptance commit and this documentation-only delta |

The earlier 83 pytest failures were reproduced as environment/working-tree issues: evidence-dependent tests lacked `F1RL_WP3_EVIDENCE`, and source-bound tests were run on a dirty tree. With the original evidence path configured and a clean committed tree, the full suite passed as recorded above.

## Remaining technical debt

- **Medium:** Canonical WP7 digest implementation remains in the protocol facade, blocked on an explicit prospective provenance/versioning decision.
- **Medium:** `wp7_scenarios.py` is a broad but cohesive integration boundary spanning registered recovery, tracks, physics policies, Gymnasium factories and reward/navigation contracts.
- **Medium:** Generic and WP3 lifecycle orchestration remain distinct, intentionally; only the equivalent artifact hash primitive is shared.
- **Low:** Active WP3 code retains `LOCAL_DEMO`/`local_demo` identifiers; renaming requires a separate compatibility migration preserving immutable identifiers and historical allow-lists.
- **Low:** PPO admission has deferred imports. Moving authorization requires an explicit neutral policy interface and regression coverage for all allow/deny paths.
- **Existing repository lint debt:** 156 Ruff F diagnostics outside files changed by this work.

## Merge readiness

**Final Phase 3–5 engineering acceptance: BLOCKED.** The code and verification gates for completed work pass; the requested Phase 5 canonical-digest responsibility split is outstanding. The full refactor must not be declared complete until an approved provenance plan is applied, the compatibility and WP7 protocol gates pass on a clean committed tree, and the resulting final audit is updated. Scientific acceptance and HELD-OUT authorization remain separate blocked decisions.
