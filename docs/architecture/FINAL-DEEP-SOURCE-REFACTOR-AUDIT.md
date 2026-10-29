# Deep source refactor final audit

**Audit date:** 2026-10-30
**Branch:** `refactor/architecture-modularization`
**Verified test HEAD:** `d587377` (source code is unchanged from `92e7e22`; later commits are documentation-only)
**Status:** **NOT COMPLETE — Phase 5 has one open source-boundary item.** Phases 3–4 and their required regressions passed. Full pytest, frontend, static and frozen-evidence gates passed on the verified clean branch state. This is an engineering status only; WP7 scientific acceptance, training authority and HELD-OUT remain blocked.

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

However, WP7 `build_protocol` hashes committed bytes under `src/`, top-level `scripts/` and `tests/`, plus explicit project/config assets. It also embeds the current Git commit SHA through `GitProvenance`; the protocol digest is then used for evaluation seed derivation. Consequently any new commit can change the prospective protocol identity, and extracting the digest module also changes the explicitly bound source inventory/bytes. The registered historical manifests/seeds/evidence are separate immutable artifacts and have not been rewritten. The user's requirement to keep registered WP7 values exact means this source extraction cannot proceed under the current contract.

The user has confirmed that registered WP7 provenance, digests, seeds, manifests and frozen evidence must remain exact. Under the current protocol construction, moving the digest code changes the bound source inventory/bytes; a facade re-export preserves the callable API but not the old source-bound protocol value. The Git commit SHA is also part of each newly built protocol, so prospective identities vary by committed revision even for documentation-only commits. Therefore this extraction stays blocked; no code was changed in this continuation and no registered or historical artifact was regenerated. Revisit only if a future versioned protocol can retain current registrations as immutable records while adding the new module without rewriting their identities.

## Verification snapshot

Full pytest ran on clean committed HEAD `d587377`; its source files are unchanged from the Phase 3–5 implementation at `92e7e22`. This audit update changes documentation only and does not touch the enumerated source-file bindings. As noted above, a future protocol built at a later commit will record that commit SHA; no existing registration, manifest or evidence was regenerated.

| Gate | Result |
| --- | --- |
| Full pytest with `F1RL_WP3_EVIDENCE` set to the provisioned original evidence directory | PASS — 3,079 passed, 3 skipped, 0 failed on clean HEAD `d587377` (1,947.35 s) |
| Phase 3 lifecycle/checkpoint/admission/cleanup target | PASS — 45 passed |
| Phase 4 inference/pilot/pipeline target | PASS — 45 passed |
| Phase 5 WP7 protocol/profile/admission/scenario matrix | PASS for implemented behavior — 114 passed; it does not prove the pending module extraction |
| Inference using frozen A-seed0 checkpoint | PASS — TRAIN Straight, seed 41877, 500 steps, `valid_truncation/step_limit`, exit 0; row and trace SHA matched pre-refactor output |
| Frozen evidence verification | PASS — fresh run: 12 models, 7 reports, 96 cell logs; historical record checks verified two records and left `original_source_reconstructed=false` |
| Frontend tests / typecheck / production build | PASS — 12 tests; typecheck and build exited 0 (existing large-chunk advisory) |
| `compileall`, YAML validation and workflow registry | PASS — compileall; 11 YAML files; all 46 registry entries resolve |
| Changed-file Ruff F / import-cycle scan / dependency check | PASS — changed Python files clean; zero top-level import cycles; `uv pip check` found 26 compatible packages |
| Repository-wide Ruff F | Existing debt — 156 findings outside changed files; no repo-wide cleanup attempted |
| `git diff --check` | PASS on the prior source acceptance commit and the current documentation-only commits |

The earlier 83 pytest failures were reproduced as environment/working-tree issues: evidence-dependent tests lacked `F1RL_WP3_EVIDENCE`, and source-bound tests were run on a dirty tree. During this audit, an initial `uv run pytest` invocation created an untracked `uv.lock`; three WP2 source-binding tests correctly rejected that dirty tree. The generated lock was removed, the targeted WP2 file passed 11/11 on clean HEAD, and the full suite then passed using `.venv/Scripts/python.exe -m pytest -q` with the evidence path configured. No source guard or test was changed.

## Remaining technical debt

- **Medium:** Canonical WP7 digest implementation remains in the protocol facade. The user requires exact preservation of registered provenance, digests, seeds, manifests and evidence; the current whole-source binding makes extraction incompatible with that requirement, so this item remains blocked.
- **Medium:** `wp7_scenarios.py` is a broad but cohesive integration boundary spanning registered recovery, tracks, physics policies, Gymnasium factories and reward/navigation contracts.
- **Medium:** Generic and WP3 lifecycle orchestration remain distinct, intentionally; only the equivalent artifact hash primitive is shared.
- **Low:** Active WP3 code retains `LOCAL_DEMO`/`local_demo` identifiers; renaming requires a separate compatibility migration preserving immutable identifiers and historical allow-lists.
- **Low:** PPO admission has deferred imports. Moving authorization requires an explicit neutral policy interface and regression coverage for all allow/deny paths.
- **Existing repository lint debt:** 156 Ruff F diagnostics outside files changed by this work.

## Merge readiness

**Final Phase 3–5 engineering acceptance: BLOCKED.** The code and verification gates for completed work pass; the requested Phase 5 canonical-digest responsibility split is outstanding. Under the current source-bound protocol, it cannot preserve the exact registered identities required by the user. Keep it blocked unless a future compatibility design can add the boundary while retaining all current registered digests, seeds and manifests unchanged. Scientific acceptance and HELD-OUT authorization remain separate blocked decisions.
