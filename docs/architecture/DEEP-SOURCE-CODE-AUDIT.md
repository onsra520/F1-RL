# Deep source code refactor audit

**Current status (2026-10-30):** Phases 1–4 are implemented and verified on source commit `92e7e22`. Phase 5 is **partial**: registration/schema validation, scenario construction and runtime capture have separate modules, but canonical digest still lives in `wp7_protocol.py`. The user requires existing registered WP7 provenance, digests, seeds, manifests and frozen evidence to remain exact. Canonical digest extraction changes source bytes included in WP7 source binding and therefore cannot meet that compatibility requirement with the current protocol construction. The extraction remains blocked; no historical or registered artifacts were regenerated. **Final Phase 3–5 acceptance is not granted.**
**Prior accepted verification snapshot:** The complete suite and static/evidence checks below passed on source commit `92e7e22`; the current HEAD `227f7c0` adds documentation only. See [`FINAL-DEEP-SOURCE-REFACTOR-AUDIT.md`](FINAL-DEEP-SOURCE-REFACTOR-AUDIT.md) for the current gate table and blocker. The generic/WP3 runner split, remaining WP7 scenario breadth, retired campaign identifiers and deferred PPO admission imports remain technical debt. No claim of scientific acceptance, HELD-OUT permission or completion of unrelated repository debt is made.
**Prior audit baseline:** e82efd8cdc6465af83d40d1eb552b1d82e22d840. Phase 3–5 implementation began from clean commit a4aff02aef39cb8cfaf9a69f50c7544e1079c500 on refactor/architecture-modularization.
**Scope:** WP3 campaign/evaluation source, adjacent tests and entrypoints, dependencies/configuration, architecture docs, and the frontend boundary. No training, HELD-OUT, Actions, Physics, PPO algorithm behavior, or frozen evidence changes.

## Pipeline trace

| Stage | Current implementation and entrypoint | Audit result |
| --- | --- | --- |
| Simulation | `src.core.simulation.Simulation`, `src.simulation`; library API | Domain implementation with a small core adapter boundary. |
| Gymnasium | `src.rl.gym_environment.F1RLEnv`, built by `src.training.env_factory` | Expected dependency from Gymnasium adapter into simulation/domain packages. |
| PPO training | Generic `src.training.TrainingRunner`; registered WP3 worker uses `scripts.training.local_simulation` → `src.interfaces.local_demo_cli` → `src.interfaces.runner.run_local_demo` | Two live paths. The campaign path has its own admission, telemetry and integrity contracts and must remain. |
| Evaluation | `src.evaluation.evaluator.evaluate_episode`; WP3 orchestration in `src.interfaces.evaluation` via `evaluation coordinate` | WP3 uses the shared episode evaluator and owns roster, resume, report and evidence handling. |
| Inference | scripts.inference.run_policy → src.interfaces.inference_cli → src.evaluation.rollout.evaluate_policy_episode | Read-only seeded rollout; keeps the established row schema, invalid-sample mapping, progress events and exit codes. Straight remains the CLI registered track. |

The frontend remains a visual editor/replay sandbox. Its documented lack of authoritative physics/evaluation and a Python bridge is an intentional boundary.

## Phase 1 — compatibility foundation

Regression coverage now pins the frozen campaign config and schedule, the exact evaluation roster hashes, six report byte hashes, strict JSON behavior, exclusive/atomic/immutable writes, completed and incomplete campaign resume, interrupted evaluation recovery, telemetry cleanup, admission guards, and inference output/exit codes. No training was needed to exercise these paths.

## Phase 2 — campaign and evaluation split

The former cross-import is removed. `evaluation.py` no longer imports Campaign helpers. Shared file primitives live in `_artifacts.py`; campaign contract, per-run validation and campaign report/evidence writing live in separate modules; evaluation roster, event records and reports also have separate modules. `campaign.py` is 424 lines and `evaluation.py` is 482 lines. The existing lazy Campaign → Evaluation call remains one-way and loads PPO evaluation only after training; no cycle remains.

Golden hashes and regression tests passed after extraction. Public campaign/evaluation entrypoints, report schemas, resume decisions, atomic writes and error handling remain covered. The existing `_source_identity` algorithm and scientific source-binding lists were not edited. New campaign manifests continue to bind the committed tree as before.

## Phase 3 — PPO lifecycle boundary

The generic src.training.TrainingRunner and WP3 campaign supervisor in src.interfaces.runner were audited together. They share checkpoint/lifecycle vocabulary but enforce different contracts: the generic runner owns run-state and observers; the WP3 supervisor owns fixed campaign admission, curriculum, telemetry/integrity, and cleanup. A shared orchestration layer would risk moving a guard across the pre-allocation boundary or changing checkpoint and failure-state semantics, so those owners remain separate. The exactly equivalent chunked SHA-256 file helper in the campaign runner now delegates to the existing internal artifact hash primitive; 45 runner lifecycle/hash regressions passed. Existing tests cover checkpoint scheduling, actual-timestep names, state, cleanup/failure, admission denial before allocation, and evidence/source identity.

## Phase 4 — inference API

src.evaluation.rollout.evaluate_policy_episode now owns the stable seeded read-only rollout row and accepts an explicit step observer. The exploratory pilot evaluate_episode is a compatibility wrapper over it. src.interfaces.inference_cli creates one seeded environment at a time and uses the same API; progress reporting no longer patches env.step. Tests pin terminal and invalid-transition rows, environment cleanup, progress callbacks and CLI exit mapping. A real frozen A-seed0 inference smoke on TRAIN Straight (seed 41877) returned the same 500-step valid_truncation/step_limit row and trace SHA-256 as the pre-refactor evaluator from a4aff02; CLI exit code was 0. This verifies rollout compatibility, not model performance or scientific acceptance.

## Phase 5 — WP7 protocol boundaries

Host fingerprint and model ZIP hashing live in src/validation/wp7_runtime. Registration loading/exact schema validation now live in src/validation/wp7_registration; registered scenario construction lives in src/validation/wp7_scenarios; wp7_protocol re-exports the established API and retains canonical digest, seed binding, runtime-manifest and validation entrypoints. Admission consumes the public runtime fingerprint API instead of importing _runtime from wp7_protocol; compatibility seams remain for restore callers. Registration/profile identities, scenario factories, manifests and denial decisions passed the WP7 target matrix (114 tests). The protocol source_files and digest naturally describe the current committed tree; historical manifests/evidence were not rewritten. Canonical digest extraction remains pending because changing the source tree changes prospective protocol digests and derived seeds.

## Remaining findings

No Critical or High finding was confirmed.

### Medium

1. **Generic and WP3 lifecycle owners remain separate.** src/training/runner.py owns generic run-state and observers; src/interfaces/runner.py owns WP3 admission, telemetry, integrity, cleanup and checkpoint contracts. They are not interchangeable. The exact file hashing duplicate is removed; no shared orchestration helper has matching side effects and failure semantics. **Impact:** future lifecycle fixes may drift. **Next step:** extract a helper only if a future comparison proves equivalent inputs, artifact effects and failure handling.

2. **WP7 scenario construction remains a broad integration boundary.** src/validation/wp7_scenarios.py intentionally joins registered recovery, tracks, physics policies, Gymnasium factories and reward/navigation contracts. Registration validation and runtime capture are separate; wp7_protocol remains the compatibility/protocol-binding facade. **Impact:** scenario edits can change the newly generated source-bound protocol digest and derived evaluation seeds. **Next step:** leave it cohesive until a smaller factory boundary can be extracted with pinned per-profile physics/reset identity; never rewrite historical protocol records.
3. **Canonical protocol digest is not independently owned.** `wp7_protocol.digest` still implements canonical JSON hashing in the compatibility facade. **Impact:** protocol orchestration and canonical serialization share a module boundary. `build_protocol` hashes all `src/**/*.py`, top-level `scripts/*.py`, `tests/**/*.py` and explicitly listed project/config assets; the top-level protocol digest then feeds evaluation seed derivation. Moving the implementation changes that bound source inventory/bytes, so a facade re-export alone cannot preserve the current registered protocol digest and derived seeds. **Next step:** leave this extraction blocked while exact registered provenance is required; only revisit if a future versioned protocol can preserve old registrations as immutable records without rewriting them.

### Low

1. **Active campaign code keeps retired demo names.** src/interfaces/local_demo_cli.py, src/interfaces/runner.py, telemetry identifiers and tests retain LOCAL_DEMO/local_demo, though WP3 calls them in production. **Impact:** contributors may delete active campaign code as demo residue. **Next step:** rename only as a separate compatibility migration that retains frozen identifiers and provenance allow-lists.

2. **PPO admission imports remain deferred.** src.rl.algorithms.ppo dynamically consults validation, interface and exploratory admission helpers. **Impact:** static import scans can miss the policy dependency. **Next step:** do not move authorization until every allow/deny path has an explicit neutral policy interface and regression tests.

No dead production module or removable dependency has enough evidence for deletion. The Campaign ↔ Evaluation cycle remains removed. WP7 compatibility facade imports point one-way to registration/scenario modules; no new module-level cycle was introduced. The deferred PPO admission imports remain a deliberate trust-boundary seam.

## Inherited evidence and verification status

- Băng's frozen training source is `fe83f517479f3b990028d6409304fae83b8199a9`. The fetched handoff branch (`97ac293`, `306d428`, `e6a5758`, `2ae51b6`) contains transferred checkpoints/manifests/recovery guidance and audited TRAIN evaluation results; it is not an executable resume HEAD. The handoff SHA manifest verifies **73/73** files. The evaluation manifest verifies **203 raw Git blobs plus one Windows CRLF checkout-equivalent log**; no evidence mismatch remains.
- The handoff records **12/12 engineering-validated PPO models**, **786,432** total timesteps, and a partial evaluation of **2,153 valid episodes**, **65 invalid cells**, and **3,542 unattempted positions**. It explicitly leaves canonical campaign reconciliation and scientific acceptance blocked. Later FULL-06 reuses the same frozen 12 model hashes and reports **96/96 TRAIN-development cells, 5,760 valid episodes, zero invalid**, without granting scientific acceptance. No training was repeated and no model/evidence file was changed.
- The separate October 19 demo-policy report is historical and records **0/8 successful finishes**; it is not used as a replacement for Băng's WP3 checkpoints or as a current pipeline baseline.

- Before source edits, targeted baseline: **141 passed**. The external WP3 evidence suite initially lacked its configured path; the files were found at `D:/Projects/.evidence/F1-RL/WP3-EVIDENCE/wp3-eval-issue51-v1`.
- After extraction: targeted regression **151 passed**. With the evidence path set, external admission/history checks passed **84 tests**. The three clean-source binding tests were rerun after commit and passed (**3 passed, 8 deselected**).
- `compileall -q src scripts` passed. Frozen asset verification passed (**12 models, 7 reports, 96 cell logs**). Ruff `F` checks passed. Full Ruff reports 25 style findings across the touched modules; the two original modules had 32 before refactoring. Error classes, timestamp formatting and loop semantics were left as-is.
- A fresh full suite on clean committed `a27f0c8` with `F1RL_WP3_EVIDENCE` set to the provisioned original evidence directory passed: **3,068 passed, 3 skipped in 1,938.31s**. This resolves the previous **2,984 passed, 83 failed, 3 skipped** report: the evidence-dependent tests lacked the environment path, and three source-binding tests ran against an intentionally dirty tree. The complete suite now passes when both conditions are satisfied.
- After the Phase 5 runtime extraction, the focused CPU/protocol/admission/restore regression passed **51 tests**. The first full run exposed six tests that relied on the old `wp7_protocol.platform` and `_cpu_model` module seams; those aliases now point to the extracted implementation. The clean-commit rerun passed **3,074 tests, 3 skipped in 1,724.33s**.
- `compileall`, Ruff `F`, frozen asset verification (**12 models, 7 reports, 96 cell logs**) and `git diff --check` passed after the extraction. No source file in Physics/PPO algorithms, config contract, model checkpoint or evidence inventory was changed.
- No immutable evidence, Physics, PPO behavior, scientific admission contract, schema, or source-hash algorithm was edited. No scientific training, HELD-OUT, or GitHub Actions ran.

## Phase 3–5 continuation and verification snapshot (not final acceptance)

- Phase commits: 878a53e (shared exact artifact hashing primitive), a821a23 (read-only rollout API and inference CLI), ff2be5d (WP7 registration/scenario split), 3f10490 (WP7 facade whitespace cleanup), and 92e7e22 (explicit public facade re-exports). Documentation update is committed separately.
- Phase 3 runner/checkpoint/admission/cleanup tests: 45 passed. Phase 4 inference/pilot/pipeline tests: 45 passed. Phase 5 WP7 protocol/profile/admission/scenario integration matrix: 114 passed. Clean source-binding tests passed 101 tests with one intentionally deselected simulation-heavy admission case before the final re-export spelling; that case's three spatial policy seeds passed in isolation afterward.
- One intermediate targeted run, concurrent with the frontend build, reported a source-protocol SHA drift for spatial seed index 2. A clean isolated rerun passed all three seeds, and the final full suite below passed the same cases on an unchanged committed tree.
- Full pytest on clean source commit 92e7e22, with F1RL_WP3_EVIDENCE set to the provisioned original directory: 3,079 passed, 3 skipped in 1,866.46 seconds. This is the final full regression for the Phase 3–5 source commits.
- A real frozen A-seed0 checkpoint inference smoke on TRAIN Straight with seed 41877 completed 500 steps as valid_truncation/step_limit, returned exit code 0, and produced the exact same output row and trace SHA-256 as the pre-refactor evaluator from a4aff02. This verifies API/output compatibility, not successful driving or model quality.
- Static import graph: zero module-level cycles. Changed Python files pass Ruff F checks. The repository-wide isolated Ruff F scan reports 156 findings, all outside files changed by this work; they are existing repository lint debt. No whole-repository lint cleanup was attempted.
- Compileall passed. YAML validation passed for 11 files; all 46 workflow registry entries resolve. uv pip check reports all 26 installed packages compatible. Frontend tests: 12 passed; typecheck and production build passed (Vite reports the existing large-chunk advisory).
- Frozen evidence verification passed: 12 models, 7 reports and 96 cell logs. Historical record verification passed for two records and correctly reports original_source_reconstructed=false. git diff --check passed.
- WP7 protocols bind source-file bytes and the current commit SHA, so future protocol digests/seeds identify each committed revision. Historical registered manifests, seeds, immutable evidence, model checkpoints and the source-hash algorithm were not edited. The canonical digest module extraction remains blocked by the exact-compatibility instruction.
- No training, HELD-OUT, GitHub Actions, Physics changes or PPO algorithm changes were made. The refactor does not grant WP7 scientific acceptance, registered evaluation authority, or HELD-OUT permission.

## Final clean-tree rerun

- On clean committed test HEAD `d587377`, using `.venv/Scripts/python.exe -m pytest -q` and the provisioned `F1RL_WP3_EVIDENCE` path, the full suite passed: **3,079 passed, 3 skipped in 1,947.35s**. A preceding `uv run pytest` created an untracked `uv.lock`; three WP2 source-binding tests correctly failed closed on that dirty tree. After removing only the generated untracked lock, the targeted WP2 source-binding file passed **11/11**, and the direct-interpreter full rerun passed. No guard or test was changed.
- On the same committed source state, `compileall -q src scripts` passed; fresh frozen-asset verification passed (**12 models, 7 reports, 96 cell logs**); fresh historical-record verification returned `HISTORICAL_RECORD_INTEGRITY_VERIFIED` for two records and `original_source_reconstructed=false`.

This verification snapshot distinguishes engineering checks from scientific acceptance. Phases 3–4 passed; Phase 5 and final architecture acceptance remain **BLOCKED** on canonical digest ownership/provenance handling. WP7 scientific acceptance, registered evaluation authority and HELD-OUT permission remain independently blocked.
