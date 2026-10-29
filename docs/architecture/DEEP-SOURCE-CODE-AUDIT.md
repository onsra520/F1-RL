# Deep source code refactor audit

**Phase status:** Phase 1 and Phase 2 implementation and targeted verification complete. The whole refactor is not complete and merge readiness is not declared.
**Baseline:** `e82efd8cdc6465af83d40d1eb552b1d82e22d840` on `refactor/architecture-modularization`.
**Scope:** WP3 campaign/evaluation source, adjacent tests and entrypoints, dependencies/configuration, architecture docs, and the frontend boundary. No training, HELD-OUT, Actions, Physics, PPO algorithm behavior, or frozen evidence changes.

## Pipeline trace

| Stage | Current implementation and entrypoint | Audit result |
| --- | --- | --- |
| Simulation | `src.core.simulation.Simulation`, `src.simulation`; library API | Domain implementation with a small core adapter boundary. |
| Gymnasium | `src.rl.gym_environment.F1RLEnv`, built by `src.training.env_factory` | Expected dependency from Gymnasium adapter into simulation/domain packages. |
| PPO training | Generic `src.training.TrainingRunner`; registered WP3 worker uses `scripts.training.local_simulation` → `src.interfaces.local_demo_cli` → `src.interfaces.runner.run_local_demo` | Two live paths. The campaign path has its own admission, telemetry and integrity contracts and must remain. |
| Evaluation | `src.evaluation.evaluator.evaluate_episode`; WP3 orchestration in `src.interfaces.evaluation` via `evaluation coordinate` | WP3 uses the shared episode evaluator and owns roster, resume, report and evidence handling. |
| Inference | `scripts.inference.run_policy` → `src.interfaces.inference_cli` → `src.exploratory.pilot.evaluate_episode` | Callable but still backed by exploratory demo code and limited to Straight. |

The frontend remains a visual editor/replay sandbox. Its documented lack of authoritative physics/evaluation and a Python bridge is an intentional boundary.

## Phase 1 — compatibility foundation

Regression coverage now pins the frozen campaign config and schedule, the exact evaluation roster hashes, six report byte hashes, strict JSON behavior, exclusive/atomic/immutable writes, completed and incomplete campaign resume, interrupted evaluation recovery, telemetry cleanup, admission guards, and inference output/exit codes. No training was needed to exercise these paths.

## Phase 2 — campaign and evaluation split

The former cross-import is removed. `evaluation.py` no longer imports Campaign helpers. Shared file primitives live in `_artifacts.py`; campaign contract, per-run validation and campaign report/evidence writing live in separate modules; evaluation roster, event records and reports also have separate modules. `campaign.py` is 424 lines and `evaluation.py` is 482 lines. The existing lazy Campaign → Evaluation call remains one-way and loads PPO evaluation only after training; no cycle remains.

Golden hashes and regression tests passed after extraction. Public campaign/evaluation entrypoints, report schemas, resume decisions, atomic writes and error handling remain covered. The existing `_source_identity` algorithm and scientific source-binding lists were not edited. New campaign manifests continue to bind the committed tree as before.

## Remaining findings

No Critical or High finding was confirmed.

### Medium

1. **Generic and WP3 training duplicate parts of the PPO lifecycle.** `src/training/runner.py` handles generic training; `src/interfaces/runner.py` (about 803 lines) owns campaign admission, telemetry, integrity, cleanup and checkpoint handling. They are not interchangeable because the WP3 runner enforces a registered simulation profile and campaign evidence contracts. **Impact:** shared lifecycle fixes may drift. **Refactor:** later define the smallest common execution primitive with explicit admission/telemetry policy, and migrate behind the existing campaign tests without changing schemas or gates.

2. **WP7 protocol construction and runtime capture share a high-fanout module.** `src/validation/wp7_protocol.py` imports track, physics, simulation, Gymnasium factory, PPO provenance and reward/navigation contracts while handling registration, protocol digests, scenario factories and runtime manifests. `src/validation/wp7_admission.py` composes protocol, feasibility, source checks and owner review. **Impact:** protocol edits can affect many runtime layers. **Refactor:** separate pure registration/digest validation from scenario factories and runtime capture, pinning canonical bytes, hashes, profile names and admission decisions first.

3. **Inference still uses an exploratory evaluator.** `src/interfaces/inference_cli.py` builds a local session, patches `env.step` for progress output and calls `src.exploratory.pilot.evaluate_episode`; only Straight is selected. `src.evaluation.evaluator.evaluate_episode` has a different result contract. **Impact:** the stable command depends on demo behavior and has a legacy result row. **Refactor:** define a read-only policy rollout API and compatibility output tests before changing the CLI or evaluator.

### Low

4. **Active campaign code keeps retired demo names.** `src/interfaces/local_demo_cli.py`, `src/interfaces/runner.py`, telemetry identifiers and tests retain `LOCAL_DEMO`/`local_demo`, though WP3 calls them in production. **Impact:** contributors may delete active campaign code as demo residue. **Refactor:** rename in a separate compatibility migration; retain frozen identifiers and provenance allow-lists.

5. **Deferred admission imports still obscure some RL layering.** `src.rl.algorithms.ppo` dynamically consults validation, interface and exploratory admission helpers. The `src.interfaces` Campaign ↔ Evaluation cycle is gone; this remaining edge is separate. **Impact:** import scans can miss policy dependencies. **Refactor:** move authorization to a neutral policy boundary only with explicit tests for every current allow/deny path.

No dead production module or removable dependency has enough evidence for deletion. The static import pass found no remaining Campaign ↔ Evaluation cycle. Dynamic imports outside that pair still need import tests.

## Delivery and verification status

- Before source edits, targeted baseline: **141 passed**. The external WP3 evidence suite initially lacked its configured path; the files were found at `D:/Projects/.evidence/F1-RL/WP3-EVIDENCE/wp3-eval-issue51-v1`.
- After extraction: targeted regression **151 passed**. With the evidence path set, external admission/history checks passed **84 tests**. The three clean-source binding tests were rerun after commit and passed (**3 passed, 8 deselected**).
- `compileall -q src scripts` passed. Frozen asset verification passed (**12 models, 7 reports, 96 cell logs**). Ruff `F` checks passed. Full Ruff reports 25 style findings across the touched modules; the two original modules had 32 before refactoring. Error classes, timestamp formatting and loop semantics were left as-is.
- The full pytest run before commit reported **2,984 passed, 83 failed, 3 skipped**. Those failures were traced to the missing `F1RL_WP3_EVIDENCE` path and clean-committed-source checks on the intentionally modified worktree. The affected evidence-dependent groups passed when rerun with the path set; the three clean-source checks passed after commit. A single full-suite rerun on the clean committed tree was not performed, so the report does not claim a one-shot full-suite pass.
- No immutable evidence, Physics, PPO behavior, scientific admission contract, schema, or source-hash algorithm was edited. No scientific training, HELD-OUT, or GitHub Actions ran.

This report tracks Phase 1–2 only. Phases 3–5 remain future work; do not treat this report as a refactor completion or merge-readiness claim.
