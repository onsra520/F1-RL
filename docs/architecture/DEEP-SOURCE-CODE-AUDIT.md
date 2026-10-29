# Deep source code refactor audit

**Status:** Audit in progress; refactor completion and merge readiness are not declared.
**Audited source:** `6f75b077a9154384a240a67865688211e9cd4a29` on `refactor/architecture-modularization`.
**Scope:** Python source, script registry and executable adapters, tests, configuration/dependency declarations, architecture and workflow documentation, plus the `web/src` frontend boundary. This audit is static and targeted; no training, HELD-OUT work, GitHub Actions, or physics/PPO behavior changes were performed.

## Pipeline trace

| Stage | Current implementation and entrypoint | Audit result |
| --- | --- | --- |
| Simulation | `src.core.simulation.Simulation`, `src.simulation`; library API | Clear domain entry; `src/core` adapts the detailed `src/simulation` model. |
| Gymnasium | `src.rl.gym_environment.F1RLEnv`, built by `src.training.env_factory` | Real Gymnasium adapter. It depends on the simulation/domain packages as expected. |
| PPO training | Generic `src.training.TrainingRunner`; the registered WP3 campaign routes through `scripts.training.local_simulation` → `src.interfaces.local_demo_cli` → `src.interfaces.runner.run_local_demo` | Two execution paths exist. The campaign path remains active because it enforces its own admission, telemetry, integrity and campaign artifact contracts. `LOCAL_DEMO` naming is historical and obscures its current campaign role. |
| Evaluation | `src.evaluation.evaluator.evaluate_episode`; registered WP3 work is orchestrated in `src.interfaces.evaluation` and exposed through `evaluation coordinate` | WP3 delegates episode measurement to the shared evaluator; its campaign-level roster, resumability, reports and evidence are specialized. |
| Inference | `scripts.inference.run_policy` → `src.interfaces.inference_cli` → `src.exploratory.pilot.evaluate_episode` | The stable inference route still uses the exploratory/demo evaluator and only selects `straight-sprint-v1`. This is a real legacy boundary to resolve, although the command remains callable. |

The frontend is explicitly a visual editor/replay sandbox. Its README disclaims authoritative physics/evaluation and a Python bridge; this is an intentional boundary, not a broken connection in the production pipeline.

## Findings

Priority is based on architecture and maintenance risk. No Critical or High runtime failure was confirmed in this audit.

### Medium

1. **Campaign and evaluation modules are oversized and import private helpers across a mutual application boundary.** `src/interfaces/campaign.py` (about 1,237 lines) owns contract preparation, preflight, child process execution, campaign state/locking, validation, artifacts and reporting. `src/interfaces/evaluation.py` (about 781 lines) owns roster creation, resumability, episode orchestration, statistics and SVG reports; it imports private serialization and file helpers from `campaign`, while `campaign` defers imports of evaluation during campaign execution. This makes edits coordinate across large modules and relies on a deferred cross-import. **Impact:** high change coupling and harder isolated tests; no import failure was observed. **Refactor:** extract stable artifact/serialization primitives to an internal package, then split campaign lifecycle, run validation and report generation behind existing public functions. Remove private cross-module imports only after compatibility tests pin manifests, hashes, resume behavior and reports.

2. **WP3 and generic training have overlapping PPO lifecycle implementations with distinct authorization contracts.** `src/training/runner.py` serves generic training, while `src/interfaces/runner.py` (about 803 lines) implements the campaign worker's PPO lifecycle, telemetry, integrity evidence, cleanup and checkpoint handling. `src/interfaces/campaign.py` launches that worker through `scripts.training.local_simulation`. They are not safely interchangeable: the campaign runner checks a registered LOCAL_DEMO research profile and owns additional evidence contracts. **Impact:** fixes to resource handling, save/load or callback lifecycle can drift between paths; deleting the campaign runner would break the real WP3 pipeline. **Refactor:** first document a shared low-level PPO execution lifecycle with injected admission/telemetry policy; preserve current gates and artifact schemas, and migrate one path only with equivalent regression and evidence-contract coverage.

3. **The WP7 protocol module combines protocol identity with runtime construction and manifest capture.** `src/validation/wp7_protocol.py` imports track, physics, simulation, Gymnasium factory, PPO provenance and reward/navigation contracts, while defining registration loading, protocol construction/validation, scenario factories and runtime-manifest validation. `src/validation/wp7_admission.py` then composes protocol, track feasibility, source checks and owner review. **Impact:** a high-fanout validation hub couples schema/provenance checks to construction of many runtime domain objects, increasing the blast radius of protocol edits. **Refactor:** split pure registration/schema/digest validation from scenario factories and runtime capture, retaining the exact canonical serialization, digest inputs, profile names and admission decisions. Add hash/manifest compatibility fixtures before moving code.

4. **The public inference adapter is still backed by exploratory demo code.** `src/interfaces/inference_cli.py` constructs a local session and monkey-patches `env.step` for progress output, then calls `src.exploratory.pilot.evaluate_episode`; it uses a synthetic permit-shaped object and fixes the route to Straight. The shared evaluator already exists at `src/evaluation/evaluator.py`, but has a stricter metrics contract and is not a drop-in replacement without review. **Impact:** the stable `inference run-policy` route depends on an exploratory implementation and its legacy row schema; future inference support is difficult to generalize. **Refactor:** create a supported policy rollout API that accepts an explicit read-only env factory and returns a stable inference result, then adapt the CLI while retaining existing output compatibility or versioning it. Add tests for no-training, deterministic seed selection, invalid model/output and progress reporting before removing exploratory code.

### Low

1. **Names retain retired-demo terminology on active pipeline components.** `src/interfaces/local_demo_cli.py`, `src/interfaces/runner.py`, campaign docstrings, telemetry schemas and related tests still say `LOCAL_DEMO`/`local_demo`; the registered campaign worker calls these components today. **Impact:** contributors may mistake required campaign code for removable demo residue. **Refactor:** rename only in a separately planned compatibility migration. Treat telemetry/schema identifiers, frozen provenance and source-hash allow-lists as immutable unless a reviewed migration explicitly covers them; update user-facing module descriptions first.

2. **The package-level import graph has intentional deferred edges that obscure layering.** Static top-level import scanning found no confirmed runtime import-time cycle; the `src.rl.episode.progress` ↔ `route` relationship is mediated by type-only/lazy imports. Other cross-layer edges include `src.rl.algorithms.ppo` consulting validation/interface/exploratory admission helpers dynamically. **Impact:** boundaries are less clear and local import scans can miss behavior-affecting dependencies; no startup cycle was reproduced. **Refactor:** move authorization decisions to a policy object passed into the PPO adapter or a neutral admission boundary, preserving defaults and all current rejection paths. Add import-boundary checks only after defining allowed package directions.

## Inventory conclusions

- No confirmed dead production module or removable dependency was identified. Dynamic imports, CLI catalog entries, subprocess module names and pinned source inventories make name-only deletion unsafe.
- No Critical/High finding or reproducible runtime import cycle was found in this static pass. This is not proof of absence; dynamic imports need runtime/import tests.
- Existing workflow registry documentation correctly retains `training local-simulation` as an internal campaign worker and describes the supported public pipeline. The historical migration map now describes the current `src.interfaces` role; original `f1rl` names remain provenance.
- Frozen receipts, evidence, Physics, PPO algorithm behavior and scientific contracts were not edited.

## Recommended refactor order

1. Add/confirm compatibility tests for WP3 campaign manifests, resume behavior, telemetry cleanup, admission decisions, evaluation reports and inference output; preserve immutable-source checks.
2. Extract campaign/evaluation shared file primitives and split those large application modules without changing serialized output.
3. Define a shared PPO execution lifecycle with separate generic and registered-campaign admission/telemetry policies; converge duplicated runner mechanics gradually.
4. Replace exploratory-backed inference with a supported read-only policy rollout API and explicit output contract.
5. Separate WP7 pure protocol validation from runtime factories/capture, then clarify legacy `LOCAL_DEMO` naming only where it does not alter frozen identifiers.

Each stage needs a narrow implementation plan and must be independently regression-tested. This report identifies work; it does not authorize or claim those architecture changes are complete.

## Verification and limits

The audit used source/config/docs inspection and static import/reference scans. No source behavior was changed. Verification passed: targeted pipeline/legacy regression (**20 passed**), registry/documentation regression (**3 passed**), import and workflow registry assertions, help dispatch for campaign/local-simulation/evaluation/inference, `compileall -q src scripts`, YAML parsing (**11 files**), frozen assets (**12 models, 7 reports, 96 cell logs**), and `git diff --check`. Full pytest and frontend checks were not repeated because this update changes documentation only. Scientific training/HELD-OUT, Actions and independent visual frontend review were not run; training/HELD-OUT and Actions are explicitly out of scope.
