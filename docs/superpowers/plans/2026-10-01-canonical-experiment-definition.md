# Canonical Experiment Definition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WP1A: a strict, immutable, draft-to-frozen canonical experiment definition without inventing research numeric defaults.

**Architecture:** Add a small `f1rl.learning.experiments` package that owns experiment DTOs, semantic-contract identifiers, YAML loading, relative-reference resolution, and the frozen-readiness gate. Keep the module explicit-only (do not re-export it from `f1rl.learning`) so base imports remain independent of optional PyYAML/SB3 dependencies. Add one checked-in draft experiment YAML that records unresolved scientific fields explicitly and references runtime/track inputs without implementing WP2+ behavior.

**Tech Stack:** Python 3.12-compatible stdlib dataclasses/pathlib/math, PyYAML 6.x from the existing `rl`/`dev` extras, existing F1-RL domain validators, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-canonical-experiment-definition-design.md`

## Global Constraints

- Do not promote numeric values from tests/benchmarks into the research protocol.
- Draft definitions may contain explicit unresolved environment fields; frozen definitions may not.
- Unknown YAML keys are rejected at every owned mapping level.
- Relative config/manifest references resolve against the experiment YAML directory.
- WP1A does not implement track loading, split logic, telemetry, evaluation, provenance, pilot training, or baseline training.
- Stable-Baselines3 production imports remain limited to `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`.
- No F1-RL production module directly imports PyTorch.
- Do not re-export the experiment package from `f1rl.learning`; importing `f1rl.learning` must remain usable without optional RL dependencies.
- Test doubles remain in dedicated test files only; no production mock/fake/test flags.

## Review Focus

- YAML boolean values passed where numeric/integer values are expected must fail rather than behaving as `0/1`.
- Non-finite numeric values (`nan`, `inf`) must fail in draft fields when provided.
- Duplicate rangefinder IDs and non-unit directions must fail through the real sandbox sensor contract.
- Relative references containing `..` must resolve deterministically from the experiment file's parent without changing process CWD.
- `status: frozen` must fail if environment fields, sensor suite, seed pairs, or frozen training references are incomplete/missing.

---

### Task 1: Define immutable experiment contracts

**Files:**
- Create: `f1rl/learning/experiments/__init__.py`
- Create: `f1rl/learning/experiments/definition.py`
- Create: `tests/learning/experiments/test_definition_contract.py`

**Interfaces:**
- Produces public explicit-package imports: `ExperimentDefinition`, `ExperimentMetadata`, `ExperimentSemantics`, `VehicleExperimentConfig`, `SimulationExperimentConfig`, `RangefinderExperimentSpec`, `ObservationExperimentConfig`, `RewardExperimentConfig`, `EpisodeExperimentConfig`, `TrainingExperimentConfig`, `SeedProtocol`, `TrackExperimentConfig`, `EvaluationExperimentConfig`.
- `ExperimentDefinition` owns all sections but does not construct tracks or training environments.

- [ ] **Step 1: Write failing immutability/type/semantic tests**

Cover frozen dataclasses, non-blank experiment ID, positive integer `protocol_version` with booleans rejected, status limited to `draft|frozen`, and exact V1 semantic identifiers:

```text
algorithm = ppo
policy = MlpPolicy
action = steering-throttle-brake0-v1
observation = rangefinders-local-navigation-v1
reward = target-progress-collision-time-v1
race = single-lap-ordered-checkpoints-v1
```

Reject unsupported semantic identifiers rather than turning WP1 into an algorithm/plugin registry.

- [ ] **Step 2: Run the new contract tests and verify RED**

Run: `python -m pytest tests/learning/experiments/test_definition_contract.py -q`

Expected: collection/import failure because `f1rl.learning.experiments` does not exist.

- [ ] **Step 3: Implement the metadata/semantics and section DTOs**

Use frozen/slots dataclasses. Research-critical environment scalars use `float | None` or `int | None` so draft state is explicit. `RangefinderExperimentSpec` is always complete when present: `id: str`, `local_direction: tuple[float, float, float]`, `max_distance: float`; validate it by constructing the existing `Vec3` + `RangefinderSpec` rather than duplicating unit-vector/range rules.

`SeedProtocol` stores `ppo: tuple[int, ...]` and `sampling: tuple[int, ...]`; each present seed must be an unsigned 32-bit integer and booleans must fail. Empty tuples remain legal at draft-contract construction time.

- [ ] **Step 4: Run contract tests and verify GREEN**

Run: `python -m pytest tests/learning/experiments/test_definition_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Commit message: `feat(research): define experiment contracts`

---

### Task 2: Add strict YAML loading and reference resolution

**Files:**
- Modify: `f1rl/learning/experiments/definition.py`
- Create: `tests/learning/experiments/test_experiment_yaml.py`

**Interfaces:**
- Produces: `ExperimentDefinition.from_yaml(path: str | Path) -> ExperimentDefinition`.
- Reference fields exposed as resolved `Path` values: `training.runtime_config`, `tracks.manifest`.

- [ ] **Step 1: Write failing strict-loader tests**

Require the root to contain exactly these mappings:

```text
experiment
semantics
environment
training
seeds
tracks
evaluation
```

Require explicit nested keys for all WP1-owned fields; reject unknown/missing section keys, malformed mappings/lists, booleans-as-numbers, non-finite numbers, malformed rangefinder vectors, duplicate rangefinder IDs, and invalid seed values.

Include a valid draft YAML with `null` numeric environment values, `rangefinders: []`, and empty seed arrays.

- [ ] **Step 2: Add failing path-resolution tests**

Given an experiment under `tmp_path / "configs/experiments/definition.yml"`, assert `../../config.yml` and `../../assets/tracks/manifest.json` resolve from that YAML directory to deterministic absolute/normalized `Path` values. Do not mutate CWD.

- [ ] **Step 3: Run YAML tests and verify RED**

Run: `python -m pytest tests/learning/experiments/test_experiment_yaml.py -q`

Expected: FAIL because `from_yaml()` is missing.

- [ ] **Step 4: Implement `ExperimentDefinition.from_yaml()`**

Use `yaml.safe_load`. Keep parsing helpers private and small. Every mapping owns an exact-key set; no `dict.get(..., hidden_default)` for research-critical fields. Resolve reference strings against `config_path.resolve().parent` after validating non-blank strings. Draft references may point to files that do not exist yet because WP2 has not produced the manifest.

- [ ] **Step 5: Run YAML + contract tests and verify GREEN**

Run: `python -m pytest tests/learning/experiments -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Commit message: `feat(research): load experiment definitions`

---

### Task 3: Implement the frozen-readiness gate

**Files:**
- Modify: `f1rl/learning/experiments/definition.py`
- Create: `tests/learning/experiments/test_frozen_experiment.py`

**Interfaces:**
- Produces: `ExperimentDefinition.require_frozen() -> None`.
- Consumes all DTOs from Tasks 1-2; it validates scientific readiness only and performs no track loading/training.

- [ ] **Step 1: Write failing readiness tests for draft and unresolved values**

Assert `require_frozen()` rejects `status != "frozen"`. For `status: frozen`, independently prove failure when any vehicle/simulation/observation/reward/episode scalar is `None`, when `rangefinders` is empty, when training `protocol_status != "frozen"`, when seed arrays are empty, have unequal lengths, or contain fewer than 3 replicate pairs, and when referenced runtime config or track manifest does not exist as a file.

- [ ] **Step 2: Write the passing complete-frozen fixture**

Use deliberately test-local numeric values (clearly marked fixture-only) to prove mechanics only. Assert a complete definition with >=3 paired seeds, non-empty valid rangefinders, existing referenced files, `training.protocol_status == "frozen"`, and `evaluation.deterministic_policy is True` passes. Do not copy these numbers to checked-in experiment YAML.

- [ ] **Step 3: Run readiness tests and verify RED**

Run: `python -m pytest tests/learning/experiments/test_frozen_experiment.py -q`

Expected: FAIL because `require_frozen()` is missing.

- [ ] **Step 4: Implement `require_frozen()`**

Return `None` on success and raise `ValueError` on any readiness violation. Check only WP1-owned readiness: do not inspect track content, split membership, dependency versions, Git state, model files, or evaluator outputs.

- [ ] **Step 5: Run all experiment tests and verify GREEN**

Run: `python -m pytest tests/learning/experiments -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Commit message: `feat(research): enforce frozen experiment readiness`

---

### Task 4: Add the canonical draft profile and documentation

**Files:**
- Create: `configs/experiments/ppo-generalization-v1.yml`
- Create: `configs/experiments/README.md`
- Modify: `configs/README.md`
- Create: `tests/learning/experiments/test_checked_in_experiment.py`

**Interfaces:**
- The checked-in YAML is intentionally `status: draft` and is loadable with `ExperimentDefinition.from_yaml(...)`.
- It must fail `require_frozen()` because WP2/WP1B/WP7 inputs are intentionally unresolved.

- [ ] **Step 1: Write failing checked-in-profile test**

Assert the canonical file loads with:

```text
id = ppo-generalization-v1
protocol_version = 1
status = draft
training.runtime_config -> repository root config.yml
tracks.manifest -> assets/tracks/manifest.json
```

Assert its explicit V1 semantics match Task 1 and its research-critical unresolved fields remain `None`/empty. Assert `require_frozen()` raises rather than filling defaults.

- [ ] **Step 2: Run the checked-in-profile test and verify RED**

Run: `python -m pytest tests/learning/experiments/test_checked_in_experiment.py -q`

Expected: FAIL because the canonical YAML does not exist.

- [ ] **Step 3: Add the draft YAML and config documentation**

Create the candidate structure from the approved spec plus the explicit `semantics` mapping pinned in Task 1. Keep every unresolved scientific numeric field `null`, rangefinders/seeds empty, `training.runtime_config: ../../config.yml`, `training.protocol_status: pilot_owned`, `tracks.manifest: ../../assets/tracks/manifest.json`, and `evaluation.deterministic_policy: true`.

Document that test/benchmark numbers are not research defaults, WP2 supplies track scale/manifest, WP1B resolves environment numbers, and WP7 freezes the final training profile.

- [ ] **Step 4: Run experiment tests and verify GREEN**

Run: `python -m pytest tests/learning/experiments -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Commit message: `docs(research): add draft generalization profile`

---

### Task 5: Whole-WP1A verification and scope audit

**Files:**
- No intended production feature expansion beyond Tasks 1-4.
- Fix only WP1A defects found by verification/review, with regression tests first.

**Interfaces:**
- Final public surface remains explicit: `from f1rl.learning.experiments import ExperimentDefinition, ...`.
- `f1rl.learning` itself remains free of experiment/PyYAML auto-imports.

- [ ] **Step 1: Run focused experiment tests**

Run: `python -m pytest tests/learning/experiments -q`

Expected: all WP1A tests pass.

- [ ] **Step 2: Run relevant learning/training regressions**

Run: `python -m pytest tests/learning tests/integration/test_training_orchestration_integration.py -q`

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run: `python -m pytest -q`

Expected: PASS with no regressions.

- [ ] **Step 4: Run static/runtime checks**

Run:

```text
python -m compileall -q f1rl scripts
python -m pip check
git diff --check
```

Expected: all exit 0; pip reports no broken requirements.

- [ ] **Step 5: Re-run dependency-boundary scans**

Verify production direct SB3 imports remain exactly the two existing owner files, direct production Torch imports remain zero, and production Mock/Fake/test-only markers remain zero.

- [ ] **Step 6: Review the complete WP1A diff against the spec**

Specifically reject accidental track loader/schema work, split/evaluation/provenance code, numeric research defaults copied from tests, new base dependency requirements, or hidden defaults in YAML parsing.

- [ ] **Step 7: Fix any Critical/Important review finding with TDD, then repeat the affected verification**

Do not waive a failing gate.

- [ ] **Step 8: Commit any review-only fix separately**

Use a narrow `fix(research): ...` commit only if verification/review required a repair.

- [ ] **Step 9: Stop at the WP1A gate**

Report exact commits, final HEAD, files changed, focused/full tests, compileall, pip check, diff check, dependency boundaries, and remaining deferred WP1B decisions. Do not start WP2 or WP1B automatically.
