# Train/Held-Out Split and Leakage Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical PPO generalization experiment declare and validate a reproducible 4-training/2-held-out track split with hash-based leakage protection.

**Architecture:** Extend the existing experiment definition with split membership, then resolve those IDs against the already-verified WP2 manifest in a dedicated `learning.experiments.split` module. Reuse the existing track manifest, environment factory, canonical initial-state builder, and scenario equality contracts rather than introducing another dataset/scenario framework.

**Tech Stack:** Python 3.12, dataclasses, PyYAML, pytest, existing `f1rl.track_assets` and `f1rl.learning` contracts.

**Spec:** `docs/superpowers/specs/2026-10-02-train-heldout-split-design.md`

## Global Constraints

- Branch: `research/train-heldout-split`.
- Base: `a76d50f93a6c63478f507f4d653b683a9e9d6174`.
- Canonical split: four training IDs and two held-out IDs exactly as specified.
- `assets/tracks/manifest.json` remains role-free.
- The six WP2 track JSON files and their hashes must not change.
- Leakage identity is `world_sha256`; `content_sha256` is not sufficient.
- Experiment remains `status: draft`; training remains `pilot_owned`; seed arrays remain empty.
- No new production SB3 or PyTorch imports.
- No production mock/fake/test-double code.

## Review Focus

1. Same geometry under different IDs or spawns must still fail because `world_sha256` collides.
2. Asset directory names must never override experiment membership; a training ID may resolve an asset stored under an `evaluation/` path and vice versa.
3. Manifest entries not selected by either partition must remain legal and must not be silently added to the experiment.
4. Draft lifecycle must remain intact even after the split is fully resolvable.
5. Real scenario construction must prove that only track/spawn-derived state varies while the WP1B environment contract stays identical.

---

### Task 1: Extend the experiment split schema and canonical declaration

**Files:**
- Modify: `f1rl/learning/experiments/definition.py`
- Modify: `configs/experiments/ppo-generalization-v1.yml`
- Modify: `tests/learning/experiments/test_definition_contract.py`
- Modify: `tests/learning/experiments/test_experiment_yaml.py`
- Modify: `tests/learning/experiments/test_checked_in_experiment.py`
- Modify: `tests/learning/experiments/test_experiment_boundaries.py`
- Modify: `tests/learning/experiments/test_frozen_experiment.py`

**Interfaces:**
- Consumes: existing `TrackExperimentConfig(manifest: Path)` and `ExperimentDefinition.from_yaml(path)`.
- Produces: `TrackExperimentConfig(manifest: Path, training_ids: tuple[str, ...], heldout_ids: tuple[str, ...])` plus the checked-in canonical 4+2 declaration.

- [ ] **Step 1: Add failing DTO validation tests**

Pin non-empty tuple requirements, non-blank string IDs, uniqueness within each partition, cross-partition disjointness, and declaration-order preservation.

Example assertions should include:

```python
config = TrackExperimentConfig(
    manifest=Path("assets/tracks/manifest.json"),
    training_ids=("train-a", "train-b"),
    heldout_ids=("eval-a",),
)
assert config.training_ids == ("train-a", "train-b")
assert config.heldout_ids == ("eval-a",)
```

Reject `("",)`, whitespace-only IDs, duplicate IDs, empty partitions, lists passed directly to the DTO, and overlap such as training `("same",)` with held-out `("same",)`.

- [ ] **Step 2: Run DTO tests to verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments/test_definition_contract.py -q
```

Expected: failures because the current DTO has only `manifest`.

- [ ] **Step 3: Extend `TrackExperimentConfig` minimally**

Keep it frozen/slots. Add exact validation from the spec without hard-coding a 4+2 count.

- [ ] **Step 4: Add failing YAML parser tests**

Pin the exact `tracks` mapping keys to:
```text
manifest
training_ids
heldout_ids
```

Tests must reject missing keys, unknown keys, non-list YAML values, blank IDs, duplicate IDs, and overlap. One test must prove YAML list order becomes tuple order unchanged.

- [ ] **Step 5: Run YAML tests to verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments/test_experiment_yaml.py -q
```

Expected: failures because `from_yaml()` only accepts `tracks.manifest`.

- [ ] **Step 6: Update `ExperimentDefinition.from_yaml()`**

Use the existing strict mapping/list helpers; convert the two YAML lists to tuples only at the loader boundary. Do not add defaults.

- [ ] **Step 7: Pin the checked-in 4+2 declaration and update existing constructor fixtures**

Update the canonical YAML with the exact approved IDs. Update `test_checked_in_experiment.py` to assert the exact tuples.

Every existing direct `TrackExperimentConfig(...)` fixture in `test_experiment_boundaries.py` and `test_frozen_experiment.py` must receive structurally valid fixture split IDs so Task 1 does not leave unrelated experiment tests broken. Do not yet change `require_frozen()` behavior.

- [ ] **Step 8: Run the whole experiment suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments -q
```

Expected: PASS, while the canonical profile remains `draft`.

- [ ] **Step 9: Commit**

```bash
git add f1rl/learning/experiments/definition.py configs/experiments/ppo-generalization-v1.yml tests/learning/experiments
git commit -m "feat(research): define canonical track split"
```

### Task 2: Resolve split membership and enforce leakage guards

**Files:**
- Create: `f1rl/learning/experiments/split.py`
- Modify: `f1rl/learning/experiments/__init__.py`
- Create: `tests/learning/experiments/test_track_split.py`

**Interfaces:**
- Consumes: `TrackExperimentConfig`, `load_track_manifest(path) -> TrackManifest`.
- Produces:
  - `ResolvedTrackSplit(training: tuple[TrackManifestEntry, ...], heldout: tuple[TrackManifestEntry, ...])`
  - `resolve_track_split(config: TrackExperimentConfig) -> ResolvedTrackSplit`

- [ ] **Step 1: Write failing resolver tests**

Cover:
- valid training/held-out resolution with preserved declaration order;
- declared ID missing from manifest;
- unselected manifest entries are ignored but legal;
- membership follows declared IDs even when manifest paths use the opposite `training/` or `evaluation/` directory name;
- duplicate `world_sha256` across partitions;
- duplicate `world_sha256` within training;
- duplicate `world_sha256` within held-out;
- same world hash with different track IDs;
- same world hash with different content hashes/spawns.

- [ ] **Step 2: Run resolver tests to verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments/test_track_split.py -q
```

Expected: import/module failures because `split.py` does not exist.

- [ ] **Step 3: Implement immutable split resolver**

Implementation constraints:
- build one manifest lookup by `track_id`;
- resolve IDs in experiment order;
- fail on absent IDs;
- concatenate selected training + held-out entries and require global uniqueness of `world_sha256`;
- do not mutate the manifest or infer roles from paths;
- do not compare `content_sha256` for leakage.

- [ ] **Step 4: Export the public split API**

Add `ResolvedTrackSplit` and `resolve_track_split` to `f1rl.learning.experiments` only. Do not re-export them from `f1rl.learning`.

- [ ] **Step 5: Run Task 2 tests**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments/test_track_split.py tests/learning/experiments/test_experiment_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add f1rl/learning/experiments/split.py f1rl/learning/experiments/__init__.py tests/learning/experiments/test_track_split.py
git commit -m "feat(research): enforce track split leakage guards"
```

### Task 3: Integrate the split into the frozen lifecycle and document usage

**Files:**
- Modify: `configs/experiments/README.md`
- Modify: `assets/tracks/README.md`
- Modify: `f1rl/learning/experiments/definition.py`
- Modify: `tests/learning/experiments/test_frozen_experiment.py`

**Interfaces:**
- Consumes: Task 1 split fields and canonical declaration plus Task 2 `resolve_track_split()`.
- Produces: future `require_frozen()` split validation and documented held-out usage rules.

- [ ] **Step 1: Replace the frozen-valid fixture manifest with real temporary track assets**

Update `_complete(tmp_path)` in `test_frozen_experiment.py` so its manifest is valid under the WP2 production loader and contains one training asset plus one held-out asset with distinct world hashes. Keep these fixtures test-owned and minimal.

- [ ] **Step 2: Add failing frozen-lifecycle split tests**

Prove:
- a frozen otherwise-valid experiment passes with a resolvable disjoint split;
- a declared missing manifest ID fails;
- duplicate selected world geometry fails;
- the canonical checked-in experiment still fails first on `status != frozen`.

- [ ] **Step 3: Run frozen lifecycle tests to verify RED**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments/test_frozen_experiment.py tests/learning/experiments/test_checked_in_experiment.py -q
```

Expected: split-specific frozen tests fail because `require_frozen()` does not yet call the resolver.

- [ ] **Step 4: Integrate split resolution into `require_frozen()`**

After existing manifest existence checks, call `resolve_track_split(self.tracks)`. Keep earlier validation order stable so draft status, unresolved fields, training status, and seeds still fail for their existing reasons.

Use a local import only if needed to avoid a definition/split import cycle; do not create another abstraction layer.

- [ ] **Step 5: Update experiment/track documentation**

Document that canonical YAML membership is authoritative, directory names are organizational, WP7 may use training IDs only, and held-out results must not tune protocol v1.

- [ ] **Step 6: Run the whole experiment suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/experiments/README.md assets/tracks/README.md f1rl/learning/experiments/definition.py tests/learning/experiments/test_frozen_experiment.py
git commit -m "feat(research): validate frozen track split"
```

### Task 4: Prove the real six-track split is fair and contract-compatible

**Files:**
- Modify: `tests/integration/test_track_asset_corpus.py`
- Create: `tests/integration/test_experiment_track_split_integration.py`

**Interfaces:**
- Consumes: canonical `ExperimentDefinition`, `resolve_track_split()`, `load_track_asset()`, `build_initial_world_state()`, `F1RLEnvFactory`, and `TrainingScenarioSet`.
- Produces: integration proof only; no new production API.

- [ ] **Step 1: Add corpus split-integrity tests**

Using the checked-in experiment and manifest, assert:
- exactly the approved 4 training IDs and 2 held-out IDs resolve;
- all six selected `world_sha256` values are distinct;
- all six entries still match WP2 manifest integrity;
- no selected path is inferred by directory name.

- [ ] **Step 2: Add canonical initial-state fairness tests**

For every resolved entry, load the asset and call:

```python
build_initial_world_state(
    asset,
    max_surface_distance=definition.simulation.max_surface_distance,
)
```

Assert spawn pose is preserved, linear/angular velocity are zero, score/interactions/collected state are empty, and `grounded is True` for all six canonical tracks.

- [ ] **Step 3: Add a test-only real factory constructor**

Inside `test_experiment_track_split_integration.py`, construct the WP1B domain values directly from `ExperimentDefinition` using existing production classes:

- `VehicleParameters`;
- `SimulationConfig`;
- `RangefinderSpec` + `SensorSuite`;
- `ObservationConfig` + `ObservationEncoder`;
- `RewardComposer(TargetProgressReward, CollisionPenalty, TimePenalty)`;
- `EpisodeConfig`.

The helper may assemble these values for tests, but no equivalent production builder is added in WP3.

- [ ] **Step 4: Prove six factories share the non-track contract**

Build one `F1RLEnvFactory` per selected asset. Then construct one `TrainingScenarioSet` from all six factories and assert successful construction plus compatible action/observation spaces.

Also assert pairwise equality of:
- `vehicle_parameters`;
- `simulation_config`;
- `observation_encoder`;
- `reward_composer`;
- `episode_config`.

Assert the track objects and spawn-derived initial states are not required to be equal.

- [ ] **Step 5: Run integration tests**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_track_asset_corpus.py tests/integration/test_experiment_track_split_integration.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_track_asset_corpus.py tests/integration/test_experiment_track_split_integration.py
git commit -m "test(research): verify split environment fairness"
```

### Task 5: Boundary audit and exact final verification

**Files:**
- Modify only if a regression test exposes a real defect.
- No production changes are expected solely for this task.

**Interfaces:**
- Consumes: all WP3 implementation from Tasks 1–4.
- Produces: verification evidence for the final WP3 branch head.

- [ ] **Step 1: Run focused WP3 tests**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments tests/integration/test_track_asset_corpus.py tests/integration/test_experiment_track_split_integration.py -q
```

Expected: PASS.

- [ ] **Step 2: Verify WP2 assets and WP1B environment values did not drift**

Use Git diff from base `a76d50f93a6c63478f507f4d653b683a9e9d6174`.

Required:
- no changes under the six `assets/tracks/**/*.json` files or `assets/tracks/manifest.json`;
- no changes to WP1B environment numeric values in the canonical experiment except insertion of the split fields;
- no changes to `config.yml`.

- [ ] **Step 3: Run learning/orchestration regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning tests/integration/test_training_orchestration_integration.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the full suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass; platform-supported skips are allowed only if already understood.

- [ ] **Step 5: Run static/dependency checks**

Run:
```powershell
.\.venv\Scripts\python.exe -m compileall -q f1rl scripts
.\.venv\Scripts\python.exe -m pip check
git diff --check a76d50f93a6c63478f507f4d653b683a9e9d6174...HEAD
```

Audit production source and require:
- direct Stable-Baselines3 import owners remain exactly `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`;
- direct production `torch` imports remain zero;
- production `Mock`/`Fake` markers remain zero;
- `f1rl.learning` does not auto-export the experiment/split API;
- `track_assets` still does not import `learning`.

- [ ] **Step 6: Whole-WP3 review**

Review the full diff from the base SHA with priority on:
1. ID/hash leakage bypasses;
2. hidden path-based split inference;
3. held-out data accidentally entering training APIs/config;
4. lifecycle prematurely becoming frozen;
5. test-only factory helpers leaking into production.

Any Critical/Important defect gets a dedicated regression test before repair.

- [ ] **Step 7: Commit any verification-only regression fix if needed**

Do not make cleanup/refactor changes merely because the review is occurring.

- [ ] **Step 8: Final branch state**

Require:
```text
git status --porcelain == empty
branch == research/train-heldout-split
all verification commands green
no merge to main
```

Stop at the WP3 integration gate and present the branch for the user's merge/push decision. Do not begin WP4 in the same implementation run.
