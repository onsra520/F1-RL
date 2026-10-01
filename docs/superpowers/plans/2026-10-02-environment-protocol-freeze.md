# Environment Protocol Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WP1B by resolving and pinning the approved protocol-v1 environment values in the canonical experiment profile, while preserving the overall experiment as an intentionally incomplete draft until later training/seed decisions.

**Architecture:** Do not add a new production subsystem. Reuse the existing strict `ExperimentDefinition` loader and existing sandbox/runtime/learning configuration classes as the contract. WP1B changes only the canonical experiment YAML, focused regression tests, and experiment documentation; existing domain implementations remain untouched.

**Tech Stack:** Python 3.11+, PyYAML through the existing dev/rl extra, existing F1-RL domain/runtime dataclasses, pytest, GitHub Actions only as temporary verification infrastructure if execution needs remote RED/GREEN evidence.

**Spec:** `docs/superpowers/specs/2026-10-01-environment-protocol-freeze-design.md`

## Global Constraints

- Keep `experiment.status: draft`.
- Keep `training.protocol_status: pilot_owned`.
- Keep `seeds.ppo: []` and `seeds.sampling: []`; WP1B must not invent replicate seeds.
- Do not modify `config.yml`, PPO hyperparameters, training budget, vector backend, or `n_envs`.
- Do not declare or enforce WP3 scientific train/held-out split semantics.
- Do not modify `assets/tracks/**`; the WP2 manifest and all six track hashes must remain byte-identical to base `7aafb5f9eb57a07898b05b68aeeb50897b6e94da`.
- Do not change action, observation, reward, episode, or race implementation semantics.
- Do not modify `f1rl/learning/experiments/definition.py` unless verification exposes a genuine WP1B blocker; any such change requires a regression test first.
- No new production dependency, builder/service, mock/fake/test-only branch, or compatibility shim.
- Stable-Baselines3 production imports remain exactly the existing two owners; direct production Torch imports remain zero.
- The exact seven rangefinder IDs, declaration order, vectors, and 30 m ranges are protocol material.
- After WP1B, changing any pinned environment number/sensor declaration is a material protocol change requiring explicit review/versioning.

## Review Focus

1. A reviewer must not be able to mistake “environment frozen” for “experiment frozen”: exact environment values are resolved while status/training/seeds remain draft-owned — Tasks 1 and 3.
2. Sensor declaration order and exact floating-point vectors must survive YAML parsing unchanged; sorting/recomputation is not allowed — Tasks 1 and 2.
3. Resolved experiment values must instantiate the real `VehicleParameters`, `SimulationConfig`, `RangefinderSpec`/`SensorSuite`, `ObservationConfig`, reward terms/composer, and `EpisodeConfig` without test doubles — Task 2.
4. `require_frozen()` must still fail because of lifecycle/training/seeds, not because environment numerics or sensors remain unresolved — Task 3.
5. WP2 assets/hashes and architecture import boundaries must remain unchanged — Task 4.

---

## File Structure

Modify only:

```text
configs/experiments/ppo-generalization-v1.yml
configs/experiments/README.md
tests/learning/experiments/test_checked_in_experiment.py
```

Create one focused compatibility test only if keeping the checked-in-profile test focused improves readability:

```text
tests/learning/experiments/test_environment_protocol.py
```

No production Python file is expected to change.

---

### Task 1: Pin the Exact Environment Protocol in the Canonical Profile

**Files:**
- Modify: `tests/learning/experiments/test_checked_in_experiment.py`
- Modify: `configs/experiments/ppo-generalization-v1.yml`

**Interfaces:**
- Consumes: `ExperimentDefinition.from_yaml(PROFILE) -> ExperimentDefinition`.
- Produces: the same canonical YAML path with all WP1B-owned environment fields resolved and all later-stage fields still draft-owned.

- [ ] **Step 1: Replace the obsolete unresolved-environment regression with exact-value assertions**

Keep the existing metadata/semantic/reference test. Replace `test_checked_in_profile_keeps_research_values_unresolved` with tests asserting exact vehicle, simulation, observation, reward, and episode values:

```python
assert definition.vehicle == VehicleExperimentConfig(
    3.0, 0.5, 0.5, 6.0, 10.0, 0.4, 0.1, 15.0, 9.81
)
assert definition.simulation == SimulationExperimentConfig(0.1, 0.75)
assert definition.observation.velocity_scale == 15.0
assert definition.observation.target_distance_scale == 80.0
assert definition.reward == RewardExperimentConfig(1.0, 5.0, 0.1)
assert definition.episode == EpisodeExperimentConfig(500)
```

Assert the exact rangefinder tuple, in this order:

```text
left-30deg  (-0.5, 0.0, 0.8660254037844386)
left-15deg  (-0.25881904510252074, 0.0, 0.9659258262890683)
left-7p5deg (-0.13052619222005157, 0.0, 0.9914448613738104)
front       (0.0, 0.0, 1.0)
right-7p5deg (0.13052619222005157, 0.0, 0.9914448613738104)
right-15deg  (0.25881904510252074, 0.0, 0.9659258262890683)
right-30deg  (0.5, 0.0, 0.8660254037844386)
```

Every range is exactly `30.0`.

Also assert `status == "draft"`, `training.protocol_status == "pilot_owned"`, and both seed tuples remain empty.

- [ ] **Step 2: Run the checked-in-profile tests and verify RED**

Run:

```bash
python -m pytest tests/learning/experiments/test_checked_in_experiment.py -q
```

Expected: FAIL because the checked-in YAML still contains `null`/empty WP1A environment fields.

- [ ] **Step 3: Resolve only the YAML `environment` section**

Write the exact values and sensor vectors from the spec. Do not modify experiment metadata, semantic IDs, training section, seed arrays, track reference, or evaluation setting.

- [ ] **Step 4: Run the checked-in-profile tests and verify GREEN**

```bash
python -m pytest tests/learning/experiments/test_checked_in_experiment.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add configs/experiments/ppo-generalization-v1.yml tests/learning/experiments/test_checked_in_experiment.py
git commit -m "feat(research): freeze environment protocol"
```

---

### Task 2: Prove the Pinned Values Instantiate Real Environment Contracts

**Files:**
- Create: `tests/learning/experiments/test_environment_protocol.py` (preferred; use the checked-in-profile file instead only if it stays compact)
- No production implementation file should change.

**Interfaces:**
- Consumes loaded `ExperimentDefinition` values.
- Instantiates existing contracts only:
  - `VehicleParameters`
  - `SimulationConfig`
  - `RangefinderSpec` + `SensorSuite`
  - `ObservationConfig`
  - `TargetProgressReward`, `CollisionPenalty`, `TimePenalty`, `RewardComposer`
  - `EpisodeConfig`
- Produces no new public API.

- [ ] **Step 1: Write compatibility assertions using only real production types**

Load `configs/experiments/ppo-generalization-v1.yml`; construct:

```python
VehicleParameters(
    definition.vehicle.wheelbase,
    definition.vehicle.ride_height,
    definition.vehicle.max_steering_angle,
    definition.vehicle.max_acceleration,
    definition.vehicle.max_braking,
    definition.vehicle.rolling_drag,
    definition.vehicle.air_drag,
    definition.vehicle.max_ground_speed,
    definition.vehicle.gravity,
)
```

Do the analogous construction for `SimulationConfig`; construct `RangefinderSpec` values from every experiment sensor and then `SensorSuite`; construct `ObservationConfig`; construct the three existing reward terms inside `RewardComposer` in semantic order `target_progress`, `collision`, `time`; construct `EpisodeConfig(500)`.

Assert each constructed object exposes exactly the resolved values and `ObservationConfig.sensor_suite.rangefinders` preserves the seven declared IDs in order.

- [ ] **Step 2: Run the compatibility test**

```bash
python -m pytest tests/learning/experiments/test_environment_protocol.py -q
```

Expected after Task 1: PASS. This task is a compatibility proof over existing production contracts; no new product code is expected.

- [ ] **Step 3: Run the full experiment-contract suite**

```bash
python -m pytest tests/learning/experiments -q
```

Expected: PASS. Generic draft/frozen fixture tests remain valid and must not be rewritten to use the scientific protocol values.

- [ ] **Step 4: Commit Task 2**

```bash
git add tests/learning/experiments/test_environment_protocol.py
git commit -m "test(research): verify environment protocol compatibility"
```

If no new file was necessary, commit only the focused test changes with the same message.

---

### Task 3: Pin the WP1B Lifecycle Boundary and Update Documentation

**Files:**
- Modify: `tests/learning/experiments/test_checked_in_experiment.py`
- Modify: `configs/experiments/README.md`

**Interfaces:**
- `ExperimentDefinition.require_frozen() -> None` remains unchanged.
- The checked-in profile is environment-resolved but whole-experiment draft.

- [ ] **Step 1: Strengthen the lifecycle regression**

For the checked-in profile, first assert no WP1B-owned scalar is `None` and there are exactly seven rangefinders. Then assert:

```python
assert definition.metadata.status == "draft"
assert definition.training.protocol_status == "pilot_owned"
assert definition.seeds.ppo == ()
assert definition.seeds.sampling == ()
with pytest.raises(ValueError, match="experiment status must be frozen"):
    definition.require_frozen()
```

This error assertion pins why the current checked-in artifact is not whole-experiment frozen; it must not fail because WP1B environment fields are missing.

- [ ] **Step 2: Run lifecycle test**

```bash
python -m pytest tests/learning/experiments/test_checked_in_experiment.py -q
```

Expected: PASS after Task 1. No production change is expected.

- [ ] **Step 3: Update experiment README**

Replace WP1A-era wording that says environment values remain `null`. Document:

```text
WP1B environment protocol is pinned:
- vehicle/simulation/sensor/observation/reward/episode values resolved
- experiment remains draft
- training remains pilot_owned
- final seeds remain empty
- WP7 owns final training profile; later experiment freeze requires training + seed completion
- changing pinned WP1B values is a material protocol change
```

Do not describe `training/` / `evaluation/` asset directories as a finalized scientific split.

- [ ] **Step 4: Run focused experiment tests and commit**

```bash
python -m pytest tests/learning/experiments -q
git add configs/experiments/README.md tests/learning/experiments/test_checked_in_experiment.py
git commit -m "docs(research): document environment freeze lifecycle"
```

---

### Task 4: Whole-WP1B Review and Exact-Final Verification

**Files:**
- No intended feature expansion.
- Fix only Critical/Important WP1B defects found by review, with a regression test first.

**Interfaces:**
- Final implementation diff should contain configuration, tests, documentation, and the plan/spec only; no new production behavior is expected.

- [ ] **Step 1: Verify WP2 asset bytes were untouched**

Run:

```bash
git diff --exit-code 7aafb5f9eb57a07898b05b68aeeb50897b6e94da...HEAD -- assets/tracks
git diff --exit-code 7aafb5f9eb57a07898b05b68aeeb50897b6e94da...HEAD -- config.yml
```

Expected: no diff. The six manifest hashes therefore remain exactly the WP2 hashes.

- [ ] **Step 2: Run focused experiment and track-boundary tests**

```bash
python -m pytest tests/learning/experiments tests/track_assets/test_boundaries.py tests/integration/test_track_asset_corpus.py -q
```

Expected: PASS.

- [ ] **Step 3: Run learning/orchestration regressions**

```bash
python -m pytest tests/learning tests/integration/test_training_orchestration_integration.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the full/static gate**

```bash
python -m pytest -q
python -m compileall -q f1rl scripts
python -m pip check
git diff --check 30e724b35cddb197ed2cd76f0dd943ebaad8c995...HEAD
git status --short
```

Expected: tests pass, compilation succeeds, `pip check` reports no broken requirements, no whitespace errors, and the worktree is clean after commits.

- [ ] **Step 5: Re-run architecture boundary assertions**

Use the existing boundary tests rather than adding another scanner. Confirm SB3 production owners remain exactly:

```text
f1rl/learning/algorithms/ppo.py
f1rl/learning/training/vector_env.py
```

Direct production Torch imports remain zero; production Mock/Fake identifiers remain zero; base `f1rl.learning` still does not auto-import experiment/PyYAML API.

- [ ] **Step 6: Whole-WP1B review**

Review `30e724b35cddb197ed2cd76f0dd943ebaad8c995...HEAD` against the approved spec. Reject as Critical/Important:

- any changed WP2 track bytes/hashes;
- any invented seeds or premature whole-experiment `frozen` status;
- any training hyperparameter/runtime-default change;
- sensor reordering or recomputed/rounded vectors;
- production-code expansion not required to load the exact profile;
- accidental WP3 split, evaluator, provenance, or pilot-training work.

Any Important/Critical defect gets a focused regression test and separate `fix(research): ...` commit, then all affected gates are rerun.

- [ ] **Step 7: Exact-final rerun and stop**

Rerun Steps 1–5 on the exact final HEAD before reporting completion. Then report final HEAD, commits, focused/full test counts, asset/config diff guards, compile/pip/diff/boundary results, and stop before WP3.

---

## Stop Condition

After Task 4, **STOP at the WP1B verification gate**. Do not start WP3 automatically. The next reviewed stage is WP3 Train/Held-out Split + Leakage Guards.