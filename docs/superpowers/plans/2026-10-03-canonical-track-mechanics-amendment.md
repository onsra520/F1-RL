# Canonical Track Mechanics Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Amend the canonical six-track corpus so boost and slow-zone mechanics are exercised by both training and held-out tracks without changing the geometry-generalization research question.

**Architecture:** Keep all physics algorithms and WP1B environment values unchanged. Modify exactly four WP2 track assets, recompute their existing semantic hashes with production track-asset code, then re-run WP3 split/leakage/fairness contracts and prove the actual canonical assets exercise boost/slow dynamics.

**Tech Stack:** Python 3.12, immutable sandbox/track contracts, JSON track assets, SHA-256 semantic fingerprints, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-canonical-track-mechanics-amendment-design.md`

## Global Constraints

- Branch: `research/track-mechanics-amendment`.
- Base HEAD: `7dc91cf900ed51ca2861d7b64bccb5998db4f0ec`.
- Do not change production physics algorithms.
- Do not change WP1B vehicle, simulation, observation, reward, action, or episode values.
- Keep the WP3 split IDs exactly 4 training + 2 held-out.
- Modify exactly four canonical track JSON files: `slalom-v1`, `right-turn-v1`, `left-turn-v1`, and `offset-slalom-v1`.
- `straight-sprint-v1` and `ramp-v1` must remain byte/semantic unchanged.
- Canonical boost: id `boost-0`, `acceleration_bonus = 3.0`.
- Canonical slow zone: id `slow-0`, `drag_multiplier = 2.0`.
- No telemetry, evaluator, pilot training, seed selection, or training-budget work belongs in this amendment.
- No production mocks/fakes/test doubles.
- No new direct production SB3 or PyTorch imports.

## Review Focus

1. A modifier volume with the right JSON shape but wrong coordinates/extents must fail canonical-corpus tests.
2. Updating one asset without updating both semantic hashes in the manifest must fail manifest integrity.
3. Held-out mechanic classes must remain a subset of training mechanic classes; no unseen mechanic may enter held-out evaluation.
4. Modifier entities must be physically exercisable on valid grounded road regions and must not overlap blockers/finish volumes.
5. The amendment must not drift any unchanged track, WP1B protocol value, split ID, or production physics file.

---

### Task 1: Pin and author the amended canonical assets atomically

**Files:**
- Modify: `tests/integration/test_track_asset_corpus.py`
- Modify: `assets/tracks/training/slalom-v1.json`
- Modify: `assets/tracks/training/right-turn-v1.json`
- Modify: `assets/tracks/evaluation/left-turn-v1.json`
- Modify: `assets/tracks/evaluation/offset-slalom-v1.json`
- Modify: `assets/tracks/manifest.json`

**Interfaces:**
- Consumes: existing `load_track_asset(path) -> TrackAsset`, `load_track_manifest(path) -> TrackManifest`.
- Produces: four amended canonical assets plus a manifest whose `world_sha256` and `content_sha256` match production semantic hashing.

- [ ] **Step 1: Add failing exact-corpus assertions**

Extend `test_track_asset_corpus.py` to assert exact modifier contracts:

```text
slalom-v1
  boost-0 center=(0,1,36) half_extents=(7,1,4) orientation=identity bonus=3.0
  slow_zones=()

offset-slalom-v1
  boost-0 center=(0,1,36) half_extents=(8,1,4) orientation=identity bonus=3.0
  slow_zones=()

right-turn-v1
  slow-0 center=(0,1,50) half_extents=(6,1,8) orientation=identity multiplier=2.0
  boost_pads=()

left-turn-v1
  slow-0 center=(0,1,50) half_extents=(6,1,8) orientation=identity multiplier=2.0
  boost_pads=()

straight-sprint-v1 and ramp-v1
  boost_pads=()
  slow_zones=()
```

Also pin unchanged hashes exactly:

```text
straight-sprint-v1
  world   685f273f62bdc35399d73be5db922198022ad2b549623924dbeb8171d4a4088b
  content 84adafbebb491af7bfbdaa9ef91e0c741023af2c5910c375cb3ce8d5f4e62070

ramp-v1
  world   775f795aa2763f64916e68e6a64fc763021e875c384050a974c747b6b4907978
  content 0fbf4b4633926944759c1972ecc15c6011785dc4d37d46f9d62c4c7d3a12c9b4
```
- [ ] **Step 2: Run the corpus test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_track_asset_corpus.py -q
```

Expected: failures because all four modifier arrays are currently empty.

- [ ] **Step 3: Edit the four JSON assets with the exact spec entities**

Add only the pinned `boost_pads` / `slow_zones` entries. Do not alter surfaces, obstacles, spawn, checkpoints, finish lines, score pickups, track IDs, or formatting of unrelated values.

- [ ] **Step 4: Recompute semantic hashes with production code**

Use a one-off Python command/script that imports `load_track_asset` from `f1rl.track_assets` and prints `track_id`, `world_sha256`, and `content_sha256` for the four modified files.

Do not hand-calculate hashes and do not add a production hash-generation helper.

- [ ] **Step 5: Update only the four corresponding manifest hash pairs**

Keep all manifest IDs and paths unchanged. Keep the straight-sprint and ramp manifest entries byte-for-byte equivalent in semantic content.

- [ ] **Step 6: Run canonical corpus + manifest integrity tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_track_asset_corpus.py tests/track_assets/test_manifest.py -q
```

Expected: PASS.

- [ ] **Step 7: Verify only the intended asset files changed**

Before commit, inspect both committed-base and working-tree scope:

```powershell
git diff --name-only 7dc91cf900ed51ca2861d7b64bccb5998db4f0ec -- assets/tracks
git diff --check
```

Require that no canonical JSON other than the four approved files plus `manifest.json` is modified. `assets/tracks/README.md` is not part of Task 1 and should remain untouched until Task 3.

- [ ] **Step 8: Commit**

```bash
git add tests/integration/test_track_asset_corpus.py assets/tracks/training/slalom-v1.json assets/tracks/training/right-turn-v1.json assets/tracks/evaluation/left-turn-v1.json assets/tracks/evaluation/offset-slalom-v1.json assets/tracks/manifest.json
git commit -m "feat(track): add canonical boost and slow mechanics"
```

### Task 2: Prove canonical modifier placement is physically valid and active

**Files:**
- Modify: `tests/integration/test_track_asset_corpus.py`
- Create: `tests/integration/test_canonical_track_mechanics.py`

**Interfaces:**
- Consumes: Task 1 amended assets, existing `TrackWorld.query_point(...)`, `step_world(...)`, `VehicleState`, `VehicleControl`, and WP1B `VehicleParameters` / `SimulationConfig`.
- Produces: integration proof that authored modifier zones lie on valid grounded road regions and measurably affect vehicle dynamics without physics-code changes.

- [ ] **Step 1: Add failing/guarding placement tests against the canonical assets**

For each modifier zone, assert:
- the zone center resolves to a valid surface contact under WP1B `max_surface_distance = 0.75`;
- a canonical grounded vehicle reference point at the zone center is inside the intended modifier query;
- modifier volumes do not overlap any obstacle volume at their centers/boundaries used by the canonical design;
- modifier volumes do not overlap the finish line;
- all existing `RaceRoute.from_track(asset.world)` calls remain valid.

Use existing volume/query APIs; do not duplicate a general OBB-intersection library solely for this amendment. Pin the specifically designed separations from the spec where full-volume overlap APIs do not exist.
- [ ] **Step 2: Add the canonical boost dynamics proof**

In `test_canonical_track_mechanics.py`, load `slalom-v1` and create a grounded vehicle state inside `boost-0` with a finite forward speed below the 15 m/s cap.

Step the canonical world once with a fixed steering/throttle control.

Construct the comparison world with `dataclasses.replace(asset.world, boost_pads=())`, step the identical state/control once, and assert:

```python
boosted.state.vehicle.speed > baseline.state.vehicle.speed
boosted.modifiers.longitudinal_acceleration_bonus == 3.0
baseline.modifiers.longitudinal_acceleration_bonus == 0.0
```

- [ ] **Step 3: Add the canonical slow-zone dynamics proof**

Load `right-turn-v1`, place an identical grounded moving state inside `slow-0`, and compare one canonical step with `dataclasses.replace(asset.world, slow_zones=())`.

Assert:

```python
slowed.state.vehicle.speed < baseline.state.vehicle.speed
slowed.modifiers.rolling_drag_multiplier == 2.0
baseline.modifiers.rolling_drag_multiplier == 1.0
```

Use the exact WP1B vehicle/simulation numeric contract in the integration fixture; do not invent alternate research parameters.

- [ ] **Step 4: Run Task 2 tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_track_asset_corpus.py tests/integration/test_canonical_track_mechanics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_track_asset_corpus.py tests/integration/test_canonical_track_mechanics.py
git commit -m "test(track): verify canonical modifier dynamics"
```

### Task 3: Revalidate WP3 scientific compatibility and document the amended corpus

**Files:**
- Modify: `tests/integration/test_experiment_track_split_integration.py`
- Modify: `assets/tracks/README.md`
- Modify: `configs/experiments/README.md`

**Interfaces:**
- Consumes: Task 1 updated manifest/assets and existing `resolve_track_split()`.
- Produces: explicit experiment-level proof that held-out mechanic classes are not novel and all WP3 invariants still hold.

- [ ] **Step 1: Add mechanic-exposure assertions to WP3 integration**

Derive mechanic-class presence from each resolved `TrackWorld`, not from file paths.

Pin:

```text
training union:
  obstacle  = present
  boost     = present
  slow      = present
  elevation = present

held-out union:
  obstacle  = present
  boost     = present
  slow      = present

held-out modifier/event mechanic classes subset of training mechanic classes

Define `elevation` for this corpus test as any surface whose forward vector has a non-zero vertical component (`abs(surface.forward.y) > 1e-12`). This pins the existing ramp representation without introducing a new production mechanic registry.
```

Do not require held-out elevation because v1 does not test elevation generalization.

- [ ] **Step 2: Re-run existing WP3 split/fairness assertions**

The test must still prove:
- exact 4 + 2 IDs unchanged;
- six unique `world_sha256` values;
- manifest integrity;
- canonical initial state fairness;
- all factories share the same WP1B non-track contract;
- action/observation spaces remain compatible.

- [ ] **Step 3: Update corpus documentation**

Document:
- which two tracks expose boost;
- which two expose slow;
- straight-sprint/ramp remain control tracks;
- held-out mechanics are intentionally known from training;
- the research question remains geometry generalization.

Do not add WP4 telemetry details beyond noting that the corpus is now ready for telemetry work.

- [ ] **Step 4: Run WP3 compatibility tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/experiments tests/integration/test_track_asset_corpus.py tests/integration/test_experiment_track_split_integration.py tests/integration/test_canonical_track_mechanics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_experiment_track_split_integration.py assets/tracks/README.md configs/experiments/README.md
git commit -m "test(research): revalidate mechanics-aware track split"
```

### Task 4: Whole-amendment audit and exact verification

**Files:**
- Modify only if a regression test exposes a real defect.
- No production physics changes are expected.

**Interfaces:**
- Consumes: all amendment commits from Tasks 1–3.
- Produces: final verification evidence and a clean branch ready to integrate before WP4 resumes.

- [ ] **Step 1: Verify asset-change scope from the base SHA**

Require exactly these canonical asset-content changes under `assets/tracks`:
- four approved JSON files;
- `manifest.json`;
- documentation may change separately.

Assert `straight-sprint-v1.json` and `ramp-v1.json` have zero diff from base.

- [ ] **Step 2: Verify protocol/config no-drift**

From base `7dc91cf900ed51ca2861d7b64bccb5998db4f0ec`, require no changes to:
- `configs/experiments/ppo-generalization-v1.yml`;
- `config.yml`;
- production physics modules under `f1rl/sandbox/dynamics/`;
- `f1rl/learning/gym_environment.py`;
- training/PPO configuration.

- [ ] **Step 3: Run focused track + experiment tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/track_assets tests/integration/test_track_asset_corpus.py tests/integration/test_canonical_track_mechanics.py tests/integration/test_experiment_track_split_integration.py tests/learning/experiments -q
```

Expected: PASS.

- [ ] **Step 4: Run learning/orchestration regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning tests/integration/test_training_orchestration_integration.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full suite and static/dependency checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q f1rl scripts
.\.venv\Scripts\python.exe -m pip check
git diff --check 7dc91cf900ed51ca2861d7b64bccb5998db4f0ec...HEAD
```

Audit and require:
- direct Stable-Baselines3 production import owners remain exactly `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`;
- direct production `torch` imports remain zero;
- production `Mock`/`Fake` markers remain zero;
- `track_assets` still does not import `learning`.
- [ ] **Step 6: Whole-amendment review**

Review the full diff from base with priority on:
1. accidental physics/protocol drift;
2. incorrect modifier placement or strength;
3. stale/incorrect manifest hashes;
4. held-out mechanic novelty;
5. changed split IDs or weakened leakage guarantees.

Any Critical/Important defect gets a dedicated regression test before repair.

- [ ] **Step 7: Final branch state**

Require:

```text
branch == research/track-mechanics-amendment
git status --porcelain == empty
all verification commands green
no merge/push
WP4 implementation not started
```

Stop at the amendment integration gate and present the branch for the user's merge/push decision. After integration, resume WP4 as a new branch/worktree from the amended main.
